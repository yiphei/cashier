import copy
import json
from typing import Set

from colorama import Style
from pydantic import BaseModel, ConfigDict, Field

from audio import get_speech_from_text
from function_call_context import FunctionCallContext, InexistentFunctionError
from graph import Direction, EdgeSchema, Graph, Node
from graph_data import cashier_graph_schema
from gui import MessageDisplay
from logger import logger
from model import Model
from model_tool_decorator import ToolRegistry
from model_turn import MessageList, TurnContainer
from model_util import CustomJSONEncoder, ModelProvider
from prompts.node_schema_selection import NodeSchemaSelectionPrompt
from prompts.off_topic import OffTopicPrompt


class ChatContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    remove_prev_tool_calls: bool
    curr_node: Node = None
    need_user_input: bool = True
    graph: Graph = Field(default_factory=Graph)
    next_edge_schemas: Set[EdgeSchema] = Field(default_factory=set)
    bwd_skip_edge_schemas: Set[EdgeSchema] = Field(default_factory=set)

    def init_node_core(
        self,
        node_schema,
        edge_schema,
        TC,
        input,
        last_msg,
        prev_node,
        direction,
        is_skip=False,
    ):
        logger.debug(
            f"[NODE_SCHEMA] Initializing node with {Style.BRIGHT}node_schema_id: {node_schema.id}{Style.NORMAL}"
        )
        new_node = node_schema.create_node(
            input, last_msg, prev_node, edge_schema, direction
        )

        TC.add_node_turn(
            new_node,
            remove_prev_tool_calls=self.remove_prev_tool_calls,
            is_skip=is_skip,
        )
        MessageDisplay.print_msg("system", new_node.prompt)

        if node_schema.first_turn and prev_node is None:
            TC.add_assistant_direct_turn(node_schema.first_turn)
            MessageDisplay.print_msg("assistant", node_schema.first_turn.msg_content)

        if edge_schema:
            self.graph.add_edge(self.curr_node, new_node, edge_schema, direction)

        self.curr_node = new_node
        self.next_edge_schemas = set(
            self.graph.graph_schema.from_node_schema_id_to_edge_schema.get(
                new_node.schema.id, []
            )
        )
        self.graph.add_bwd_skip_edge_schemas(self.curr_node, self.bwd_skip_edge_schemas)

    def init_next_node(self, node_schema, edge_schema, TC, input=None):
        if self.curr_node:
            self.curr_node.mark_as_completed()

        if input is None and edge_schema:
            input = edge_schema.new_input_from_state_fn(self.curr_node.state)

        if edge_schema:
            edge_schema, input = self.graph.compute_next_edge_schema(
                edge_schema, input, self.curr_node
            )
            node_schema = edge_schema.to_node_schema

        direction = Direction.FWD
        prev_node = self.graph.get_prev_node(edge_schema, direction)

        last_msg = TC.get_user_message(content_only=True)

        self.init_node_core(
            node_schema, edge_schema, TC, input, last_msg, prev_node, direction, False
        )

    def init_skip_node(
        self,
        node_schema,
        edge_schema,
        TC,
    ):
        direction = Direction.FWD
        if edge_schema and edge_schema.from_node_schema == node_schema:
            direction = Direction.BWD

        if direction == Direction.BWD:
            self.bwd_skip_edge_schemas.clear()

        prev_node = self.graph.get_prev_node(edge_schema, direction)
        input = prev_node.input

        last_msg = TC.get_asst_message(content_only=True)

        self.init_node_core(
            node_schema, edge_schema, TC, input, last_msg, prev_node, direction, True
        )


def should_skip_node_schema(model, TM, current_node_schema, all_node_schemas):
    if len(all_node_schemas) == 1:
        return None

    model_name = "claude-3.5"
    model_provider = Model.get_model_provider(model_name)
    node_conv_msgs = copy.deepcopy(
        TM.model_provider_to_message_manager[model_provider].node_conversation_dicts
    )
    last_customer_msg = node_conv_msgs.get_item_type_by_idx(
        MessageList.ItemType.USER, -1
    )
    prompt = OffTopicPrompt(
        node_prompt=current_node_schema.node_prompt,
        state_json_schema=current_node_schema.state_pydantic_model.model_json_schema(),
        tool_defs=json.dumps(
            ToolRegistry.get_tool_defs_from_names(
                current_node_schema.tool_fn_names,
                model_provider,
                current_node_schema.tool_registry,
            )
        ),
        last_customer_msg=last_customer_msg,
    )

    if model_provider == ModelProvider.ANTHROPIC:
        node_conv_msgs.append({"role": "user", "content": prompt})
    elif model_provider == ModelProvider.OPENAI:
        node_conv_msgs.append({"role": "system", "content": prompt})

    chat_completion = model.chat(
        model_name=model_name,
        message_dicts=node_conv_msgs,
        response_format=OffTopicPrompt.response_format,
        logprobs=True,
        temperature=0,
    )
    is_on_topic = chat_completion.get_message_prop("output")
    if model_provider == ModelProvider.OPENAI:
        prob = chat_completion.get_prob(-2)
        logger.debug(f"IS_ON_TOPIC: {is_on_topic} with {prob}")
    else:
        logger.debug(f"IS_ON_TOPIC: {is_on_topic}")

    if is_on_topic:
        return None

    node_conv_msgs.pop()

    prompt = NodeSchemaSelectionPrompt(
        all_node_schemas=all_node_schemas,
        model_provider=model_provider,
        last_customer_msg=last_customer_msg,
    )

    if model_provider == ModelProvider.ANTHROPIC:
        node_conv_msgs.append({"role": "user", "content": prompt})
    elif model_provider == ModelProvider.OPENAI:
        node_conv_msgs.append({"role": "system", "content": prompt})

    chat_completion = model.chat(
        model_name=model_name,
        message_dicts=node_conv_msgs,
        response_format=NodeSchemaSelectionPrompt.response_format,
        logprobs=True,
        temperature=0,
    )

    agent_id = chat_completion.get_message_prop("agent_id")
    if model_provider == ModelProvider.OPENAI:
        prob = chat_completion.get_prob(-2)
        logger.debug(f"AGENT_ID: {agent_id} with {prob}")
    else:
        logger.debug(f"AGENT_ID: {agent_id}")

    return agent_id if agent_id != current_node_schema.id else None


def handle_skip(model, TC, CT):
    fwd_skip_edge_schemas = CT.graph.compute_fwd_skip_edge_schemas(
        CT.curr_node, CT.next_edge_schemas
    )
    bwd_skip_edge_schemas = CT.bwd_skip_edge_schemas

    all_node_schemas = [CT.curr_node.schema]
    all_node_schemas += [edge.to_node_schema for edge in fwd_skip_edge_schemas]
    all_node_schemas += [edge.from_node_schema for edge in bwd_skip_edge_schemas]

    node_schema_id = should_skip_node_schema(
        model, TC, CT.curr_node.schema, all_node_schemas
    )

    if node_schema_id is not None:
        for edge_schema in fwd_skip_edge_schemas:
            if edge_schema.to_node_schema.id == node_schema_id:
                return (
                    edge_schema,
                    CT.graph.graph_schema.node_schema_id_to_node_schema[node_schema_id],
                )

        for edge_schema in bwd_skip_edge_schemas:
            if edge_schema.from_node_schema.id == node_schema_id:
                return (
                    edge_schema,
                    CT.graph.graph_schema.node_schema_id_to_node_schema[node_schema_id],
                )

    return None, None


class AgentExecutor:

    def __init__(
        self,
        model,
        elevenlabs_client,
        graph_schema,
        audio_output,
        remove_prev_tool_calls,
    ):
        self.model = model
        self.elevenlabs_client = elevenlabs_client
        self.graph_schema = graph_schema
        self.remove_prev_tool_calls = remove_prev_tool_calls
        self.audio_output = audio_output
        self.TC = TurnContainer()
        self.CT = ChatContext(
            graph=Graph(graph_schema=cashier_graph_schema),
            remove_prev_tool_calls=remove_prev_tool_calls,
        )
        self.CT.init_next_node(
            cashier_graph_schema.start_node_schema, None, self.TC, None
        )
        self.force_tool_choice = None

    def add_user_turn(self, msg):
        MessageDisplay.print_msg("user", msg)
        self.TC.add_user_turn(msg)
        skip_edge_schema, skip_node_schema = handle_skip(self.model, self.TC, self.CT)
        if skip_edge_schema is not None:
            self.CT.init_skip_node(
                skip_node_schema,
                skip_edge_schema,
                self.TC,
            )
            self.force_tool_choice = "get_state"
        self.CT.curr_node.update_first_user_message()

    def add_assistant_turn(self, model_completion):
        message = model_completion.get_or_stream_message()
        if message is not None:
            if self.audio_output:
                get_speech_from_text(message, self.elevenlabs_client)
                MessageDisplay.display_assistant_message(model_completion.msg_content)
            else:
                MessageDisplay.display_assistant_message(message)
            self.CT.need_user_input = True

        fn_id_to_output = {}
        new_edge_schema = None
        for function_call in model_completion.get_or_stream_fn_calls():
            function_args = function_call.function_args
            logger.debug(
                f"[FUNCTION_CALL] {Style.BRIGHT}name: {function_call.function_name}, id: {function_call.tool_call_id}{Style.NORMAL} with args:\n{json.dumps(function_args, indent=4)}"
            )
            with FunctionCallContext() as fn_call_context:
                if (
                    function_call.function_name
                    not in self.CT.curr_node.schema.tool_fn_names
                ):
                    raise InexistentFunctionError(function_call.function_name)

                if function_call.function_name.startswith("get_state"):
                    fn_output = getattr(self.CT.curr_node, function_call.function_name)(
                        **function_args
                    )
                elif function_call.function_name.startswith("update_state"):
                    fn_output = self.CT.curr_node.update_state(**function_args)
                else:
                    fn = ToolRegistry.GLOBAL_FN_NAME_TO_FN[function_call.function_name]
                    fn_output = fn(**function_args)

            if fn_call_context.has_exception():
                logger.debug(
                    f"[FUNCTION_EXCEPTION] {Style.BRIGHT}name: {function_call.function_name}, id: {function_call.tool_call_id}{Style.NORMAL} with exception:\n{str(fn_call_context.exception)}"
                )
                fn_id_to_output[function_call.tool_call_id] = fn_call_context.exception
            else:
                logger.debug(
                    f"[FUNCTION_RETURN] {Style.BRIGHT}name: {function_call.function_name}, id: {function_call.tool_call_id}{Style.NORMAL} with output:\n{json.dumps(fn_output, cls=CustomJSONEncoder, indent=4)}"
                )
                fn_id_to_output[function_call.tool_call_id] = fn_output

            self.CT.need_user_input = False

            if (
                not fn_call_context.has_exception()
                and function_call.function_name.startswith("update_state")
            ):
                for edge_schema in self.CT.next_edge_schemas:
                    if edge_schema.check_state_condition(self.CT.curr_node.state):
                        new_edge_schema = edge_schema
                        break

        self.TC.add_assistant_turn(
            model_completion.msg_content,
            model_completion.model_provider,
            model_completion.fn_calls,
            fn_id_to_output,
        )

        if new_edge_schema:
            new_node_schema = new_edge_schema.to_node_schema
            self.CT.init_next_node(
                new_node_schema,
                new_edge_schema,
                self.TC,
            )

    def get_model_completion_args(self):
        force_tool_choice = self.force_tool_choice
        self.force_tool_choice = None
        return {
            "turn_container": self.TC,
            "tool_names_or_tool_defs": self.CT.curr_node.schema.tool_fn_names,
            "extra_tool_registry": self.CT.curr_node.schema.tool_registry,
            "force_tool_choice": force_tool_choice,
        }
