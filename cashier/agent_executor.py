import copy
import json

from colorama import Style

from cashier.audio import get_speech_from_text
from cashier.function_call_context import FunctionCallContext, InexistentFunctionError
from cashier.graph import Direction, Graph
from cashier.gui import MessageDisplay
from cashier.logger import logger
from cashier.model import Model
from cashier.model_turn import TurnContainer
from cashier.model_util import CustomJSONEncoder, FunctionCall, ModelProvider
from cashier.prompts.node_schema_selection import NodeSchemaSelectionPrompt
from cashier.prompts.off_topic import OffTopicPrompt


def should_skip_node_schema(model, TM, current_node_schema, all_node_schemas):
    if len(all_node_schemas) == 1:
        return None

    model_name = "claude-3.5"
    model_provider = Model.get_model_provider(model_name)
    node_conv_msgs = copy.deepcopy(
        TM.model_provider_to_message_manager[model_provider].node_conversation_dicts
    )
    last_customer_msg = TM.get_user_message(content_only=True)

    prompt = OffTopicPrompt(
        background_prompt=current_node_schema.node_system_prompt.BACKGROUND_PROMPT(),
        node_prompt=current_node_schema.node_prompt,
        state_json_schema=current_node_schema.state_pydantic_model.model_json_schema(),
        tool_defs=json.dumps(
            current_node_schema.tool_registry.get_tool_defs(
                model_provider=model_provider
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

        self.curr_node = None
        self.need_user_input = True
        self.graph = Graph(graph_schema=graph_schema)
        self.next_edge_schemas = set()
        self.bwd_skip_edge_schemas = set()

        self.init_next_node(graph_schema.start_node_schema, None, None)
        self.force_tool_choice = None

    def init_node_core(
        self,
        node_schema,
        edge_schema,
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

        self.TC.add_node_turn(
            new_node,
            remove_prev_tool_calls=self.remove_prev_tool_calls,
            is_skip=is_skip,
        )
        MessageDisplay.print_msg("system", new_node.prompt)

        if node_schema.first_turn and prev_node is None:
            self.TC.add_assistant_direct_turn(node_schema.first_turn)
            MessageDisplay.print_msg("assistant", node_schema.first_turn.msg_content)

        if edge_schema:
            self.graph.add_edge(self.curr_node, new_node, edge_schema, direction)

        self.curr_node = new_node
        self.next_edge_schemas = set(
            self.graph.graph_schema.from_node_schema_id_to_edge_schema.get(
                new_node.schema.id, []
            )
        )
        self.bwd_skip_edge_schemas = self.graph.compute_bwd_skip_edge_schemas(
            self.curr_node, self.bwd_skip_edge_schemas
        )

    def init_next_node(self, node_schema, edge_schema, input=None):
        if self.curr_node:
            self.curr_node.mark_as_completed()

        if input is None and edge_schema:
            input = edge_schema.new_input_fn(self.curr_node.state, self.curr_node.input)

        if edge_schema:
            edge_schema, input = self.graph.compute_next_edge_schema(
                edge_schema, input, self.curr_node
            )
            node_schema = edge_schema.to_node_schema

        direction = Direction.FWD
        prev_node = self.graph.get_prev_node(edge_schema, direction)

        last_msg = self.TC.get_user_message(content_only=True)

        self.init_node_core(
            node_schema, edge_schema, input, last_msg, prev_node, direction, False
        )

    def init_skip_node(
        self,
        node_schema,
        edge_schema,
    ):
        direction = Direction.FWD
        if edge_schema and edge_schema.from_node_schema == node_schema:
            direction = Direction.BWD

        if direction == Direction.BWD:
            self.bwd_skip_edge_schemas.clear()

        prev_node = self.graph.get_prev_node(edge_schema, direction)
        input = prev_node.input

        last_msg = self.TC.get_asst_message(content_only=True)

        self.init_node_core(
            node_schema, edge_schema, input, last_msg, prev_node, direction, True
        )

    def handle_skip(self):
        fwd_skip_edge_schemas = self.graph.compute_fwd_skip_edge_schemas(
            self.curr_node, self.next_edge_schemas
        )
        bwd_skip_edge_schemas = self.bwd_skip_edge_schemas

        all_node_schemas = [self.curr_node.schema]
        all_node_schemas += [edge.to_node_schema for edge in fwd_skip_edge_schemas]
        all_node_schemas += [edge.from_node_schema for edge in bwd_skip_edge_schemas]

        node_schema_id = should_skip_node_schema(
            self.model, self.TC, self.curr_node.schema, all_node_schemas
        )

        if node_schema_id is not None:
            for edge_schema in fwd_skip_edge_schemas:
                if edge_schema.to_node_schema.id == node_schema_id:
                    return (
                        edge_schema,
                        self.graph.graph_schema.node_schema_id_to_node_schema[
                            node_schema_id
                        ],
                    )

            for edge_schema in bwd_skip_edge_schemas:
                if edge_schema.from_node_schema.id == node_schema_id:
                    return (
                        edge_schema,
                        self.graph.graph_schema.node_schema_id_to_node_schema[
                            node_schema_id
                        ],
                    )

        return None, None

    def add_user_turn(self, msg):
        MessageDisplay.print_msg("user", msg)
        self.TC.add_user_turn(msg)
        skip_edge_schema, skip_node_schema = self.handle_skip()
        if skip_edge_schema is not None:
            self.init_skip_node(
                skip_node_schema,
                skip_edge_schema,
            )

            fake_fn_call = FunctionCall.create_fake_fn_call("get_state", None, {})
            self.TC.add_assistant_turn(
                None,
                ModelProvider.ANTHROPIC,
                self.curr_node.schema.tool_registry,
                [fake_fn_call],
                {fake_fn_call.tool_call_id: self.curr_node.get_state()},
            )

        self.curr_node.update_first_user_message()

    def execute_function_call(self, fn_call, fn_callback=None):
        function_args = fn_call.function_args
        logger.debug(
            f"[FUNCTION_CALL] {Style.BRIGHT}name: {fn_call.function_name}, id: {fn_call.tool_call_id}{Style.NORMAL} with args:\n{json.dumps(function_args, indent=4)}"
        )
        with FunctionCallContext() as fn_call_context:
            if (
                fn_call.function_name
                not in self.curr_node.schema.tool_registry.tool_names
            ):
                raise InexistentFunctionError(fn_call.function_name)

            if fn_call.function_name.startswith("get_state"):
                fn_output = getattr(self.curr_node, fn_call.function_name)(
                    **function_args
                )
            elif fn_call.function_name.startswith("update_state"):
                fn_output = self.curr_node.update_state(**function_args)
            elif fn_callback is not None:
                # TODO: this exists for benchmarking. remove this once done
                fn_output = fn_callback(**function_args)
                if fn_output and (
                    type(fn_output) != str or not fn_output.strip().startswith("Error:")
                ):
                    fn_output = json.loads(fn_output)
            else:
                fn = self.curr_node.schema.tool_registry.fn_name_to_fn[
                    fn_call.function_name
                ]
                fn_output = fn(**function_args)

        if fn_call_context.has_exception():
            logger.debug(
                f"[FUNCTION_EXCEPTION] {Style.BRIGHT}name: {fn_call.function_name}, id: {fn_call.tool_call_id}{Style.NORMAL} with exception:\n{str(fn_call_context.exception)}"
            )
            return fn_call_context.exception, False
        else:
            logger.debug(
                f"[FUNCTION_RETURN] {Style.BRIGHT}name: {fn_call.function_name}, id: {fn_call.tool_call_id}{Style.NORMAL} with output:\n{json.dumps(fn_output, cls=CustomJSONEncoder, indent=4)}"
            )
            return fn_output, True

    def add_assistant_turn(self, model_completion, fn_callback=None):
        message = model_completion.get_or_stream_message()
        if message is not None:
            if self.audio_output:
                get_speech_from_text(message, self.elevenlabs_client)
                MessageDisplay.display_assistant_message(model_completion.msg_content)
            else:
                MessageDisplay.display_assistant_message(message)
            self.need_user_input = True

        fn_id_to_output = {}
        new_edge_schema = None
        for function_call in model_completion.get_or_stream_fn_calls():
            fn_id_to_output[function_call.tool_call_id], is_success = (
                self.execute_function_call(function_call, fn_callback)
            )

            self.need_user_input = False

            if is_success and function_call.function_name.startswith("update_state"):
                for edge_schema in self.next_edge_schemas:
                    if edge_schema.check_state_condition(self.curr_node.state):
                        new_edge_schema = edge_schema
                        break

        self.TC.add_assistant_turn(
            model_completion.msg_content,
            model_completion.model_provider,
            self.curr_node.schema.tool_registry,
            model_completion.fn_calls,
            fn_id_to_output,
        )

        if new_edge_schema:
            new_node_schema = new_edge_schema.to_node_schema
            self.init_next_node(
                new_node_schema,
                new_edge_schema,
            )

    def get_model_completion_kwargs(self):
        force_tool_choice = self.force_tool_choice
        self.force_tool_choice = None
        return {
            "turn_container": self.TC,
            "tool_registry": self.curr_node.schema.tool_registry,
            "force_tool_choice": force_tool_choice,
        }
