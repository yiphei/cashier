import json
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union, cast

from colorama import Style

from cashier.audio import get_speech_from_text
from cashier.graph.edge_schema import EdgeSchema
from cashier.graph.graph_schema import Graph, GraphSchema
from cashier.graph.node_schema import Direction, Node, NodeSchema
from cashier.gui import MessageDisplay
from cashier.logger import logger
from cashier.model.model_completion import ModelOutput
from cashier.model.model_turn import AssistantTurn
from cashier.model.model_util import CustomJSONEncoder, FunctionCall, ModelProvider
from cashier.prompts.node_schema_selection import NodeSchemaSelectionPrompt
from cashier.prompts.off_topic import OffTopicPrompt
from cashier.tool.function_call_context import (
    FunctionCallContext,
    InexistentFunctionError,
)
from cashier.turn_container import TurnContainer


def should_change_node_schema(
    TM: TurnContainer,
    current_node_schema: NodeSchema,
    all_node_schemas: Set[NodeSchema],
    is_wait: bool,
) -> Optional[int]:
    if len(all_node_schemas) == 1:
        return None
    return NodeSchemaSelectionPrompt.run(
        "claude-3.5",
        current_node_schema=current_node_schema,
        tc=TM,
        all_node_schemas=all_node_schemas,
        is_wait=is_wait,
    )


class AgentExecutor:

    def __init__(
        self,
        graph_schema: GraphSchema,
        audio_output: bool,
        remove_prev_tool_calls: bool,
    ):
        self.graph_schema = graph_schema
        self.remove_prev_tool_calls = remove_prev_tool_calls
        self.audio_output = audio_output
        self.last_model_provider = None
        self.TC = TurnContainer()

        self.curr_node = None  # type: Node # type: ignore
        self.need_user_input = True
        self.graph = Graph(graph_schema=graph_schema)
        self.next_edge_schemas: Set[EdgeSchema] = set()
        self.bwd_skip_edge_schemas: Set[EdgeSchema] = set()

        self.init_next_node(graph_schema.start_node_schema, None, None)
        self.force_tool_choice = None
        self.new_edge_schema = None

    def init_node_core(
        self,
        node_schema: NodeSchema,
        edge_schema: Optional[EdgeSchema],
        input: Any,
        last_msg: Optional[str],
        prev_node: Optional[Node],
        direction: Direction,
        is_skip: bool = False,
    ) -> None:
        logger.debug(
            f"[NODE_SCHEMA] Initializing node with {Style.BRIGHT}node_schema_id: {node_schema.id}{Style.NORMAL}"
        )
        new_node = node_schema.create_node(
            input, last_msg, edge_schema, prev_node, direction  # type: ignore
        )

        self.TC.add_node_turn(
            new_node,
            remove_prev_tool_calls=self.remove_prev_tool_calls,
            is_skip=is_skip,
        )
        MessageDisplay.print_msg("system", new_node.prompt)

        if node_schema.first_turn and prev_node is None:
            assert isinstance(node_schema.first_turn, AssistantTurn)
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

    def init_next_node(
        self,
        node_schema: NodeSchema,
        edge_schema: Optional[EdgeSchema],
        input: Any = None,
    ) -> None:
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
        node_schema: NodeSchema,
        edge_schema: EdgeSchema,
    ) -> None:
        direction = Direction.FWD
        if edge_schema and edge_schema.from_node_schema == node_schema:
            direction = Direction.BWD

        if direction == Direction.BWD:
            self.bwd_skip_edge_schemas.clear()

        prev_node = self.graph.get_prev_node(edge_schema, direction)
        assert prev_node is not None
        input = prev_node.input

        last_msg = self.TC.get_asst_message(content_only=True)

        self.init_node_core(
            node_schema, edge_schema, input, last_msg, prev_node, direction, True
        )

    def handle_skip(
        self,
        fwd_skip_edge_schemas: Set[EdgeSchema],
        bwd_skip_edge_schemas: Set[EdgeSchema],
    ) -> Union[Tuple[EdgeSchema, NodeSchema], Tuple[None, None]]:
        all_node_schemas = {self.curr_node.schema}
        all_node_schemas.update(edge.to_node_schema for edge in fwd_skip_edge_schemas)
        all_node_schemas.update(edge.from_node_schema for edge in bwd_skip_edge_schemas)

        node_schema_id = should_change_node_schema(
            self.TC, self.curr_node.schema, all_node_schemas, False
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

    def handle_wait(
        self,
        fwd_skip_edge_schemas: Set[EdgeSchema],
        bwd_skip_edge_schemas: Set[EdgeSchema],
    ) -> Union[Tuple[EdgeSchema, NodeSchema], Tuple[None, None]]:
        remaining_edge_schemas = (
            set(self.graph_schema.edge_schemas)
            - fwd_skip_edge_schemas
            - bwd_skip_edge_schemas
        )

        all_node_schemas = {self.curr_node.schema}
        all_node_schemas.update(edge.to_node_schema for edge in remaining_edge_schemas)

        node_schema_id = should_change_node_schema(
            self.TC, self.curr_node.schema, all_node_schemas, True
        )

        if node_schema_id is not None:
            for edge_schema in remaining_edge_schemas:
                if edge_schema.to_node_schema.id == node_schema_id:
                    return (
                        edge_schema,
                        self.graph.graph_schema.node_schema_id_to_node_schema[
                            node_schema_id
                        ],
                    )

        return None, None

    def handle_is_off_topic(
        self,
    ) -> Union[Tuple[EdgeSchema, NodeSchema, bool], Tuple[None, None, bool]]:
        fwd_skip_edge_schemas = self.graph.compute_fwd_skip_edge_schemas(
            self.curr_node, self.next_edge_schemas
        )
        bwd_skip_edge_schemas = self.bwd_skip_edge_schemas

        edge_schema, node_schema = self.handle_wait(
            fwd_skip_edge_schemas, bwd_skip_edge_schemas
        )
        if edge_schema:
            return edge_schema, node_schema, True  # type: ignore

        edge_schema, node_schema = self.handle_skip(
            fwd_skip_edge_schemas, bwd_skip_edge_schemas
        )
        return edge_schema, node_schema, False  # type: ignore

    def add_user_turn(
        self, msg: str, model_provider: Optional[ModelProvider] = None
    ) -> None:
        MessageDisplay.print_msg("user", msg)
        self.TC.add_user_turn(msg)
        if not OffTopicPrompt.run(
            "claude-3.5", current_node_schema=self.curr_node.schema, tc=self.TC
        ):
            edge_schema, node_schema, is_wait = self.handle_is_off_topic()
            if edge_schema and node_schema:
                if is_wait:
                    fake_fn_call = FunctionCall.create(
                        api_id_model_provider=None,
                        api_id=None,
                        name="think",
                        args={
                            "thought": "At least part of the customer request/question is off-topic for the current conversation and will actually be addressed later. According to the policies, I must tell the customer that 1) their off-topic request/question will be addressed later and 2) we must finish the current business before we can get to it. I must refuse to engage with the off-topic request/question in any way."
                        },
                    )
                    self.TC.add_assistant_turn(
                        None,
                        model_provider
                        or self.last_model_provider
                        or ModelProvider.OPENAI,
                        self.curr_node.schema.tool_registry,
                        [fake_fn_call],
                        {fake_fn_call.id: None},
                    )
                else:
                    self.init_skip_node(
                        node_schema,
                        edge_schema,
                    )

                    fake_fn_call = FunctionCall.create(
                        api_id=None,
                        api_id_model_provider=None,
                        name="get_state",
                        args={},
                    )
                    self.TC.add_assistant_turn(
                        None,
                        model_provider
                        or self.last_model_provider
                        or ModelProvider.OPENAI,
                        self.curr_node.schema.tool_registry,
                        [fake_fn_call],
                        {fake_fn_call.id: self.curr_node.get_state()},
                    )
        self.curr_node.update_first_user_message()

    def execute_function_call(
        self, fn_call: FunctionCall, fn_callback: Optional[Callable] = None
    ) -> Tuple[Any, bool]:
        function_args = fn_call.args
        logger.debug(
            f"[FUNCTION_CALL] {Style.BRIGHT}name: {fn_call.name}, id: {fn_call.id}{Style.NORMAL} with args:\n{json.dumps(function_args, indent=4)}"
        )
        with FunctionCallContext() as fn_call_context:
            if fn_call.name not in self.curr_node.schema.tool_registry.tool_names:
                raise InexistentFunctionError(fn_call.name)

            if fn_call.name.startswith("get_state"):
                fn_output = getattr(self.curr_node, fn_call.name)(**function_args)
            elif fn_call.name.startswith("update_state"):
                fn_output = self.curr_node.update_state(**function_args)  # type: ignore
            elif fn_callback is not None:
                # TODO: this exists for benchmarking. remove this once done
                fn_output = fn_callback(**function_args)
                if fn_output and (
                    type(fn_output) != str or not fn_output.strip().startswith("Error:")
                ):
                    fn_output = json.loads(fn_output)
            else:
                fn = self.curr_node.schema.tool_registry.fn_name_to_fn[fn_call.name]
                fn_output = fn(**function_args)

        if fn_call_context.has_exception():
            logger.debug(
                f"[FUNCTION_EXCEPTION] {Style.BRIGHT}name: {fn_call.name}, id: {fn_call.id}{Style.NORMAL} with exception:\n{str(fn_call_context.exception)}"
            )
            return fn_call_context.exception, False
        else:
            logger.debug(
                f"[FUNCTION_RETURN] {Style.BRIGHT}name: {fn_call.name}, id: {fn_call.id}{Style.NORMAL} with output:\n{json.dumps(fn_output, cls=CustomJSONEncoder, indent=4)}"
            )
            return fn_output, True

    def add_assistant_turn(
        self, model_completion: ModelOutput, fn_callback: Optional[Callable] = None
    ) -> None:
        if (
            self.new_edge_schema is not None
            and self.curr_node.schema.run_assistant_turn_before_transition
        ):
            self.curr_node.has_run_assistant_turn_before_transition = True

        self.last_model_provider = model_completion.model_provider
        message = model_completion.get_or_stream_message()
        if message is not None:
            if self.audio_output:
                get_speech_from_text(message)
                MessageDisplay.display_assistant_message(
                    cast(str, model_completion.msg_content)
                )
            else:
                MessageDisplay.display_assistant_message(message)
            self.need_user_input = True

        fn_id_to_output = {}
        if self.new_edge_schema is None:
            for function_call in model_completion.get_or_stream_fn_calls():
                fn_id_to_output[function_call.id], is_success = (
                    self.execute_function_call(function_call, fn_callback)
                )

                self.need_user_input = False

                for edge_schema in self.next_edge_schemas:
                    if edge_schema.check_transition_config(
                        self.curr_node.state, function_call, is_success
                    ):
                        self.new_edge_schema = edge_schema
                        break

        self.TC.add_assistant_turn(
            model_completion.msg_content,
            model_completion.model_provider,
            self.curr_node.schema.tool_registry,
            model_completion.fn_calls,
            fn_id_to_output,
        )

        if self.new_edge_schema and (
            not self.curr_node.schema.run_assistant_turn_before_transition
            or self.curr_node.has_run_assistant_turn_before_transition
        ):
            new_node_schema = self.new_edge_schema.to_node_schema
            self.init_next_node(
                new_node_schema,
                self.new_edge_schema,
            )
            self.new_edge_schema = None

    def get_model_completion_kwargs(self) -> Dict[str, Any]:
        force_tool_choice = self.force_tool_choice
        self.force_tool_choice = None
        return {
            "turn_container": self.TC,
            "tool_registry": self.curr_node.schema.tool_registry,
            "force_tool_choice": force_tool_choice,
        }
