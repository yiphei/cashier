import json
from typing import Any, Callable, Dict, Optional, Tuple, cast

from colorama import Style

from cashier.audio import get_speech_from_text
from cashier.graph.graph_schema import Graph, GraphSchema
from cashier.graph.node_schema import Direction
from cashier.graph.request_graph import GraphEdgeSchema, RequestGraph
from cashier.gui import MessageDisplay
from cashier.logger import logger
from cashier.model.model_completion import ModelOutput
from cashier.model.model_util import CustomJSONEncoder, FunctionCall, ModelProvider
from cashier.tool.function_call_context import (
    FunctionCallContext,
    InexistentFunctionError,
)
from cashier.turn_container import TurnContainer


class AgentExecutor:

    def __init__(
        self,
        audio_output: bool,
        remove_prev_tool_calls: bool,
        request_graph_schema=None,
    ):
        self.request_graph_schema = request_graph_schema
        self.request_graph = RequestGraph(None, self.request_graph_schema)
        self.curr_graph_schema = None

        self.remove_prev_tool_calls = remove_prev_tool_calls
        self.audio_output = audio_output
        self.last_model_provider = None
        self.TC = TurnContainer()

        self.need_user_input = True
        # self.next_edge_schemas: Set[EdgeSchema] = set()
        # self.bwd_skip_edge_schemas: Set[EdgeSchema] = set()

        self.graph = None
        self.request_graph.init_node_core(
            self.request_graph_schema.start_node_schema,
            None,
            None,
            None,
            None,
            Direction.FWD,
            self.TC,
            self.remove_prev_tool_calls,
        )
        self.force_tool_choice = None
        self.new_edge_schema = None
        self.new_node_schema = None

    def add_user_turn(
        self, msg: str, model_provider: Optional[ModelProvider] = None
    ) -> None:
        MessageDisplay.print_msg("user", msg)
        self.TC.add_user_turn(msg)
        model_provider = (
            model_provider or self.last_model_provider or ModelProvider.OPENAI
        )
        self.request_graph.handle_user_turn(
            msg, self.TC, model_provider, self.remove_prev_tool_calls
        )
        if isinstance(self.request_graph.curr_node, Graph):
            self.graph = self.request_graph.curr_node

    def execute_function_call(
        self, fn_call: FunctionCall, fn_callback: Optional[Callable] = None
    ) -> Tuple[Any, bool]:
        function_args = fn_call.args
        logger.debug(
            f"[FUNCTION_CALL] {Style.BRIGHT}name: {fn_call.name}, id: {fn_call.id}{Style.NORMAL} with args:\n{json.dumps(function_args, indent=4)}"
        )
        with FunctionCallContext() as fn_call_context:
            if fn_call.name not in self.graph.curr_node.schema.tool_registry.tool_names:
                raise InexistentFunctionError(fn_call.name)

            if fn_call.name.startswith("get_state"):
                fn_output = getattr(self.graph.curr_node, fn_call.name)(**function_args)
            elif fn_call.name.startswith("update_state"):
                fn_output = self.graph.curr_node.update_state(**function_args)  # type: ignore
            elif fn_callback is not None:
                # TODO: this exists for benchmarking. remove this once done
                fn_output = fn_callback(**function_args)
                if fn_output and (
                    type(fn_output) is not str
                    or not fn_output.strip().startswith("Error:")
                ):
                    fn_output = json.loads(fn_output)
            else:
                fn = self.graph.curr_node.schema.tool_registry.fn_name_to_fn[
                    fn_call.name
                ]
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
            return fn_output, (
                type(fn_output) is not str or not fn_output.strip().startswith("Error:")
            )

    def add_assistant_turn(
        self, model_completion: ModelOutput, fn_callback: Optional[Callable] = None
    ) -> None:
        if (
            self.new_edge_schema is not None
            and self.graph.curr_node.schema.run_assistant_turn_before_transition
        ):
            self.graph.curr_node.has_run_assistant_turn_before_transition = True

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

        if self.graph is not None:
            fn_id_to_output = {}
            fn_calls = []
            if self.new_edge_schema is None:
                for function_call in model_completion.get_or_stream_fn_calls():
                    fn_id_to_output[function_call.id], is_success = (
                        self.execute_function_call(function_call, fn_callback)
                    )
                    fn_calls.append(function_call)
                    self.need_user_input = False

                    edge_schemas = (
                        self.request_graph_schema.from_node_schema_id_to_edge_schema[
                            self.curr_graph_schema.id
                        ]
                        if self.graph.curr_node.schema
                        == self.graph.graph_schema.last_node_schema
                        else self.graph.next_edge_schemas
                    )
                    for edge_schema in edge_schemas:
                        if edge_schema.check_transition_config(
                            self.graph.curr_node.state, function_call, is_success
                        ):
                            if isinstance(edge_schema, GraphEdgeSchema):
                                fake_fn_call = FunctionCall.create(
                                    api_id_model_provider=None,
                                    api_id=None,
                                    name="think",
                                    args={
                                        "thought": f"I just completed the current request. The next request to be addressed is: {self.request_graph.tasks[self.request_graph.current_graph_schema_idx + 1]}. I must explicitly inform the customer that the current request is completed and that I will address the next request right away. Only after I informed the customer do I receive the tools to address the next request."
                                    },
                                )
                                fn_id_to_output[fake_fn_call.id] = None
                                fn_calls.append(fake_fn_call)
                            self.new_edge_schema = edge_schema
                            if isinstance(edge_schema, GraphEdgeSchema):
                                self.new_node_schema = edge_schema.to_node_schema
                            else:
                                self.new_node_schema = edge_schema.to_node_schema
                            break

            self.TC.add_assistant_turn(
                model_completion.msg_content,
                model_completion.model_provider,
                self.graph.curr_node.schema.tool_registry,
                fn_calls,
                fn_id_to_output,
            )
            if self.new_edge_schema and (
                not self.graph.curr_node.schema.run_assistant_turn_before_transition
                or self.graph.curr_node.has_run_assistant_turn_before_transition
            ):
                input = None
                if isinstance(self.new_node_schema, GraphSchema):
                    new_input = self.new_edge_schema.new_input_fn(self.graph.state)
                    self.request_graph.current_graph_schema_idx += 1
                    self.curr_graph_schema = self.request_graph.graph_schema_sequence[
                        self.request_graph.current_graph_schema_idx
                    ]
                    self.graph = Graph(
                        input=new_input, graph_schema=self.curr_graph_schema
                    )
                    self.new_node_schema, temp_edge_schema = (
                        self.graph.compute_init_node_edge_schema()
                    )
                    self.new_edge_schema = None
                    input = temp_edge_schema.new_input_fn(self.graph.state)

                self.request_graph.curr_node.init_next_node(
                    self.new_node_schema,
                    self.new_edge_schema,
                    self.TC,
                    self.remove_prev_tool_calls,
                    input,
                )
                self.new_edge_schema = None
                self.new_node_schema = None

        else:
            self.TC.add_assistant_turn(
                model_completion.msg_content, model_completion.model_provider
            )
            self.new_edge_schema = None

    def get_model_completion_kwargs(self) -> Dict[str, Any]:
        force_tool_choice = self.force_tool_choice
        self.force_tool_choice = None
        target_graph = self.graph or self.request_graph or None
        return {
            "turn_container": self.TC,
            "tool_registry": (
                target_graph.curr_node.schema.tool_registry
                if target_graph is not None
                else None
            ),
            "force_tool_choice": force_tool_choice,
            "exclude_update_state_fns": (
                not target_graph.curr_node.first_user_message
                if target_graph.curr_node is not None
                else False
            ),
        }
