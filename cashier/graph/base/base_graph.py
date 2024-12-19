import json
from collections import defaultdict, deque
from typing import Any, Callable, List, Literal, Optional, Set, Tuple, overload

from colorama import Style

from cashier.graph.base.base_executable import BaseGraphExecutable
from cashier.graph.conversation_node import (
    ConversationNode,
    ConversationNodeSchema,
    Direction,
)
from cashier.graph.edge_schema import Edge, EdgeSchema, FwdSkipType
from cashier.graph.mixin.has_id_mixin import HasIdMixin
from cashier.graph.mixin.has_status_mixin import Status
from cashier.gui import MessageDisplay
from cashier.logger import logger
from cashier.model.model_completion import ModelOutput
from cashier.model.model_turn import AssistantTurn
from cashier.model.model_util import (
    CustomJSONEncoder,
    FunctionCall,
    create_think_fn_call,
)
from cashier.tool.function_call_context import (
    FunctionCallContext,
    InexistentFunctionError,
)


class BaseGraphSchema:
    def __init__(
        self,
        description: str,
        node_schemas: List[ConversationNode],
    ):
        self.description = description
        self.node_schemas = node_schemas

        self.node_schema_id_to_node_schema = {
            node_schema.id: node_schema for node_schema in self.node_schemas
        }

    def get_leaf_conv_node_schemas(self):
        all_node_schemas = []
        for node_schema in self.get_node_schemas():
            if isinstance(node_schema, BaseGraphSchema):
                all_node_schemas.extend(node_schema.get_leaf_conv_node_schemas())
            else:
                all_node_schemas.append(node_schema)
        return all_node_schemas


class BaseGraph(BaseGraphExecutable, HasIdMixin):
    def __init__(
        self,
        input: Any,
        schema: BaseGraphSchema,
        edge_schemas=None,
        request=None,
        state=None,
    ):
        HasIdMixin.__init__(self)
        BaseGraphExecutable.__init__(self, state)
        self.input = input
        self.schema = schema
        self.request = request
        self.parent = None

        # graph schema
        self.edge_schemas = edge_schemas or []
        self.edge_schema_id_to_edge_schema = {
            edge_schema.id: edge_schema for edge_schema in self.edge_schemas
        }
        self.from_node_schema_id_to_edge_schema = {
            edge_schema.from_node_schema.id: edge_schema
            for edge_schema in self.edge_schemas
        }
        self.to_node_schema_id_to_edge_schema = {
            edge_schema.to_node_schema.id: edge_schema
            for edge_schema in self.edge_schemas
        }

        # graph instance
        self.node_schema_id_to_nodes = defaultdict(list)
        self.edge_schema_id_to_edges = defaultdict(list)
        self.to_node_id_to_edge = defaultdict(lambda: None)

        # transition
        self.next_edge_schema: Optional[EdgeSchema] = None
        self.bwd_skip_edge_schemas: Set[EdgeSchema] = set()
        self.new_edge_schema = None
        self.new_node_schema = None

    def add_edge_schema(self, edge_schema):
        self.edge_schemas.append(edge_schema)
        self.edge_schema_id_to_edge_schema[edge_schema.id] = edge_schema
        self.from_node_schema_id_to_edge_schema[edge_schema.from_node_schema.id] = (
            edge_schema
        )
        self.to_node_schema_id_to_edge_schema[edge_schema.to_node_schema.id] = (
            edge_schema
        )

    @property
    def transition_queue(self):
        sub_queue = (
            self.curr_node.transition_queue
            if isinstance(self.curr_node, BaseGraph)
            else deque()
        )
        return sub_queue + self.local_transition_queue

    @property
    def curr_conversation_node(self):
        return (
            self.curr_node.curr_conversation_node
            if isinstance(self.curr_node, BaseGraph)
            else self.curr_node
        )

    @property
    def curr_graph(self):
        return self.curr_conversation_node.parent

    def add_fwd_edge(
        self,
        from_node: ConversationNode,
        to_node: ConversationNode,
        edge_schema: EdgeSchema,
    ) -> None:
        edge = Edge(from_node, to_node, edge_schema)
        self.edge_schema_id_to_edges[edge_schema.id].append(edge)
        self.to_node_id_to_edge[to_node.id] = edge

    @overload
    def get_edge_by_edge_schema_id(  # noqa: E704
        self, edge_schema_id: int, idx: int = -1, raise_if_none: Literal[True] = True
    ) -> Edge: ...

    @overload
    def get_edge_by_edge_schema_id(  # noqa: E704
        self, edge_schema_id: int, idx: int = -1, raise_if_none: Literal[False] = False
    ) -> Optional[Edge]: ...

    def get_edge_by_edge_schema_id(
        self, edge_schema_id: int, idx: int = -1, raise_if_none: bool = True
    ) -> Optional[Edge]:
        edge = (
            self.edge_schema_id_to_edges[edge_schema_id][idx]
            if len(self.edge_schema_id_to_edges[edge_schema_id]) >= abs(idx)
            else None
        )
        if edge is None and raise_if_none:
            raise ValueError()
        return edge

    def get_edge_schema_by_from_node_schema_id(
        self, node_schema_id: int
    ) -> Optional[EdgeSchema]:
        return self.from_node_schema_id_to_edge_schema.get(node_schema_id, None)

    def get_prev_node(
        self,
        edge_schema: Optional[EdgeSchema],
        node_schema,
        direction: Optional[Direction] = None,
    ) -> Optional[ConversationNode]:
        if (
            edge_schema
            and self.get_edge_by_edge_schema_id(edge_schema.id, raise_if_none=False)
            is not None
        ):
            edge = self.get_edge_by_edge_schema_id(edge_schema.id)
            return edge.to_node if direction == Direction.FWD else edge.from_node
        elif (
            edge_schema is None
            and node_schema
            and self.node_schema_id_to_nodes[node_schema.id]
        ):
            return self.node_schema_id_to_nodes[node_schema.id][-1]
        else:
            return None

    def is_prev_from_node_completed(
        self, edge_schema: EdgeSchema, is_start_node: bool
    ) -> bool:
        idx = -1 if is_start_node else -2
        edge = self.get_edge_by_edge_schema_id(edge_schema.id, idx, raise_if_none=False)
        return edge.from_node.status == Status.COMPLETED if edge else False

    def compute_next_edge_schema(
        self,
        start_edge_schema: EdgeSchema,
        start_input: Any,
    ) -> Tuple[EdgeSchema, Any]:
        next_edge_schema = start_edge_schema
        edge_schema = start_edge_schema
        input = start_input
        while (
            self.get_edge_by_edge_schema_id(next_edge_schema.id, raise_if_none=False)
            is not None
        ):
            edge = self.get_edge_by_edge_schema_id(next_edge_schema.id)
            from_node = edge.from_node
            to_node = edge.to_node
            if from_node.schema == self.curr_node.schema:
                from_node = self.curr_node

            can_skip, skip_type = next_edge_schema.can_skip(
                self.state,  # TODO: this class does not explicitly have a state
                from_node,
                to_node,
                self.is_prev_from_node_completed(
                    next_edge_schema, from_node == self.curr_node
                ),
            )

            if can_skip:
                edge_schema = next_edge_schema

                next_next_edge_schema = self.get_edge_schema_by_from_node_schema_id(
                    to_node.schema.id
                )

                if next_next_edge_schema:
                    next_edge_schema = next_next_edge_schema
                else:
                    input = to_node.input
                    break
            elif skip_type == FwdSkipType.SKIP_IF_INPUT_UNCHANGED:
                if from_node.status != Status.COMPLETED:
                    input = from_node.input
                else:
                    edge_schema = next_edge_schema
                    if from_node != self.curr_node:
                        input = edge_schema.to_node_schema.get_input(
                            from_node.state, edge_schema
                        )
                break
            else:
                if from_node != self.curr_node:
                    input = from_node.input
                break

        return edge_schema, input

    def add_edge(
        self,
        curr_node: ConversationNode,
        new_node: ConversationNode,
        edge_schema: EdgeSchema,
        direction: Direction = Direction.FWD,
    ) -> None:
        if direction == Direction.FWD:
            immediate_from_node = curr_node
            if edge_schema.from_node_schema != curr_node.schema:
                from_node = self.edge_schema_id_to_edges[edge_schema.id][-1].from_node
                immediate_from_node = from_node
                while from_node.schema != curr_node.schema:
                    edge = self.to_node_id_to_edge[from_node.id]
                    prev_edge_schema = edge.schema
                    from_node = edge.from_node
                    to_node = edge.to_node

                self.add_fwd_edge(curr_node, to_node, prev_edge_schema)  # type: ignore

            self.add_fwd_edge(immediate_from_node, new_node, edge_schema)
        elif direction == Direction.BWD:
            if self.to_node_id_to_edge[new_node.id]:
                edge = self.to_node_id_to_edge[new_node.id]
                self.add_fwd_edge(
                    edge.from_node,
                    new_node,
                    self.to_node_id_to_edge[new_node.id].schema,
                )

    def post_node_init(
        self,
        edge_schema: Optional[EdgeSchema],
        prev_node: Optional[ConversationNode],
        TC,
        is_skip: bool = False,
    ) -> None:
        node_schema = self.curr_node.schema
        if not isinstance(node_schema, BaseGraphSchema):
            TC.add_node_turn(
                self.curr_node,
                is_skip=is_skip,
            )
            MessageDisplay.print_msg("system", self.curr_node.prompt)

            if node_schema.first_turn and prev_node is None:
                assert isinstance(node_schema.first_turn, AssistantTurn)
                TC.add_assistant_direct_turn(node_schema.first_turn)
                MessageDisplay.print_msg(
                    "assistant", node_schema.first_turn.msg_content
                )

    def get_request_for_init_graph_core(self, increase_counter=True):
        return self.request

    def init_node_core(
        self,
        node_schema: ConversationNodeSchema,
        edge_schema: Optional[EdgeSchema],
        input: Any,
        last_msg: Optional[str],
        prev_node: Optional[ConversationNode],
        direction: Direction,
        request,
    ):
        logger.debug(
            f"[NODE_SCHEMA] Initializing node with {Style.BRIGHT}node_schema_id: {node_schema.id}{Style.NORMAL}"
        )
        new_node = node_schema.create_node(
            input, last_msg, edge_schema, prev_node, direction, request  # type: ignore
        )
        self.node_schema_id_to_nodes[node_schema.id].append(new_node)
        new_node.parent = self

        if edge_schema and self.curr_node is not None:
            self.add_edge(self.curr_node, new_node, edge_schema, direction)

        return new_node

    def init_node(
        self,
        node_schema: ConversationNodeSchema,
        edge_schema: Optional[EdgeSchema],
        input: Any,
        last_msg: Optional[str],
        prev_node: Optional[ConversationNode],
        direction: Direction,
        TC,
    ) -> None:

        if isinstance(node_schema, BaseGraphSchema):
            request = self.get_request_for_init_graph_core(True)
        else:
            request = self.request

        new_node = self.init_node_core(
            node_schema, edge_schema, input, last_msg, prev_node, direction, request
        )
        self.curr_node = new_node

        self.post_node_init(
            edge_schema,
            prev_node,
            TC,
            False,
        )

        if isinstance(node_schema, BaseGraphSchema):
            node_schema, edge_schema = new_node.compute_init_node_edge_schema()
            self.curr_node.init_next_node(node_schema, edge_schema, TC, None)

    def _init_next_node(
        self,
        node_schema,
        edge_schema,
        TC,
        direction,
        last_msg,
        input,
    ) -> None:
        if input is None and edge_schema:
            # TODO: this is bad. refactor this
            if hasattr(self, "state"):
                input = node_schema.get_input(self.state, edge_schema)
            else:
                input = node_schema.get_input(self.curr_node.state, edge_schema)

        if edge_schema:
            edge_schema, input = self.compute_next_edge_schema(edge_schema, input)
            node_schema = edge_schema.to_node_schema

        prev_node = self.get_prev_node(edge_schema, node_schema, direction)

        self.init_node(
            node_schema,
            edge_schema,
            input,
            last_msg,
            prev_node,
            direction,
            TC,
        )

    def init_next_node(
        self,
        node_schema: ConversationNodeSchema,
        edge_schema: Optional[EdgeSchema],
        TC,
        input: Any = None,
    ) -> None:
        curr_node = self.curr_node
        parent_node = self
        transition_queue = self.transition_queue
        while transition_queue:
            curr_node = transition_queue.popleft()
            parent_node = curr_node.parent
            assert curr_node.status == Status.TRANSITIONING
            curr_node.mark_as_completed()
            parent_node.local_transition_queue.clear()

        direction = Direction.FWD
        last_msg = TC.get_user_message(content_only=True)

        parent_node._init_next_node(
            node_schema,
            edge_schema,
            TC,
            direction,
            last_msg,
            input,
        )

    def execute_function_call(
        self, fn_call: FunctionCall, fn_callback: Optional[Callable] = None
    ) -> Tuple[Any, bool]:
        function_args = fn_call.args
        logger.debug(
            f"[FUNCTION_CALL] {Style.BRIGHT}name: {fn_call.name}, id: {fn_call.id}{Style.NORMAL} with args:\n{json.dumps(function_args, indent=4)}"
        )
        with FunctionCallContext() as fn_call_context:
            if (
                fn_call.name
                not in self.curr_conversation_node.schema.tool_registry.tool_names
            ):
                raise InexistentFunctionError(fn_call.name)

            if fn_call.name.startswith("get_state"):
                fn_output = getattr(self.curr_conversation_node, fn_call.name)(
                    **function_args
                )
            elif fn_call.name.startswith("update_state"):
                fn_output = self.curr_conversation_node.update_state(**function_args)  # type: ignore
            elif fn_callback is not None:
                # TODO: this exists for benchmarking. remove this once done
                fn_output = fn_callback(**function_args)
                if fn_output and (
                    type(fn_output) is not str
                    or not fn_output.strip().startswith("Error:")
                ):
                    fn_output = json.loads(fn_output)
            else:
                fn = self.curr_conversation_node.schema.tool_registry.fn_name_to_fn[
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

    def handle_assistant_turn(
        self, model_completion: ModelOutput, TC, fn_callback: Optional[Callable] = None
    ) -> None:
        from cashier.graph.request_graph import RequestGraph

        if (
            self.transition_queue
            and self.transition_queue[-1].schema.run_assistant_turn_before_transition
        ):
            self.transition_queue[-1].has_run_assistant_turn_before_transition = True

        need_user_input = True
        fn_id_to_output = {}
        fn_calls = []
        if self.new_edge_schema is None:
            for function_call in model_completion.get_or_stream_fn_calls():
                fn_id_to_output[function_call.id], is_success = (
                    self.execute_function_call(function_call, fn_callback)
                )
                fn_calls.append(function_call)
                need_user_input = False

                (
                    new_edge_schema,
                    new_node_schema,
                ) = self.check_transition(function_call, is_success)
                if new_node_schema is not None:
                    self.new_edge_schema = new_edge_schema
                    self.new_node_schema = new_node_schema
                    if isinstance(self, RequestGraph):
                        if (
                            self.curr_node is not None
                            and not isinstance(self.curr_node, ConversationNode)
                            and self.curr_node.status == Status.TRANSITIONING
                            and self.current_graph_schema_idx < len(self.requests) - 1
                        ):
                            fake_fn_call = create_think_fn_call(
                                f"I just completed the current request. The next request to be addressed is: {self.requests[self.current_graph_schema_idx + 1]}. I must explicitly inform the customer that the current request is completed and that I will address the next request right away. Only after I informed the customer do I receive the tools to address the next request."
                            )
                            fn_id_to_output[fake_fn_call.id] = None
                            fn_calls.append(fake_fn_call)
                    break

        TC.add_assistant_turn(
            model_completion.msg_content,
            model_completion.model_provider,
            self.curr_conversation_node.schema.tool_registry,
            fn_calls,
            fn_id_to_output,
        )

        if self.transition_queue and (
            not self.transition_queue[-1].schema.run_assistant_turn_before_transition
            or self.transition_queue[-1].has_run_assistant_turn_before_transition
        ):
            self.init_next_node(
                self.new_node_schema,
                self.new_edge_schema,
                TC,
                None,
            )
            self.new_edge_schema = None
            self.new_node_schema = None

        return need_user_input
