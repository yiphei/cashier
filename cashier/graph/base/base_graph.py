import json
from collections import defaultdict, deque
from typing import Any, Callable, List, Literal, Optional, Set, Tuple, overload

from colorama import Style

from cashier.graph.base.base_executable import BaseExecutable
from cashier.graph.conversation_node import (
    ConversationNode,
    ConversationNodeSchema,
    Direction,
)
from cashier.graph.edge_schema import Edge, EdgeSchema, FwdSkipType
from cashier.graph.mixin.has_id_mixin import HasIdMixin
from cashier.graph.mixin.has_status_mixin import HasStatusMixin, Status
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


class BaseGraph(BaseExecutable, HasStatusMixin, HasIdMixin):
    def __init__(self, input: Any, schema: BaseGraphSchema, request=None):
        HasStatusMixin.__init__(self)
        HasIdMixin.__init__(self)
        self.input = input
        self.schema = schema
        self.edge_schema_id_to_edges = defaultdict(list)
        self.from_node_schema_id_to_last_edge_schema_id = defaultdict(lambda: None)
        self.to_node_id_to_edge = defaultdict(lambda: None)
        self.edge_schema_id_to_from_node = {}
        self.curr_node = None
        self.next_edge_schemas: Set[EdgeSchema] = set()
        self.bwd_skip_edge_schemas: Set[EdgeSchema] = set()
        self.request = request
        self.new_edge_schema = None
        self.new_node_schema = None
        self.local_transition_queue = deque()
        self.parent = None

        self.edge_schemas = schema.edge_schemas
        self.edge_schema_id_to_edge_schema = {
            edge_schema.id: edge_schema for edge_schema in self.edge_schemas
        }
        self.from_node_schema_id_to_edge_schema = defaultdict(list)
        for edge_schema in self.edge_schemas:
            self.from_node_schema_id_to_edge_schema[
                edge_schema.from_node_schema.id
            ].append(edge_schema)

    @property
    def transition_queue(self):
        from cashier.graph.graph_schema import Graph

        sub_queue = (
            self.curr_node.transition_queue
            if isinstance(self.curr_node, Graph)
            else deque()
        )
        return sub_queue + self.local_transition_queue

    @property
    def curr_conversation_node(self):
        from cashier.graph.graph_schema import Graph

        return (
            self.curr_node.curr_conversation_node
            if isinstance(self.curr_node, Graph)
            else self.curr_node
        )

    def add_fwd_edge(
        self,
        from_node: ConversationNode,
        to_node: ConversationNode,
        edge_schema: EdgeSchema,
    ) -> None:
        edge = Edge(from_node, to_node, edge_schema)
        self.edge_schema_id_to_edges[edge_schema.id].append(edge)
        self.to_node_id_to_edge[to_node.id] = edge
        self.from_node_schema_id_to_last_edge_schema_id[from_node.schema.id] = (
            edge_schema.id
        )
        self.edge_schema_id_to_from_node[edge_schema.id] = from_node

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

    def get_last_edge_schema_by_from_node_schema_id(
        self, node_schema_id: int
    ) -> Optional[EdgeSchema]:
        edge_schema_id = self.from_node_schema_id_to_last_edge_schema_id[node_schema_id]
        return (
            self.schema.edge_schema_id_to_edge_schema[edge_schema_id]
            if edge_schema_id
            else None
        )

    def get_prev_node(
        self, edge_schema: Optional[EdgeSchema], direction: Direction
    ) -> Optional[ConversationNode]:
        if (
            edge_schema
            and self.get_edge_by_edge_schema_id(edge_schema.id, raise_if_none=False)
            is not None
        ):
            edge = self.get_edge_by_edge_schema_id(edge_schema.id)
            return edge.to_node if direction == Direction.FWD else edge.from_node
        else:
            return None

    def compute_bwd_skip_edge_schemas(self) -> Set[EdgeSchema]:
        from_node = self.curr_node
        new_edge_schemas = set()
        curr_bwd_skip_edge_schemas = self.bwd_skip_edge_schemas
        while self.to_node_id_to_edge[from_node.id] is not None:
            edge = self.to_node_id_to_edge[from_node.id]
            if edge.schema in curr_bwd_skip_edge_schemas:
                break
            new_edge_schemas.add(edge.schema)
            assert from_node == edge.to_node
            from_node = edge.from_node

        self.bwd_skip_edge_schemas = new_edge_schemas | curr_bwd_skip_edge_schemas

    def compute_fwd_skip_edge_schemas(self) -> Set[EdgeSchema]:
        start_node = self.curr_node
        start_edge_schemas = self.next_edge_schemas
        fwd_jump_edge_schemas = set()
        edge_schemas = deque(start_edge_schemas)
        while edge_schemas:
            edge_schema = edge_schemas.popleft()
            if (
                self.get_edge_by_edge_schema_id(edge_schema.id, raise_if_none=False)
                is not None
            ):
                edge = self.get_edge_by_edge_schema_id(edge_schema.id)
                from_node = edge.from_node
                to_node = edge.to_node
                if from_node.schema == start_node.schema:
                    from_node = start_node

                if edge_schema.can_skip(
                    self.state,  # TODO: this class does not explicitly have a state
                    from_node,
                    to_node,
                    self.is_prev_from_node_completed(
                        edge_schema, from_node == start_node
                    ),
                )[0]:
                    fwd_jump_edge_schemas.add(edge_schema)
                    next_edge_schema = self.get_last_edge_schema_by_from_node_schema_id(
                        to_node.schema.id
                    )
                    if next_edge_schema:
                        edge_schemas.append(next_edge_schema)

        return fwd_jump_edge_schemas

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

                next_next_edge_schema = (
                    self.get_last_edge_schema_by_from_node_schema_id(to_node.schema.id)
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
                from_node = self.edge_schema_id_to_from_node[edge_schema.id]
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

            self.edge_schema_id_to_from_node[edge_schema.id] = new_node

    def init_conversation_core(
        self,
        node_schema: ConversationNodeSchema,
        edge_schema: Optional[EdgeSchema],
        input: Any,
        last_msg: Optional[str],
        prev_node: Optional[ConversationNode],
        direction: Direction,
        TC,
        is_skip: bool = False,
    ) -> None:
        logger.debug(
            f"[NODE_SCHEMA] Initializing node with {Style.BRIGHT}node_schema_id: {node_schema.id}{Style.NORMAL}"
        )
        new_node = node_schema.create_node(
            input, last_msg, edge_schema, prev_node, direction, self.request  # type: ignore
        )
        new_node.parent = self

        TC.add_node_turn(
            new_node,
            is_skip=is_skip,
        )
        MessageDisplay.print_msg("system", new_node.prompt)

        if node_schema.first_turn and prev_node is None:
            assert isinstance(node_schema.first_turn, AssistantTurn)
            TC.add_assistant_direct_turn(node_schema.first_turn)
            MessageDisplay.print_msg("assistant", node_schema.first_turn.msg_content)

        # TODO: this is bad. refactor this
        if edge_schema and self.curr_node is not None:
            self.add_edge(self.curr_node, new_node, edge_schema, direction)

        self.curr_node = new_node

    def init_graph_core(
        self,
        node_schema: ConversationNodeSchema,
        edge_schema: Optional[EdgeSchema],
        input: Any,
        last_msg: Optional[str],
        prev_node: Optional[ConversationNode],
        direction: Direction,
        TC,
        is_skip: bool = False,
    ) -> None:
        self.current_graph_schema_idx += 1

        graph = node_schema.create_node(
            input=input, request=self.requests[self.current_graph_schema_idx]
        )
        graph.parent = self

        # TODO: this is bad. refactor this
        if edge_schema and self.curr_node is not None:
            self.add_edge(self.curr_node, graph, edge_schema, direction)

        self.curr_node = graph

        node_schema, edge_schema = graph.compute_init_node_edge_schema()
        self.curr_node.init_next_node(node_schema, edge_schema, TC, None)

    def init_node_core(
        self,
        node_schema: ConversationNodeSchema,
        edge_schema: Optional[EdgeSchema],
        input: Any,
        last_msg: Optional[str],
        prev_node: Optional[ConversationNode],
        direction: Direction,
        TC,
        is_skip: bool = False,
    ) -> None:
        from cashier.graph.graph_schema import GraphSchema

        if isinstance(node_schema, GraphSchema):
            fn = self.init_graph_core
        else:
            fn = self.init_conversation_core

        fn(
            node_schema,
            edge_schema,
            input,
            last_msg,
            prev_node,
            direction,
            TC,
            is_skip,
        )

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

        prev_node = self.get_prev_node(edge_schema, direction)

        self.init_node_core(
            node_schema,
            edge_schema,
            input,
            last_msg,
            prev_node,
            direction,
            TC,
            False,
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
            # TODO: this is bad. refactor this
            if (
                curr_node.state is not None
                and getattr(parent_node, "state", None) is not None
            ):
                old_state = parent_node.state.model_dump()
                set_fields = parent_node.state.model_fields_set
                child_state = curr_node.state.model_dump(
                    exclude=curr_node.state.resettable_fields
                )
                new_state = old_state | child_state
                new_set_fields = set_fields | child_state.keys()
                parent_node.state = parent_node.state.__class__(**new_state)
                parent_node.state.__pydantic_fields_set__ = new_set_fields

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

    def _init_skip_node(
        self,
        node_schema: ConversationNodeSchema,
        edge_schema: EdgeSchema,
        direction,
        last_msg,
        TC,
    ) -> None:

        if direction == Direction.BWD:
            self.bwd_skip_edge_schemas.clear()

        prev_node = self.get_prev_node(edge_schema, direction)
        assert prev_node is not None
        input = prev_node.input

        last_msg = TC.get_asst_message(content_only=True)
        self.init_node_core(
            node_schema,
            edge_schema,
            input,
            last_msg,
            prev_node,
            direction,
            TC,
            True,
        )

    def init_skip_node(
        self,
        node_schema: ConversationNodeSchema,
        edge_schema: EdgeSchema,
        TC,
    ) -> None:
        parent_node = self

        while edge_schema.from_node_schema not in parent_node.schema.node_schemas:
            parent_node = parent_node.curr_node

        direction = Direction.FWD
        if edge_schema and edge_schema.from_node_schema == node_schema:
            direction = Direction.BWD

        last_msg = TC.get_asst_message(content_only=True)
        parent_node._init_skip_node(
            node_schema,
            edge_schema,
            direction,
            last_msg,
            TC,
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
