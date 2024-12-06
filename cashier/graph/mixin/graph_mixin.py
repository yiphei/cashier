from collections import defaultdict, deque
from typing import Any, List, Literal, Optional, Set, Tuple, overload
from venv import logger

from colorama import Style

from cashier.graph.conversation_node import (
    ConversationNode,
    ConversationNodeSchema,
    Direction,
)
from cashier.graph.edge_schema import Edge, EdgeSchema, FwdSkipType
from cashier.gui import MessageDisplay
from cashier.model.model_turn import AssistantTurn


class HasGraphSchemaMixin:
    def __init__(
        self,
        description: str,
        edge_schemas: List[EdgeSchema],
        node_schemas: List[ConversationNode],
    ):
        self.description = description
        self.edge_schemas = edge_schemas
        self.node_schemas = node_schemas

        self.node_schema_id_to_node_schema = {
            node_schema.id: node_schema for node_schema in self.node_schemas
        }
        self.edge_schema_id_to_edge_schema = {
            edge_schema.id: edge_schema for edge_schema in self.edge_schemas
        }
        self.from_node_schema_id_to_edge_schema = defaultdict(list)
        for edge_schema in self.edge_schemas:
            self.from_node_schema_id_to_edge_schema[
                edge_schema.from_node_schema.id
            ].append(edge_schema)


class HasGraphMixin:
    def __init__(
        self,
        schema: HasGraphSchemaMixin,
    ):
        self.schema = schema
        self.edge_schema_id_to_edges = defaultdict(list)
        self.from_node_schema_id_to_last_edge_schema_id = defaultdict(lambda: None)
        self.edge_schema_id_to_from_node = {}
        self.curr_node = None
        self.next_edge_schemas: Set[EdgeSchema] = set()
        self.bwd_skip_edge_schemas: Set[EdgeSchema] = set()

    def add_fwd_edge(
        self,
        from_node: ConversationNode,
        to_node: ConversationNode,
        edge_schema_id: int,
    ) -> None:
        self.edge_schema_id_to_edges[edge_schema_id].append(Edge(from_node, to_node))
        self.from_node_schema_id_to_last_edge_schema_id[from_node.schema.id] = (
            edge_schema_id
        )
        self.edge_schema_id_to_from_node[edge_schema_id] = from_node

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
            from_node, to_node = self.get_edge_by_edge_schema_id(edge_schema.id)
            return to_node if direction == Direction.FWD else from_node
        else:
            return None

    def compute_bwd_skip_edge_schemas(
        self,
        start_node: ConversationNode,
        curr_bwd_skip_edge_schemas: Set[EdgeSchema],
    ) -> Set[EdgeSchema]:
        from_node = start_node
        new_edge_schemas = set()
        while from_node.in_edge_schema is not None:
            if from_node.in_edge_schema in curr_bwd_skip_edge_schemas:
                break
            new_edge_schemas.add(from_node.in_edge_schema)
            new_from_node, to_node = self.get_edge_by_edge_schema_id(
                from_node.in_edge_schema.id
            )
            assert from_node == to_node
            from_node = new_from_node

        return new_edge_schemas | curr_bwd_skip_edge_schemas

    def compute_fwd_skip_edge_schemas(
        self, start_node: ConversationNode, start_edge_schemas: Set[EdgeSchema]
    ) -> Set[EdgeSchema]:
        fwd_jump_edge_schemas = set()
        edge_schemas = deque(start_edge_schemas)
        while edge_schemas:
            edge_schema = edge_schemas.popleft()
            if (
                self.get_edge_by_edge_schema_id(edge_schema.id, raise_if_none=False)
                is not None
            ):
                from_node, to_node = self.get_edge_by_edge_schema_id(edge_schema.id)
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
        return edge[0].status == ConversationNode.Status.COMPLETED if edge else False

    def compute_next_edge_schema(
        self,
        start_edge_schema: EdgeSchema,
        start_input: Any,
        curr_node: ConversationNode,
    ) -> Tuple[EdgeSchema, Any]:
        next_edge_schema = start_edge_schema
        edge_schema = start_edge_schema
        input = start_input
        while (
            self.get_edge_by_edge_schema_id(next_edge_schema.id, raise_if_none=False)
            is not None
        ):
            from_node, to_node = self.get_edge_by_edge_schema_id(next_edge_schema.id)
            if from_node.schema == curr_node.schema:
                from_node = curr_node

            can_skip, skip_type = next_edge_schema.can_skip(
                self.state,  # TODO: this class does not explicitly have a state
                from_node,
                to_node,
                self.is_prev_from_node_completed(
                    next_edge_schema, from_node == curr_node
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
                if from_node.status != ConversationNode.Status.COMPLETED:
                    input = from_node.input
                else:
                    edge_schema = next_edge_schema
                    if from_node != curr_node:
                        input = edge_schema.new_input_fn(
                            from_node.state, from_node.input
                        )
                break
            else:
                if from_node != curr_node:
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
                    prev_edge_schema = from_node.in_edge_schema
                    from_node, to_node = self.get_edge_by_edge_schema_id(
                        prev_edge_schema.id  # type: ignore
                    )

                self.add_fwd_edge(curr_node, to_node, prev_edge_schema.id)  # type: ignore

            self.add_fwd_edge(immediate_from_node, new_node, edge_schema.id)
        elif direction == Direction.BWD:
            if new_node.in_edge_schema:
                from_node, _ = self.get_edge_by_edge_schema_id(
                    new_node.in_edge_schema.id
                )
                self.add_fwd_edge(from_node, new_node, new_node.in_edge_schema.id)

            self.edge_schema_id_to_from_node[edge_schema.id] = new_node

    @classmethod
    def check_single_transition(cls, state, fn_call, is_fn_call_success, edge_schemas):
        for edge_schema in edge_schemas:
            if edge_schema.check_transition_config(state, fn_call, is_fn_call_success):
                new_edge_schema = edge_schema
                new_node_schema = edge_schema.to_node_schema
                return new_edge_schema, new_node_schema
        return None, None

    def init_conversation_core(
        self,
        node_schema: ConversationNodeSchema,
        edge_schema: Optional[EdgeSchema],
        parent_node,
        input: Any,
        last_msg: Optional[str],
        prev_node: Optional[ConversationNode],
        direction: Direction,
        TC,
        remove_prev_tool_calls,
        is_skip: bool = False,
    ) -> None:
        logger.debug(
            f"[NODE_SCHEMA] Initializing node with {Style.BRIGHT}node_schema_id: {node_schema.id}{Style.NORMAL}"
        )
        new_node = node_schema.create_node(
            input, last_msg, edge_schema, prev_node, direction, parent_node.request if parent_node.__class__.__name__ == "Graph" else None  # type: ignore
        )

        TC.add_node_turn(
            new_node,
            remove_prev_tool_calls=remove_prev_tool_calls,
            is_skip=is_skip,
        )
        MessageDisplay.print_msg("system", new_node.prompt)

        if node_schema.first_turn and prev_node is None:
            assert isinstance(node_schema.first_turn, AssistantTurn)
            TC.add_assistant_direct_turn(node_schema.first_turn)
            MessageDisplay.print_msg("assistant", node_schema.first_turn.msg_content)

        if edge_schema:
            parent_node.add_edge(
                parent_node.curr_node, new_node, edge_schema, direction
            )

        parent_node.curr_node = new_node

        if (
            parent_node.__class__.__name__ == "Graph"
        ):  # TODO: remove this after refactor
            parent_node.next_edge_schemas = set(
                parent_node.schema.from_node_schema_id_to_edge_schema.get(
                    new_node.schema.id, []
                )
            )
            parent_node.bwd_skip_edge_schemas = (
                parent_node.compute_bwd_skip_edge_schemas(
                    parent_node.curr_node, parent_node.bwd_skip_edge_schemas
                )
            )

    def init_next_node(
        self,
        node_schema: ConversationNodeSchema,
        edge_schema: Optional[EdgeSchema],
        TC,
        remove_prev_tool_calls,
        input: Any = None,
    ) -> None:
        curr_node = self.curr_node
        parent_node = self
        if curr_node is not None and edge_schema:
            while curr_node.schema != edge_schema.from_node_schema:
                parent_node = curr_node
                curr_node = curr_node.curr_node

            if isinstance(curr_node, ConversationNode):
                curr_node.mark_as_completed()
            if curr_node.state is not None and parent_node.state is not None:
                old_state = parent_node.state.model_dump()
                new_state = old_state | curr_node.state.model_dump(
                    exclude=curr_node.state.resettable_fields
                )
                parent_node.state = parent_node.state.__class__(**new_state)

        if input is None and edge_schema:
            input = edge_schema.new_input_fn(parent_node.state)

        if edge_schema:
            edge_schema, input = parent_node.compute_next_edge_schema(
                edge_schema, input, curr_node
            )
            node_schema = edge_schema.to_node_schema

        direction = Direction.FWD
        prev_node = parent_node.get_prev_node(edge_schema, direction)

        last_msg = TC.get_user_message(content_only=True)

        self.init_node_core(
            node_schema,
            edge_schema,
            parent_node,
            input,
            last_msg,
            prev_node,
            direction,
            TC,
            remove_prev_tool_calls,
            False,
        )

    def init_skip_node(
        self,
        node_schema: ConversationNodeSchema,
        edge_schema: EdgeSchema,
        TC,
        remove_prev_tool_calls,
    ) -> None:
        parent_node = self

        while edge_schema.from_node_schema not in parent_node.schema.node_schemas:
            parent_node = parent_node.curr_node

        direction = Direction.FWD
        if edge_schema and edge_schema.from_node_schema == node_schema:
            direction = Direction.BWD

        if direction == Direction.BWD:
            parent_node.bwd_skip_edge_schemas.clear()

        prev_node = parent_node.get_prev_node(edge_schema, direction)
        assert prev_node is not None
        input = prev_node.input

        last_msg = TC.get_asst_message(content_only=True)
        self.init_node_core(
            node_schema,
            edge_schema,
            parent_node,
            input,
            last_msg,
            prev_node,
            direction,
            TC,
            remove_prev_tool_calls,
            True,
        )
