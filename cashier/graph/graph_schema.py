from __future__ import annotations

from collections import defaultdict, deque
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    overload,
)

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cashier.graph.edge_schema import Edge, EdgeSchema, FwdSkipType
from cashier.graph.node_schema import Direction, Node, NodeSchema


def _get_next_id() -> int:
    if not hasattr(_get_next_id, "counter"):
        _get_next_id.counter = 0
    _get_next_id.counter += 1
    return _get_next_id.counter


class GraphSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: Optional[int] = Field(default_factory=_get_next_id)
    output_schema: Type[BaseModel]
    description: str
    start_node_schema: NodeSchema
    last_node_schema: NodeSchema
    last_node_success_fn: Callable
    edge_schemas: List[EdgeSchema]
    node_schemas: List[NodeSchema]
    state_schema: Type[BaseModel]

    @model_validator(mode="after")
    def init_computed_fields(self) -> GraphSchema:
        self._node_schema_id_to_node_schema = {
            node_schema.id: node_schema for node_schema in self.node_schemas
        }
        self._edge_schema_id_to_edge_schema = {
            edge_schema.id: edge_schema for edge_schema in self.edge_schemas
        }
        self._from_node_schema_id_to_edge_schema = defaultdict(list)
        for edge_schema in self.edge_schemas:
            self._from_node_schema_id_to_edge_schema[
                edge_schema.from_node_schema.id
            ].append(edge_schema)

        return self

    @property
    def node_schema_id_to_node_schema(self) -> Dict[int, NodeSchema]:
        return self._node_schema_id_to_node_schema

    @property
    def edge_schema_id_to_edge_schema(self) -> Dict[int, EdgeSchema]:
        return self._edge_schema_id_to_edge_schema

    @property
    def from_node_schema_id_to_edge_schema(self) -> Dict[int, List[EdgeSchema]]:
        return self._from_node_schema_id_to_edge_schema


class Graph(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph_schema: GraphSchema
    edge_schema_id_to_edges: Dict[int, List[Edge]] = Field(
        default_factory=lambda: defaultdict(list)
    )
    from_node_schema_id_to_last_edge_schema_id: Dict[int, int] = Field(
        default_factory=lambda: defaultdict(lambda: None)
    )
    edge_schema_id_to_from_node: Dict[int, Node] = Field(default_factory=dict)
    state: BaseModel

    def __init__(self, **kwargs):
        if "state" not in kwargs:
            kwargs["state"] = kwargs["graph_schema"].state_schema()
        else:
            raise ValueError("state must not be provided")
        super().__init__(**kwargs)

    def add_fwd_edge(self, from_node: Node, to_node: Node, edge_schema_id: int) -> None:
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
            self.graph_schema.edge_schema_id_to_edge_schema[edge_schema_id]
            if edge_schema_id
            else None
        )

    def get_prev_node(
        self, edge_schema: Optional[EdgeSchema], direction: Direction
    ) -> Optional[Node]:
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
        self, start_node: Node, curr_bwd_skip_edge_schemas: Set[EdgeSchema]
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
        self, start_node: Node, start_edge_schemas: Set[EdgeSchema]
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
        return edge[0].status == Node.Status.COMPLETED if edge else False

    def compute_next_edge_schema(
        self, start_edge_schema: EdgeSchema, start_input: Any, curr_node: Node
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
                if from_node.status != Node.Status.COMPLETED:
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
        curr_node: Node,
        new_node: Node,
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
