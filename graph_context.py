from collections import defaultdict, deque
from typing import Dict, List

from pydantic import BaseModel, ConfigDict, Field

from graph import Direction, Edge, FwdSkipType, Node
from graph_data import EDGE_SCHEMA_ID_TO_EDGE_SCHEMA


class Graph(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    edge_schema_id_to_edges: Dict[str, List[Edge]] = Field(
        default_factory=lambda: defaultdict(list)
    )
    from_node_schema_id_to_edge_schema_id: Dict[str, str] = Field(
        default_factory=lambda: defaultdict(lambda: None)
    )
    edge_schema_id_to_from_node: Dict[str, None] = Field(
        default_factory=lambda: defaultdict(lambda: None)
    )

    def add_edge(self, from_node, to_node, edge_schema_id):
        self.edge_schema_id_to_edges[edge_schema_id].append(Edge(from_node, to_node))
        self.from_node_schema_id_to_edge_schema_id[from_node.schema.id] = edge_schema_id
        self.edge_schema_id_to_from_node[edge_schema_id] = from_node

    def get_edge_by_edge_schema_id(self, edge_schema_id, idx=-1):
        return (
            self.edge_schema_id_to_edges[edge_schema_id][idx]
            if len(self.edge_schema_id_to_edges[edge_schema_id]) >= abs(idx)
            else None
        )

    def edge_schema_by_from_node_schema_id(self, node_schema_id):
        edge_schema_id = self.from_node_schema_id_to_edge_schema_id[node_schema_id]
        return EDGE_SCHEMA_ID_TO_EDGE_SCHEMA[edge_schema_id] if edge_schema_id else None

    def get_prev_node(self, edge_schema, direction):
        if edge_schema and self.get_edge_by_edge_schema_id(edge_schema.id) is not None:
            from_node, to_node = self.get_edge_by_edge_schema_id(edge_schema.id)
            return to_node if direction == Direction.FWD else from_node
        else:
            return None

    def compute_bwd_skip_edge_schemas(self, start_node, bwd_skip_edge_schemas):
        from_node = start_node
        while from_node.in_edge_schema is not None:
            if from_node.in_edge_schema in bwd_skip_edge_schemas:
                return
            bwd_skip_edge_schemas.add(from_node.in_edge_schema)
            new_from_node, to_node = self.get_edge_by_edge_schema_id(
                from_node.in_edge_schema.id
            )
            assert from_node == to_node
            from_node = new_from_node

    def compute_fwd_skip_edge_schemas(self, start_node, start_edge_schemas):
        fwd_jump_edge_schemas = set()
        edge_schemas = deque(start_edge_schemas)
        while edge_schemas:
            edge_schema = edge_schemas.popleft()
            if self.get_edge_by_edge_schema_id(edge_schema.id) is not None:
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
                    next_edge_schema = self.edge_schema_by_from_node_schema_id(
                        to_node.schema.id
                    )
                    if next_edge_schema:
                        edge_schemas.append(next_edge_schema)

        return fwd_jump_edge_schemas

    def is_prev_from_node_completed(self, edge_schema, is_start_node):
        idx = -1 if is_start_node else -2
        edge = self.get_edge_by_edge_schema_id(edge_schema.id, idx)
        return edge[0].status == Node.Status.COMPLETED if edge else False

    def compute_next_edge_schema(self, start_edge_schema, start_input, curr_node):
        next_edge_schema = start_edge_schema
        edge_schema = start_edge_schema
        input = start_input
        while self.get_edge_by_edge_schema_id(next_edge_schema.id) is not None:
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

                next_next_edge_schema = self.edge_schema_by_from_node_schema_id(
                    to_node.schema.id
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
                        input = edge_schema.new_input_from_state_fn(from_node.state)
                break
            else:
                if from_node != curr_node:
                    input = from_node.input
                break

        return edge_schema, input

    def bridge_edges(self, edge_schema, direction, curr_node, new_node):
        if edge_schema:
            if direction == Direction.FWD:
                immediate_from_node = curr_node
                if edge_schema.from_node_schema != curr_node.schema:
                    from_node = self.edge_schema_id_to_from_node[edge_schema.id]
                    immediate_from_node = from_node
                    while from_node.schema != curr_node.schema:
                        prev_edge_schema = from_node.in_edge_schema
                        from_node, to_node = self.get_edge_by_edge_schema_id(
                            prev_edge_schema.id
                        )

                    self.add_edge(curr_node, to_node, prev_edge_schema.id)

                self.add_edge(immediate_from_node, new_node, edge_schema.id)
            elif direction == Direction.BWD:
                if new_node.in_edge_schema:
                    from_node, _ = self.get_edge_by_edge_schema_id(
                        new_node.in_edge_schema.id
                    )
                    self.add_edge(from_node, new_node, new_node.in_edge_schema.id)

                self.edge_schema_id_to_from_node[edge_schema.id] = new_node
