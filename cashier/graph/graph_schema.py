from __future__ import annotations

from typing import Any, List, Type

from pydantic import BaseModel

from cashier.graph.edge_schema import EdgeSchema
from cashier.graph.mixin.auto_mixin_init import AutoMixinInit
from cashier.graph.mixin.graph_mixin import HasGraphMixin, HasGraphSchemaMixin
from cashier.graph.mixin.has_id_mixin import HasIdMixin
from cashier.graph.node_schema import NodeSchema
from cashier.graph.state import HasStateMixin, HasStateSchemaMixin


class GraphSchema(
    HasIdMixin, HasGraphSchemaMixin, HasStateSchemaMixin, metaclass=AutoMixinInit
):
    def __init__(
        self,
        output_schema: Type[BaseModel],
        description: str,
        start_node_schema: NodeSchema,
        last_node_schema: NodeSchema,
        edge_schemas: List[EdgeSchema],
        node_schemas: List[NodeSchema],
        state_pydantic_model: Type[BaseModel],
    ):
        self.output_schema = output_schema
        self.start_node_schema = start_node_schema
        self.last_node_schema = last_node_schema


class Graph(HasGraphMixin, HasStateMixin):
    def __init__(
        self,
        input: Any,
        graph_schema: HasGraphSchemaMixin,
    ):
        HasGraphMixin.__init__(self, graph_schema)
        HasStateMixin.__init__(self, graph_schema.state_pydantic_model(**(input or {})))

    def compute_init_node_edge_schema(
        self,
    ):
        node_schema = self.graph_schema.start_node_schema
        edge_schema = None
        next_edge_schemas = self.graph_schema.from_node_schema_id_to_edge_schema[
            node_schema.id
        ]
        while next_edge_schemas:
            passed_check = False
            for next_edge_schema in next_edge_schemas:
                if next_edge_schema.check_transition_config(
                    self.state,
                    None,
                    None,
                    check_resettable_fields=False,
                ):
                    passed_check = True
                    node_schema = next_edge_schema.to_node_schema
                    edge_schema = next_edge_schema
                    next_edge_schemas = (
                        self.graph_schema.from_node_schema_id_to_edge_schema[
                            node_schema.id
                        ]
                    )

            if not passed_check:
                break

        return node_schema, edge_schema
