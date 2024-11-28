from __future__ import annotations

from typing import List, Type

from pydantic import BaseModel

from cashier.graph.edge_schema import EdgeSchema
from cashier.graph.new_classes import (
    GraphMixin,
    GraphSchemaMixin,
    StateMixin,
    StateSchemaMixin,
)
from cashier.graph.node_schema import NodeSchema


class GraphSchema(GraphSchemaMixin, StateSchemaMixin):
    _counter = 0

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
        GraphSchema._counter += 1
        self.id = GraphSchema._counter
        GraphSchemaMixin.__init__(
            self,
            description,
            edge_schemas,
            node_schemas,
        )
        StateSchemaMixin.__init__(self, state_pydantic_model)
        self.output_schema = output_schema
        self.start_node_schema = start_node_schema
        self.last_node_schema = last_node_schema


class Graph(GraphMixin, StateMixin):
    def __init__(
        self,
        graph_schema: GraphSchemaMixin,
    ):
        GraphMixin.__init__(self, graph_schema)
        StateMixin.__init__(self, graph_schema.state_pydantic_model)
