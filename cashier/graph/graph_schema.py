from __future__ import annotations

from typing import List, Type

from pydantic import BaseModel

from cashier.graph.auto_mixin_init import AutoMixinInit
from cashier.graph.edge_schema import EdgeSchema
from cashier.graph.graph_mixin import HasGraphMixin, HasGraphSchemaMixin
from cashier.graph.has_id_mixin import HasIdMixin
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
        graph_schema: HasGraphSchemaMixin,
    ):
        HasGraphMixin.__init__(self, graph_schema)
        HasStateMixin.__init__(self, graph_schema.state_pydantic_model)
