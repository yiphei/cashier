from __future__ import annotations

from typing import Any, List, Type

from pydantic import BaseModel

from cashier.graph.base.base_edge_schema import BaseTransitionConfig
from cashier.graph.base.base_graph import BaseGraphSchema
from cashier.graph.base.base_terminable_graph import (
    BaseTerminableGraph,
    BaseTerminableGraphSchema,
)
from cashier.graph.conversation_node import ConversationNodeSchema
from cashier.graph.edge_schema import EdgeSchema


class GraphSchema(BaseTerminableGraphSchema):
    def __init__(
        self,
        output_schema: Type[BaseModel],
        description: str,
        start_node_schema: ConversationNodeSchema,
        end_node_schema: ConversationNodeSchema,
        edge_schemas: List[EdgeSchema],
        node_schemas: List[ConversationNodeSchema],
        state_schema: Type[BaseModel],
        completion_config: BaseTransitionConfig,
        run_assistant_turn_before_transition: bool = False,
    ):
        self.edge_schemas = edge_schemas
        self.start_node_schema = start_node_schema
        self.end_node_schema = end_node_schema
        BaseTerminableGraphSchema.__init__(
            self,
            description,
            node_schemas,
            state_schema,
            run_assistant_turn_before_transition,
            completion_config,
        )
        self.output_schema = output_schema

    def create_node(self, input, last_msg, edge_schema, prev_node, direction, request):
        return Graph(
            input=input,
            request=request,
            schema=self,
        )


class Graph(BaseTerminableGraph):
    def __init__(
        self,
        input: Any,
        request: str,
        schema: BaseGraphSchema,
    ):
        super().__init__(input, request, schema, schema.edge_schemas)
