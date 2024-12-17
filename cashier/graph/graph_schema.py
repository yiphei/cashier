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
        last_node_schema: ConversationNodeSchema,
        edge_schemas: List[EdgeSchema],
        node_schemas: List[ConversationNodeSchema],
        state_schema: Type[BaseModel],
        completion_config: BaseTransitionConfig,
        run_assistant_turn_before_transition: bool = False,
    ):
        BaseTerminableGraphSchema.__init__(
            self,
            description,
            node_schemas,
            state_schema,
            run_assistant_turn_before_transition,
            completion_config,
        )
        self.edge_schemas = edge_schemas
        self.output_schema = output_schema
        self.start_node_schema = start_node_schema
        self.last_node_schema = last_node_schema

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

    def compute_init_node_edge_schema(
        self,
    ):
        from cashier.graph.and_graph_schema import ANDGraphSchema

        node_schema = self.schema.start_node_schema
        edge_schema = None
        next_edge_schema = self.from_node_schema_id_to_edge_schema[node_schema.id]
        passed_check = True
        while passed_check:
            passed_check = False
            if next_edge_schema.check_transition_config(
                self.state,
                None,
                None,
                check_resettable_fields=False,
            ) and not isinstance(
                next_edge_schema.from_node_schema, ANDGraphSchema
            ):  # TODO: fix this
                passed_check = True
                node_schema = next_edge_schema.to_node_schema
                edge_schema = next_edge_schema
                next_edge_schema = self.schema.from_node_schema_id_to_edge_schema.get(
                    node_schema.id, None
                )
                break

        return node_schema, edge_schema
