from __future__ import annotations

from typing import Any, List, Optional, Type

from pydantic import BaseModel

from cashier.graph.base.base_graph import BaseGraphSchema
from cashier.graph.base.base_terminable_graph import (
    BaseTerminableGraph,
    BaseTerminableGraphSchema,
)
from cashier.graph.conversation_node import ConversationNode, ConversationNodeSchema
from cashier.graph.edge_schema import EdgeSchema
from cashier.graph.mixin.has_status_mixin import Status


class ANDGraphSchema(BaseTerminableGraphSchema):
    def __init__(
        self,
        description: str,
        node_schemas: List[ConversationNodeSchema],
        state_schema: Type[BaseModel],
        default_start_node_schema: ConversationNodeSchema,
        default_edge_schemas: Optional[List[EdgeSchema]] = None,
        run_assistant_turn_before_transition: bool = False,
    ):
        for node_schema in node_schemas:
            assert node_schema.completion_config is not None
        self.default_start_node_schema = default_start_node_schema
        self.default_edge_schemas = default_edge_schemas
        if not default_edge_schemas:
            self.default_edge_schemas = []
            for i in range(1, len(node_schemas)):
                from_node_schema = node_schemas[i - 1]
                to_node_schema = node_schemas[i]
                edge_schema = EdgeSchema(
                    from_node_schema=from_node_schema,
                    to_node_schema=to_node_schema,
                )
                self.default_edge_schemas.append(edge_schema)

        self.end_node_schema = node_schemas[-1]

        BaseTerminableGraphSchema.__init__(
            self,
            description,
            node_schemas,
            state_schema,
            run_assistant_turn_before_transition,
        )

        self.default_from_node_schema_id_to_edge_schema = {
            edge_schema.from_node_schema.id: edge_schema
            for edge_schema in self.default_edge_schemas
        }

    @property
    def start_node_schema(self):
        return self.default_start_node_schema

    def create_node(self, input, last_msg, edge_schema, prev_node, direction, request):
        return ANDGraph(
            input=input,
            request=request,
            schema=self,
        )

    def get_edge_schemas(self):
        return self.default_edge_schemas


class ANDGraph(BaseTerminableGraph):
    def __init__(
        self,
        input: Any,
        request: str,
        schema: BaseGraphSchema,
    ):
        super().__init__(input, request, schema)
        self.visited_node_schemas = set()

    def get_next_edge_schema(self):
        return self.schema.default_from_node_schema_id_to_edge_schema.get(
            self.curr_node.schema.id, None
        )

    def is_completed(self, fn_call, is_fn_call_success):
        return (
            len(self.visited_node_schemas) == len(self.schema.node_schemas)
            and self.curr_node.status == Status.INTERNALLY_COMPLETED
        )

    def post_node_init(
        self,
        edge_schema: Optional[EdgeSchema],
        prev_node: Optional[ConversationNode],
        TC,
        is_skip: bool = False,
    ) -> None:
        super().post_node_init(
            edge_schema,
            prev_node,
            TC,
            is_skip,
        )
        self.visited_node_schemas.add(self.curr_node.schema)
        if edge_schema and edge_schema not in self.edge_schemas:
            self.add_edge_schema(edge_schema)
