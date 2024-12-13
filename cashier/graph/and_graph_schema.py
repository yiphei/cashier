from __future__ import annotations

from collections import defaultdict
from typing import Any, List, Optional, Type

from pydantic import BaseModel

from cashier.graph.base.base_graph import BaseGraphSchema
from cashier.graph.base.base_terminable_graph import (
    BaseTerminableGraph,
    BaseTerminableGraphSchema,
)
from cashier.graph.conversation_node import ConversationNode, ConversationNodeSchema, Direction
from cashier.graph.edge_schema import EdgeSchema


class ANDGraphSchema(BaseTerminableGraphSchema):
    def __init__(
        self,
        description: str,
        node_schemas: List[ConversationNodeSchema],
        state_schema: Type[BaseModel],
        default_edge_schemas: List[EdgeSchema],
        default_start_node_schema: ConversationNodeSchema,
        run_assistant_turn_before_transition: bool = False,
    ):
        BaseTerminableGraphSchema.__init__(
            self,
            description,
            node_schemas,
            state_schema,
            run_assistant_turn_before_transition,
        )
        self.default_start_node_schema = default_start_node_schema
        self.default_edge_schemas = default_edge_schemas

        self.default_edge_schema_id_to_edge_schema = {
            edge_schema.id: edge_schema for edge_schema in self.default_edge_schemas
        }
        self.default_from_node_schema_id_to_edge_schema = defaultdict(list)
        for edge_schema in self.default_edge_schemas:
            self.default_from_node_schema_id_to_edge_schema[
                edge_schema.from_node_schema.id
            ].append(edge_schema)

    def create_node(self, input, request):
        return ANDGraph(
            input=input,
            request=request,
            schema=self,
        )

    # TODO: refactor this to be shared with the get_input in ConversationNodeSchema
    def get_input(self, state, edge_schema):
        if edge_schema.new_input_fn is not None:
            return edge_schema.new_input_fn(state)
        else:
            return None


class ANDGraph(BaseTerminableGraph):
    def __init__(
        self,
        input: Any,
        request: str,
        schema: BaseGraphSchema,
    ):
        super().__init__(input, request, schema)

    def compute_init_node_edge_schema(
        self,
    ):
        node_schema = self.schema.default_start_node_schema
        edge_schema = None
        next_edge_schemas = self.schema.default_from_node_schema_id_to_edge_schema[
            node_schema.id
        ]
        passed_check = True
        while passed_check:
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
                        self.schema.default_from_node_schema_id_to_edge_schema[
                            node_schema.id
                        ]
                    )
                    break

        return node_schema, edge_schema

    def compute_next_edge_schemas_for_init_conversation_core(self):
        return set(
            self.schema.default_from_node_schema_id_to_edge_schema.get(self.curr_node.schema.id, [])
        )
    

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
        super().init_conversation_core(
            node_schema,
            edge_schema,
            input,
            last_msg,
            prev_node,
            direction,
            TC,
            is_skip,
        )
        self.add_edge_schema(edge_schema)