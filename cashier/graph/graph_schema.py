from __future__ import annotations

from collections import defaultdict
from typing import Any, List, Optional, Set, Type

from pydantic import BaseModel

from cashier.graph.base.base_edge_schema import BaseTransitionConfig
from cashier.graph.base.base_graph import BaseGraphSchema
from cashier.graph.base.base_terminable_graph import (
    BaseTerminableGraph,
    BaseTerminableGraphSchema,
)
from cashier.graph.conversation_node import ConversationNodeSchema
from cashier.graph.edge_schema import EdgeSchema
from cashier.prompts.node_schema_selection import NodeSchemaSelectionPrompt
from cashier.turn_container import TurnContainer


def should_change_node_schema(
    TM: TurnContainer,
    current_node_schema: ConversationNodeSchema,
    all_node_schemas: Set[ConversationNodeSchema],
    is_wait: bool,
) -> Optional[int]:
    if len(all_node_schemas) == 1:
        return None
    return NodeSchemaSelectionPrompt.run(
        current_node_schema=current_node_schema,
        tc=TM,
        all_node_schemas=all_node_schemas,
        is_wait=is_wait,
    )


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
        )
        self.edge_schemas = edge_schemas
        self.output_schema = output_schema
        self.start_node_schema = start_node_schema
        self.last_node_schema = last_node_schema
        self.completion_config = completion_config

        self.edge_schema_id_to_edge_schema = {
            edge_schema.id: edge_schema for edge_schema in self.edge_schemas
        }
        self.from_node_schema_id_to_edge_schema = defaultdict(list)
        for edge_schema in self.edge_schemas:
            self.from_node_schema_id_to_edge_schema[
                edge_schema.from_node_schema.id
            ].append(edge_schema)

    def create_node(self, input, request):
        return Graph(
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


class Graph(BaseTerminableGraph):
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
        node_schema = self.schema.start_node_schema
        edge_schema = None
        next_edge_schemas = self.from_node_schema_id_to_edge_schema[node_schema.id]
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
                    next_edge_schemas = self.schema.from_node_schema_id_to_edge_schema[
                        node_schema.id
                    ]
                    break

        return node_schema, edge_schema
