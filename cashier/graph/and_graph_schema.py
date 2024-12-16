from __future__ import annotations

from collections import defaultdict
from typing import Any, List, Optional, Set, Tuple, Type, Union
import copy

from pydantic import BaseModel

from cashier.graph.base.base_graph import BaseGraphSchema
from cashier.graph.base.base_terminable_graph import (
    BaseTerminableGraph,
    BaseTerminableGraphSchema,
    should_change_node_schema,
)
from cashier.graph.conversation_node import (
    ConversationNode,
    ConversationNodeSchema,
    Direction,
)
from cashier.graph.edge_schema import EdgeSchema
from cashier.graph.mixin.has_status_mixin import Status
from cashier.tool.tool_registry import ToolRegistry


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
        BaseTerminableGraphSchema.__init__(
            self,
            description,
            node_schemas,
            state_schema,
            run_assistant_turn_before_transition,
        )
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

        all_tool_defs = []
        all_tool_defs = []
        for node_schema in node_schemas:
            all_tool_defs.extend(
                list(node_schema.tool_registry.openai_tool_name_to_tool_def.values())
            )
        self.tool_registry = ToolRegistry(all_tool_defs)
        self.node_prompt = description

        self.default_edge_schema_id_to_edge_schema = {
            edge_schema.id: edge_schema for edge_schema in self.default_edge_schemas
        }
        self.default_from_node_schema_id_to_edge_schema = defaultdict(list)
        for edge_schema in self.default_edge_schemas:
            self.default_from_node_schema_id_to_edge_schema[
                edge_schema.from_node_schema.id
            ].append(edge_schema)

    def create_node(self, input, request, prev_node=None):
        if prev_node is not None:
            return copy.deepcopy(prev_node)
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
        self.visited_node_schemas = set()

    def compute_init_node_edge_schema(
        self,
    ):
        node_schema = self.schema.default_start_node_schema
        edge_schema = None
        # next_edge_schemas = self.schema.default_from_node_schema_id_to_edge_schema[
        #     node_schema.id
        # ]
        # passed_check = True
        # while passed_check:
        #     passed_check = False
        #     for next_edge_schema in next_edge_schemas:
        #         if next_edge_schema.check_transition_config(
        #             self.state,
        #             None,
        #             None,
        #             check_resettable_fields=False,
        #         ):
        #             passed_check = True
        #             node_schema = next_edge_schema.to_node_schema
        #             edge_schema = next_edge_schema
        #             next_edge_schemas = (
        #                 self.schema.default_from_node_schema_id_to_edge_schema[
        #                     node_schema.id
        #                 ]
        #             )
        #             break

        return node_schema, edge_schema

    def handle_wait(
        self,
        fwd_skip_edge_schemas: Set[EdgeSchema],
        TC,
    ) -> Union[Tuple[EdgeSchema, ConversationNodeSchema], Tuple[None, None]]:
        remaining_node_schemas = (
            set(self.schema.node_schemas) - self.visited_node_schemas
        )
        remaining_node_schemas.add(self.curr_node.schema)
        node_schema_id = should_change_node_schema(
            TC, self.curr_node.schema, remaining_node_schemas, True
        )
        node_schema = None
        if node_schema_id is not None:
            node_schema = self.schema.node_schema_id_to_node_schema[node_schema_id]

        return None, node_schema

    def compute_next_edge_schemas_for_init_conversation_core(self):
        return set(
            self.schema.default_from_node_schema_id_to_edge_schema.get(
                self.curr_node.schema.id, []
            )
        )

    def check_self_completion(self, fn_call, is_fn_call_success):
        self_completion = (
            len(self.visited_node_schemas) == len(self.schema.node_schemas)
            and self.curr_node.status == Status.TRANSITIONING
        )
        if self_completion:
            # TODO: this is bad. refactor this. also, generalize this to all graphs
            parent_node = self
            curr_node = self.curr_node
            old_state = parent_node.state.model_dump()
            set_fields = parent_node.state.model_fields_set
            child_state = curr_node.state.model_dump(
                exclude=curr_node.state.resettable_fields
            )
            new_state = old_state | child_state
            new_set_fields = set_fields | child_state.keys()
            parent_node.state = parent_node.state.__class__(**new_state)
            parent_node.state.__pydantic_fields_set__ = new_set_fields
            self.mark_as_transitioning()
        return self_completion

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
        prev_fn_caller=None,
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
            prev_fn_caller=None,
        )
        self.visited_node_schemas.add(node_schema)
        if edge_schema:
            self.add_edge_schema(edge_schema)
