from __future__ import annotations

import json
from typing import Any, List, Type

from cashier.graph.base.base_edge_schema import BaseEdgeSchema
from cashier.graph.base.base_graph import BaseGraph, BaseGraphSchema
from cashier.graph.conversation_node import ConversationNodeSchema
from cashier.graph.edge_schema import EdgeSchema
from cashier.graph.graph_schema import Graph
from cashier.graph.mixin.auto_mixin_init import AutoMixinInit
from cashier.graph.mixin.has_id_mixin import HasIdMixin
from cashier.graph.mixin.has_status_mixin import Status
from cashier.logger import logger
from cashier.model.model_util import CustomJSONEncoder, create_think_fn_call
from cashier.prompts.graph_schema_addition import GraphSchemaAdditionPrompt
from cashier.prompts.graph_schema_selection import GraphSchemaSelectionPrompt
from cashier.prompts.node_system import NodeSystemPrompt
from cashier.prompts.off_topic import OffTopicPrompt


class RequestGraph(BaseGraph):

    def __init__(
        self,
        input: Any,
        schema: BaseGraphSchema,
    ):
        super().__init__(schema)
        self.input = input
        self.requests = []
        self.graph_schema_sequence = []
        self.current_graph_schema_idx = -1
        self.graph_schema_id_to_task = {}

    def get_graph_schemas(self, request):
        agent_selections = GraphSchemaSelectionPrompt.run(
            request=request, graph_schemas=self.schema.node_schemas
        )
        self.graph_schema_sequence = []
        self.requests = []
        self.graph_schema_id_to_task = {}
        self.current_graph_schema_idx = -1
        for agent_selection in agent_selections:
            self.graph_schema_sequence.append(
                self.schema.node_schema_id_to_node_schema[agent_selection.agent_id]
            )
            self.requests.append(agent_selection.task)
            self.graph_schema_id_to_task[agent_selection.agent_id] = (
                agent_selection.task
            )

        logger.debug(
            f"agent_selections: {json.dumps(agent_selections, cls=CustomJSONEncoder, indent=4)}"
        )

    def add_tasks(self, request, tc):
        agent_selection = GraphSchemaAdditionPrompt.run(
            graph_schemas=self.schema.node_schemas,
            curr_agent_id=self.graph_schema_sequence[self.current_graph_schema_idx].id,
            curr_task=self.requests[self.current_graph_schema_idx],
            tc=tc,
            all_tasks=self.requests,
        )

        logger.debug(
            f"agent_selection: {json.dumps(agent_selection, cls=CustomJSONEncoder, indent=4)}"
        )
        if agent_selection is not None:
            self.graph_schema_sequence.append(
                self.schema.node_schema_id_to_node_schema[agent_selection.agent_id]
            )
            self.requests.append(agent_selection.task)
            self.graph_schema_id_to_task[agent_selection.agent_id] = (
                agent_selection.task
            )

        return True if agent_selection else False

    def handle_user_turn(self, msg, TC, model_provider):
        if isinstance(self.curr_node, Graph):
            if not OffTopicPrompt.run(
                current_node_schema=self.curr_node.curr_node.schema,
                tc=TC,
            ):
                has_new_task = self.add_tasks(msg, TC)
                if has_new_task:
                    fake_fn_call = create_think_fn_call(
                        "At least part of the customer request/question is off-topic for the current conversation and will actually be addressed later. According to the policies, I must tell the customer that 1) their off-topic request/question will be addressed later and 2) we must finish the current business before we can get to it. I must refuse to engage with the off-topic request/question in any way."
                    )
                    TC.add_assistant_turn(
                        None,
                        model_provider,
                        self.curr_node.curr_node.schema.tool_registry,
                        [fake_fn_call],
                        {fake_fn_call.id: None},
                    )
                else:
                    self.curr_node.handle_user_turn(
                        msg,
                        TC,
                        model_provider,
                        run_off_topic_check=False,
                    )
            self.curr_node.curr_node.update_first_user_message()  # TODO: remove this after refactor
        else:
            self.get_graph_schemas(msg)
            if len(self.graph_schema_sequence) > 0:
                self.init_next_node(
                    self.graph_schema_sequence[0],
                    None,
                    TC,
                    None,
                )

    def check_self_transition(
        self,
        fn_call,
        is_fn_call_success,
        parent_edge_schemas=None,
        new_edge_schema=None,
        new_node_schema=None,
    ):
        edge_schemas = self.from_node_schema_id_to_edge_schema[self.curr_node.schema.id]
        if self.curr_node.status == Status.TRANSITIONING:
            if len(edge_schemas) == 1:
                new_edge_schema = edge_schemas[0]
                new_node_schema = new_edge_schema.to_node_schema
            else:
                new_edge_schema = None
                new_node_schema = self.schema.default_node_schema
        return new_edge_schema, new_node_schema

    def get_next_edge_schema(self):
        return self.from_node_schema_id_to_edge_schema[self.curr_node.schema.id]


class RequestGraphSchema(BaseGraphSchema):
    def __init__(
        self,
        node_prompt: str,
        node_system_prompt: Type[NodeSystemPrompt],
        description: str,
        edge_schemas: List[EdgeSchema],
        node_schemas: List[ConversationNodeSchema],
    ):
        super().__init__(description, node_schemas)
        self.edge_schemas = edge_schemas
        self.start_node_schema = ConversationNodeSchema(node_prompt, node_system_prompt)
        self.default_node_schema = ConversationNodeSchema(
            "You have just finished helping the customer with their requests. Ask if they need anything else.",
            node_system_prompt,
        )


class GraphEdgeSchema(BaseEdgeSchema, HasIdMixin, metaclass=AutoMixinInit):
    pass
