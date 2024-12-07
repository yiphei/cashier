from __future__ import annotations

import json
from typing import Any, List, Type

from cashier.graph.base.base_edge_schema import BaseEdgeSchema
from cashier.graph.base.graph_base import BaseGraph, BaseGraphSchema
from cashier.graph.conversation_node import ConversationNodeSchema
from cashier.graph.edge_schema import EdgeSchema
from cashier.graph.graph_schema import Graph
from cashier.graph.mixin.auto_mixin_init import AutoMixinInit
from cashier.graph.mixin.has_id_mixin import HasIdMixin
from cashier.logger import logger
from cashier.model.model_util import CustomJSONEncoder, FunctionCall
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
        self.tasks = []
        self.graph_schema_sequence = []
        self.current_graph_schema_idx = -1
        self.graph_schema_id_to_task = {}
        self.curr_conversation_node = None

    def get_graph_schemas(self, request):
        agent_selections = GraphSchemaSelectionPrompt.run(
            "claude-3.5", request=request, graph_schemas=self.schema.node_schemas
        )
        for agent_selection in agent_selections:
            self.graph_schema_sequence.append(
                self.schema.node_schema_id_to_node_schema[agent_selection.agent_id]
            )
            self.tasks.append(agent_selection.task)
            self.graph_schema_id_to_task[agent_selection.agent_id] = (
                agent_selection.task
            )

        logger.debug(
            f"agent_selections: {json.dumps(agent_selections, cls=CustomJSONEncoder, indent=4)}"
        )

    def add_tasks(self, request, tc):
        agent_selection = GraphSchemaAdditionPrompt.run(
            "claude-3.5",
            graph_schemas=self.schema.node_schemas,
            curr_agent_id=self.graph_schema_sequence[self.current_graph_schema_idx].id,
            curr_task=self.tasks[self.current_graph_schema_idx],
            tc=tc,
            all_tasks=self.tasks,
        )

        logger.debug(
            f"agent_selection: {json.dumps(agent_selection, cls=CustomJSONEncoder, indent=4)}"
        )
        if agent_selection is not None:
            self.graph_schema_sequence.append(
                self.schema.node_schema_id_to_node_schema[agent_selection.agent_id]
            )
            self.tasks.append(agent_selection.task)
            self.graph_schema_id_to_task[agent_selection.agent_id] = (
                agent_selection.task
            )

        return True if agent_selection else False

    def handle_user_turn(self, msg, TC, model_provider, remove_prev_tool_calls):
        if isinstance(self.curr_node, Graph):
            if not OffTopicPrompt.run(
                "claude-3.5",
                current_node_schema=self.curr_node.curr_node.schema,
                tc=TC,
            ):
                has_new_task = self.add_tasks(msg, TC)
                if has_new_task:
                    fake_fn_call = FunctionCall.create(
                        api_id_model_provider=None,
                        api_id=None,
                        name="think",
                        args={
                            "thought": "At least part of the customer request/question is off-topic for the current conversation and will actually be addressed later. According to the policies, I must tell the customer that 1) their off-topic request/question will be addressed later and 2) we must finish the current business before we can get to it. I must refuse to engage with the off-topic request/question in any way."
                        },
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
                        remove_prev_tool_calls,
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
                    remove_prev_tool_calls,
                    None,
                )

    def check_self_transition(self, fn_call, is_fn_call_success):
        fake_fn_call = None
        edge_schemas = self.schema.from_node_schema_id_to_edge_schema[
            self.curr_node.schema.id
        ]
        new_edge_schema, new_node_schema = self.check_node_transition(
            self.curr_node.state, fn_call, is_fn_call_success, edge_schemas
        )
        if new_node_schema is not None and isinstance(self.curr_node, Graph) and self.current_graph_schema_idx < len(self.tasks)-1:
            fake_fn_call = FunctionCall.create(
                api_id_model_provider=None,
                api_id=None,
                name="think",
                args={
                    "thought": f"I just completed the current request. The next request to be addressed is: {self.tasks[self.current_graph_schema_idx + 1]}. I must explicitly inform the customer that the current request is completed and that I will address the next request right away. Only after I informed the customer do I receive the tools to address the next request."
                },
            )
        return new_edge_schema, new_node_schema, False, fake_fn_call, None


class RequestGraphSchema(BaseGraphSchema):
    def __init__(
        self,
        node_prompt: str,
        node_system_prompt: Type[NodeSystemPrompt],
        description: str,
        edge_schemas: List[EdgeSchema],
        node_schemas: List[ConversationNodeSchema],
    ):
        super().__init__(description, edge_schemas, node_schemas)
        self.start_node_schema = ConversationNodeSchema(node_prompt, node_system_prompt)


class GraphEdgeSchema(BaseEdgeSchema, HasIdMixin, metaclass=AutoMixinInit):
    pass
