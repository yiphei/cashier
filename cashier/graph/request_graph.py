from __future__ import annotations

import json
from typing import Any, List, Optional, Type

from cashier.graph.edge_schema import EdgeSchema
from cashier.graph.graph_schema import Graph, GraphSchema
from cashier.graph.mixin.auto_mixin_init import AutoMixinInit
from cashier.graph.mixin.base_edge_schema import BaseEdgeSchema
from cashier.graph.mixin.graph_mixin import HasGraphMixin, HasGraphSchemaMixin
from cashier.graph.mixin.has_id_mixin import HasIdMixin
from cashier.graph.node_schema import NodeSchema
from cashier.logger import logger
from cashier.model.model_util import CustomJSONEncoder, FunctionCall
from cashier.prompts.graph_schema_addition import GraphSchemaAdditionPrompt
from cashier.prompts.graph_schema_selection import GraphSchemaSelectionPrompt
from cashier.prompts.node_system import NodeSystemPrompt
from cashier.prompts.off_topic import OffTopicPrompt


class Ref:
    def __init__(self, obj, attr):
        self._obj = obj
        self._attr = attr

    def __get__(self, instance, owner):
        return getattr(self._obj, self._attr)

    def __set__(self, instance, value):
        setattr(self._obj, self._attr, value)

    # Optional: make it behave more like a regular variable
    def __repr__(self):
        return repr(self.__get__(None, None))

    def __getattr__(self, name):
        # Get the current value and access the requested attribute
        value = self.__get__(None, None)
        return getattr(value, name)


class RequestGraph(HasGraphMixin):

    def __init__(
        self,
        input: Any,
        graph_schema: HasGraphSchemaMixin,
    ):
        HasGraphMixin.__init__(self, graph_schema)
        self.tasks = []
        self.graph_schema_sequence = []
        self.current_graph_schema_idx = -1
        self.graph_schema_id_to_task = {}
        self.curr_executable = None

    def get_graph_schemas(self, request):
        agent_selections = GraphSchemaSelectionPrompt.run(
            "claude-3.5", request=request, graph_schemas=self.graph_schema.node_schemas
        )
        for agent_selection in agent_selections:
            self.graph_schema_sequence.append(
                self.graph_schema.node_schema_id_to_node_schema[
                    agent_selection.agent_id
                ]
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
            graph_schemas=self.graph_schema.node_schemas,
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
                self.graph_schema.node_schema_id_to_node_schema[
                    agent_selection.agent_id
                ]
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

    def init_next_node(
        self,
        node_schema: NodeSchema,
        edge_schema: Optional[EdgeSchema],
        TC,
        remove_prev_tool_calls,
        input: Any = None,
    ) -> None:
        if isinstance(node_schema, GraphSchema):
            self.current_graph_schema_idx += 1
            if input is None and edge_schema is not None:
                input = edge_schema.new_input_fn(self.state)

            graph = Graph(
                input=input,
                request=self.tasks[self.current_graph_schema_idx],
                graph_schema=node_schema,
            )
            self.curr_executable = Ref(graph, "curr_executable")
            self.curr_node = graph
            node_schema, edge_schema = graph.compute_init_node_edge_schema()
            input = None
            if edge_schema is not None:
                input = edge_schema.new_input_fn(self.curr_node.state)
            self.curr_node.init_next_node(
                node_schema, edge_schema, TC, remove_prev_tool_calls, input
            )
        elif node_schema in self.graph_schema.node_schemas:
            super().init_next_node(
                node_schema, edge_schema, TC, remove_prev_tool_calls, input
            )
        else:
            self.curr_node.init_next_node(
                node_schema, edge_schema, TC, remove_prev_tool_calls, input
            )


class RequestGraphSchema(HasGraphSchemaMixin, metaclass=AutoMixinInit):
    def __init__(
        self,
        node_prompt: str,
        node_system_prompt: Type[NodeSystemPrompt],
        description: str,
        edge_schemas: List[EdgeSchema],
        node_schemas: List[NodeSchema],
    ):
        self.start_node_schema = NodeSchema(node_prompt, node_system_prompt)


class GraphEdgeSchema(BaseEdgeSchema, HasIdMixin, metaclass=AutoMixinInit):
    pass
