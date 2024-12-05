from __future__ import annotations

import json
from typing import Any, List, Type

from cashier.graph.edge_schema import EdgeSchema
from cashier.graph.mixin.auto_mixin_init import AutoMixinInit
from cashier.graph.mixin.base_edge_schema import BaseEdgeSchema
from cashier.graph.mixin.graph_mixin import HasGraphMixin, HasGraphSchemaMixin
from cashier.graph.mixin.has_id_mixin import HasIdMixin
from cashier.graph.node_schema import NodeSchema
from cashier.logger import logger
from cashier.model.model_util import CustomJSONEncoder
from cashier.prompts.graph_schema_addition import GraphSchemaAdditionPrompt
from cashier.prompts.graph_schema_selection import GraphSchemaSelectionPrompt
from cashier.prompts.node_system import NodeSystemPrompt


class RequestGraph(HasGraphMixin):

    def __init__(
        self,
        input: Any,
        graph_schema: HasGraphSchemaMixin,
    ):
        HasGraphMixin.__init__(self, graph_schema)
        self.tasks = []
        self.graph_schema_sequence = []
        self.current_graph_schema_idx = 0
        self.graph_schema_id_to_task = {}

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
