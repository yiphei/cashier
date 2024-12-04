from __future__ import annotations

import json
from typing import Any, Optional

from cashier.graph.edge_schema import (
    EdgeSchema,
    FunctionState,
    FunctionTransitionConfig,
    StateTransitionConfig,
)
from cashier.graph.new_classes import (
    AutoMixinInit,
    Direction,
    HasActionableMixin,
    HasActionableSchemaMixin,
    HasGraphMixin,
    HasGraphSchemaMixin,
)
from cashier.graph.state_model import BaseStateModel
from cashier.logger import logger
from cashier.model.model_util import CustomJSONEncoder
from cashier.prompts.graph_schema_addition import GraphSchemaAdditionPrompt
from cashier.prompts.graph_schema_selection import GraphSchemaSelectionPrompt


class RequestGraph(HasGraphMixin, HasActionableMixin):

    def __init__(
        self,
        schema: RequestGraphSchema,
        input: Any,
        state: BaseStateModel,
        prompt: str,
        in_edge_schema: Optional[EdgeSchema],
        direction: Direction = Direction.FWD,
    ):

        HasGraphMixin.__init__(self, schema)
        HasActionableMixin.__init__(
            self, schema, input, state, prompt, in_edge_schema, direction
        )

        self.tasks = []
        self.graph_schema_sequence = []
        self.current_graph_schema_idx = 0
        self.graph_schema_id_to_task = {}

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
                self.schema.graph_schema_id_to_graph_schema[agent_selection.agent_id]
            )
            self.tasks.append(agent_selection.task)
            self.graph_schema_id_to_task[agent_selection.agent_id] = (
                agent_selection.task
            )

        return True if agent_selection else False


class RequestGraphSchema(
    HasGraphSchemaMixin, HasActionableSchemaMixin, metaclass=AutoMixinInit
):
    instance_cls = RequestGraph


class GraphEdgeSchema:
    _counter = 0

    def __init__(
        self,
        from_graph_schema,
        to_graph_schema,
        transition_config,
    ):
        GraphEdgeSchema._counter += 1
        self.id = GraphEdgeSchema._counter

        self.from_graph_schema = from_graph_schema
        self.to_graph_schema = to_graph_schema
        self.transition_config = transition_config

    def check_transition_config(
        self, state: BaseStateModel, fn_call, is_fn_call_success
    ) -> bool:
        if isinstance(self.transition_config, FunctionTransitionConfig):
            if self.transition_config.state == FunctionState.CALLED:
                return fn_call.name == self.transition_config.fn_name
            elif self.transition_config.state == FunctionState.CALLED_AND_SUCCEEDED:
                return (
                    fn_call.name == self.transition_config.fn_name
                    and is_fn_call_success
                )
        elif isinstance(self.transition_config, StateTransitionConfig):
            return self.transition_config.state_check_fn(state)
