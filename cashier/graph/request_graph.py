import json

from cashier.graph.edge_schema import FunctionState, FunctionTransitionConfig, StateTransitionConfig
from cashier.graph.state_model import BaseStateModel
from cashier.logger import logger
from cashier.model.model_util import CustomJSONEncoder
from cashier.prompts.graph_schema_addition import GraphSchemaAdditionPrompt
from cashier.prompts.graph_schema_selection import GraphSchemaSelectionPrompt

from collections import defaultdict


class RequestGraphSchema:
    def __init__(self, graph_schemas, graph_edge_schemas, system_prompt):
        self.graph_schemas = graph_schemas
        self.graph_edge_schemas = graph_edge_schemas
        self.graph_schema_id_to_graph_schema = {
            graph_schema.id: graph_schema for graph_schema in self.graph_schemas
        }
        self.system_prompt = system_prompt
        self.from_graph_schema_id_to_graph_edge_schemas = defaultdict(list)
        for graph_edge_schema in graph_edge_schemas:
            self.from_graph_schema_id_to_graph_edge_schemas[graph_edge_schema.from_graph_schema.id].append(graph_edge_schema)


class RequestGraph:

    def __init__(self, schema):
        self.schema = schema
        self.tasks = []
        self.graph_schema_sequence = []
        self.current_graph_schema_idx = 0
        self.graph_schema_id_to_task = {}

    def get_graph_schemas(self, request):
        agent_selections = GraphSchemaSelectionPrompt.run(
            "claude-3.5", request=request, graph_schemas=self.schema.graph_schemas
        )
        for agent_selection in agent_selections:
            self.graph_schema_sequence.append(
                self.schema.graph_schema_id_to_graph_schema[agent_selection.agent_id]
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
            graph_schemas=self.schema.graph_schemas,
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