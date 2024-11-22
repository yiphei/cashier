from cashier.model.model_util import CustomJSONEncoder
from cashier.prompt_action.graph_addition_action import GraphAdditionAction
from cashier.prompt_action.graph_selection_action import GraphSelectionAction
from cashier.logger import logger
import json

class RequestGraphSchema:
    def __init__(self, graph_schemas, system_prompt):
        self.graph_schemas = graph_schemas
        self.graph_schema_id_to_graph_schema = {
            graph_schema.id: graph_schema for graph_schema in self.graph_schemas
        }
        self.system_prompt = system_prompt


class RequestGraph:

    def __init__(self, schema):
        self.schema = schema
        self.tasks = []
        self.graph_schema_sequence = []
        self.current_graph_schema_idx = 0
        self.graph_schema_id_to_task = {}

    def get_graph_schemas(self, request):
        agent_selections = GraphSelectionAction.run(
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

    def add_tasks(self, request):
        agent_selections = GraphAdditionAction.run(
            "claude-3.5",
            graph_schemas=self.schema.graph_schemas,
            request=request,
            curr_agent_id=self.graph_schema_sequence[self.current_graph_schema_idx].id,
            curr_task=self.tasks[self.current_graph_schema_idx],
        )

        logger.debug(f"agent_selections: {json.dumps(agent_selections, cls=CustomJSONEncoder, indent=4)}")

        for agent_selection in agent_selections:
            self.graph_schema_sequence.append(
                self.schema.graph_schema_id_to_graph_schema[agent_selection.agent_id]
            )
            self.tasks.append(agent_selection.task)
            self.graph_schema_id_to_task[agent_selection.agent_id] = (
                agent_selection.task
            )

        return True if agent_selections else False
