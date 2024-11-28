import json
from collections import defaultdict
from typing import List, Optional, Type, Union

from pydantic import BaseModel

from cashier.graph.edge_schema import (
    EdgeSchema,
    FunctionState,
    FunctionTransitionConfig,
    StateTransitionConfig,
)
from cashier.graph.state_model import BaseStateModel
from cashier.logger import logger
from cashier.model.model_turn import ModelTurn
from cashier.model.model_util import CustomJSONEncoder
from cashier.prompts.graph_schema_addition import GraphSchemaAdditionPrompt
from cashier.prompts.graph_schema_selection import GraphSchemaSelectionPrompt
from cashier.graph.new_classes import GraphMixin, GraphSchemaMixin, ActionableMixin, ActionableSchemaMixin
from cashier.prompts.node_system import NodeSystemPrompt
from cashier.tool.tool_registry import ToolRegistry
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam


class RequestGraphSchema(GraphSchemaMixin, ActionableSchemaMixin):
    def __init__(self,
                         description: str,
        edge_schemas: List[EdgeSchema],
        node_schemas: List[ActionableMixin],
                         node_prompt: str,
        node_system_prompt: Type[NodeSystemPrompt],
        input_pydantic_model: Optional[Type[BaseModel]] = None,
        state_pydantic_model: Optional[Type[BaseStateModel]] = None,
        tool_registry_or_tool_defs: Optional[
            Union[ToolRegistry, List[ChatCompletionToolParam]]
        ] = None,
        first_turn: Optional[ModelTurn] = None,
        run_assistant_turn_before_transition: bool = False,
        tool_names: Optional[List[str]] = None,
                 ):
            GraphSchemaMixin.__init__(self, 
                                      description,
                                      edge_schemas,
                                      node_schemas
                                      )
            ActionableSchemaMixin.__init__(self,
                                           node_prompt,
                                           node_system_prompt,
                                           input_pydantic_model,
                                           state_pydantic_model,
                                           tool_registry_or_tool_defs,
                                           first_turn,
                                           run_assistant_turn_before_transition,
                                           tool_names,
                                           )


class RequestGraph:

    def __init__(self, schema):
        self.schema = schema
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
