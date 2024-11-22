from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict

from cashier.graph.graph_schema import GraphSchema
from cashier.model.model_completion import ModelOutput
from cashier.model.model_util import ModelProvider
from cashier.prompt_action.prompt_action_base import PromptActionBase
from cashier.prompts.graph_schema_selection import GraphSchemaSelectionPrompt


class Input(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph_schemas: List[GraphSchema]
    request: str


class GraphSelectionAction(PromptActionBase[Input]):
    prompt = GraphSchemaSelectionPrompt
    input_kwargs = Input

    @classmethod
    def get_model_completion_args(
        cls, model_provider: ModelProvider, input: Input
    ) -> Dict[str, Any]:
        prompt = cls.prompt(
            graph_schemas=input.graph_schemas,
            request=input.request,
        )

        msgs = [{"role": "user", "content": prompt}]

        return {"message_dicts": msgs, "logprobs": True, "temperature": 0}

    @classmethod
    def get_output(
        cls, model_provider: ModelProvider, chat_completion: ModelOutput, input: Input
    ) -> bool:
        return chat_completion.get_message_prop("agent_selections")
