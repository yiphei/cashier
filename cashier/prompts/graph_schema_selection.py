from typing import Any, List

from pydantic import BaseModel, Field

from cashier.graph.graph_schema import GraphSchema
from cashier.model.model_completion import ModelOutput
from cashier.model.model_util import ModelProvider
from cashier.prompts.base_prompt import BasePrompt


class AgentSelection(BaseModel):
    agent_id: int
    task: str = Field(
        description="The appropriate task description for this agent to perform"
    )


class Response(BaseModel):
    agent_selections: List[AgentSelection]


class GraphSchemaSelectionPrompt(BasePrompt):

    response_format = Response

    def dynamic_prompt(  # type: ignore
        self,
        graph_schemas: List[GraphSchema],
        request: str,
    ) -> str:
        prompt = (
            "You are an AI-agent orchestration engine and your job is to select the best AI agents for a customer request. "
            "Each AI agent is defined by 2 attributes: description and output schema. "
            "The description <description> is a verbal description of what the agent's conversation is supposed to be about and what they are expected to do. "
            "The state <output_schema> represents the JSON schema of the agent's final output.\n\n"
        )
        for graph_schema in graph_schemas:
            prompt += (
                f"<agent id={graph_schema.id}>\n"
                "<description>\n"
                f"{graph_schema.description}\n"
                "</description>\n\n"
                "<output_schema>\n"
                f"{graph_schema.output_schema.model_json_schema()}\n"
                "</output_schema>\n\n"
                "</agent>\n\n"
            )

        prompt += (
            "Given a customer request and the above list of AI agents with their attributes, "
            "determine which AI agents can best address the request. "
            "Respond by returning the AI agent IDs in the correct logical order along with the transcription of the customer request into an approtiate task description, for each agent ID. The task description must be a paraphrase of the customer request (e.g. 'the customer wants ...'). You must return at least one agent ID and each agent ID must be unique. If no combination of agents can address the request, return an empty list.\n\n"
            "<customer_request>\n"
            f"{request}\n"
            "</customer_request>\n\n"
        )
        return prompt


    @classmethod
    def get_output(
        cls, model_provider: ModelProvider, chat_completion: ModelOutput, input: Any
    ) -> bool:
        return chat_completion.get_message_prop("agent_selections")