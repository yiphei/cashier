from typing import List

from pydantic import BaseModel

from cashier.graph.graph_schema import GraphSchema
from cashier.prompts.base_prompt import BasePrompt


class Response(BaseModel):
    agent_ids: List[int]


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
            "Respond by returning the AI agent IDs in the correct logical order. You must return at least one agent ID and each agent ID must be unique. If no combination of agents can address the request, return an empty list.\n\n"
            "<customer_request>\n"
            f"{request}\n"
            "</customer_request>\n\n"
        )
        return prompt
