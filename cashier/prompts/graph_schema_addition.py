import copy
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from cashier.graph.graph_schema import GraphSchema
from cashier.model.model_completion import ModelOutput
from cashier.model.model_util import ModelProvider
from cashier.prompts.base_prompt import BasePrompt
from cashier.turn_container import TurnContainer


class AgentSelection(BaseModel):
    agent_id: int
    task: str = Field(
        description="The appropriate task description for this agent to perform"
    )


class Response(BaseModel):
    agent_selection: Optional[AgentSelection]


class RunInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph_schemas: List[GraphSchema]
    request: str
    curr_agent_id: int
    curr_task: str
    tc: TurnContainer

class GraphSchemaAdditionPrompt(BasePrompt):

    response_format = Response

    run_input_kwargs = RunInput
    def dynamic_prompt(  # type: ignore
        self,
        graph_schemas: List[GraphSchema],
        request: str,
        curr_agent_id: int,
        curr_task: str,
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
            f"Right now, AI agent with ID {curr_agent_id} is helping the customer with this request: {curr_task}. "
            "Given a conversation with a customer and the list above of AI agents with their attributes, "
            "determine if 1) the last customer message contains an entirely new request that requires an AI agent different from the current one. "
            "If so, respond by returning the AI agent ID along with the transcription of the customer request into an approtiate task description. The task description must be a paraphrase of the customer request (e.g. 'the customer wants ...'). If not, return an empty list.\n\n"
            "<last_customer_message>\n"
            f"{request}\n"
            "</last_customer_message>\n\n"
        )
        return prompt

    @classmethod
    def get_model_completion_args(
        cls, model_provider: ModelProvider, input: Any
    ) -> Dict[str, Any]:
        tc = input.tc
        node_conv_msgs = copy.deepcopy(
            tc.model_provider_to_message_manager[model_provider].node_conversation_dicts
        )
        last_customer_msg = tc.get_user_message(content_only=True)

        prompt = cls(
            graph_schemas=input.graph_schemas,
            request=input.request,
            curr_agent_id=input.curr_agent_id,
            curr_task=input.curr_task,
            last_customer_msg = last_customer_msg,
        )

        if model_provider == ModelProvider.ANTHROPIC:
            node_conv_msgs.append({"role": "user", "content": prompt})
        elif model_provider == ModelProvider.OPENAI:
            node_conv_msgs.append({"role": "system", "content": prompt})

        return {"message_dicts": node_conv_msgs, "logprobs": True, "temperature": 0}


    @classmethod
    def get_output(
        cls, model_provider: ModelProvider, chat_completion: ModelOutput, input: Any
    ) -> bool:
        return chat_completion.get_message_prop("agent_selection")
