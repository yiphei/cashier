import json
from typing import List

from pydantic import BaseModel

from cashier.graph_folder.node_schema import NodeSchema
from cashier.model.model_util import ModelProvider
from cashier.prompts.base_prompt import BasePrompt


class Response(BaseModel):
    agent_id: int


class NodeSchemaSelectionPrompt(BasePrompt):

    response_format = Response

    def dynamic_prompt(  # type: ignore
        self,
        all_node_schemas: List[NodeSchema],
        model_provider: ModelProvider,
        last_customer_msg: str,
    ) -> str:
        prompt = (
            "You are an AI-agent orchestration engine and your job is to select the best AI agent. "
            "Each AI agent is defined by 3 attributes: instructions, state, and tools (i.e. functions). "
            "The instructions <instructions> describe what the agent's conversation is supposed to be about and what they are expected to do. "
            "The state <state> keeps track of important data during the conversation. "
            "The tools <tools> represent explicit actions that the agent can perform.\n\n"
        )
        for node_schema in all_node_schemas:
            prompt += (
                f"<agent id={node_schema.id}>\n"
                "<instructions>\n"
                f"{node_schema.node_prompt}\n"
                "</instructions>\n\n"
                "<state>\n"
                f"{node_schema.state_pydantic_model.model_json_schema()}\n"
                "</state>\n\n"
                "<tools>\n"
                f"{json.dumps(node_schema.tool_registry.get_tool_defs(model_provider=model_provider))}\n"
                "</tools>\n"
                "</agent>\n\n"
            )

        prompt += (
            "All agents share the following background:\n"  # type: ignore
            "<background>\n"
            f"{all_node_schemas[0].node_system_prompt.BACKGROUND_PROMPT()}\n"
            "</background>\n\n"
            "Given a conversation with a customer and the list above of AI agents with their attributes, "
            "determine which AI agent can best continue the conversation, especially given last customer message, in accordance with the universal guidelines defined in <guidelines>. "
            "Respond by returning the AI agent ID.\n\n"
            "<guidelines>\n"
            "<state_guidelines>\n"
            "- Among the tools provided, there are functions for getting and updating the state defined in <state>. "
            "For state updates, the agent will have field specific update functions, whose names are `update_state_<field>` and where <field> is a state field.\n"
            "- The agent must update the state whenever applicable and as soon as possible. They cannot proceed to the next stage of the conversation without updating the state\n"
            "- Only the agent can update the state, so there is no need to udpate the state to the same value that had already been updated to in the past.\n"
            + "</state_guidelines>\n"
            "<tools_guidelines>\n"
            "- Minimize reliance on external knowledge. Always retrieve information from the system prompts and available tools. "
            "If they dont provide the information needed, the agent must say they do not know.\n"
            "- the agent must AVOID stating/mentioning that they can/will perform an action if there are no tools (including state updates) associated with that action.\n"
            "- if the agent needs to perform an action, they can only state to the customer that they performed it after the associated tool (including state update) calls have been successfull.\n"
            "</tools_guidelines>\n"
            "<general_guidelines>\n"
            "- the agent needs to think step-by-step before responding.\n"
            "- the agent must decline to do anything that is not explicitly covered by <instructions> and <guidelines>.\n"
            + "</general_guidelines>\n"
            "</guidelines>\n\n"
            "<last_customer_message>\n"
            f"{last_customer_msg}\n"
            "</last_customer_message>\n\n"
        )
        return prompt
