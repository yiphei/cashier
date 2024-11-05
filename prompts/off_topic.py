from prompts.base_prompt import Prompt
from graph import BACKGROUND
import json
from model_tool_decorator import ToolRegistry

class OffTopicPrompt(Prompt):

    def dynamic_prompt(self, all_node_schemas, model_provider, last_customer_msg):
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
                f"{json.dumps(ToolRegistry.get_tool_defs_from_names(node_schema.tool_fn_names, model_provider, node_schema.tool_registry))}\n"
                "</tools>\n"
                "</agent>\n\n"
            )

        prompt += (
            "All agents share the following background:\n"
            "<background>\n"
            f"{BACKGROUND}\n"
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
