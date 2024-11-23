from typing import Any, Dict
from pydantic import BaseModel, ConfigDict

from cashier.graph.node_schema import NodeSchema
from cashier.model.model_completion import ModelOutput
from cashier.model.model_util import ModelProvider
from cashier.prompts.base_prompt import BasePrompt
from cashier.turn_container import TurnContainer
import copy
import json
from cashier.logger import logger


class Response(BaseModel):
    output: bool


class RunInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    current_node_schema: NodeSchema
    tc: TurnContainer

class OffTopicPrompt(BasePrompt):
    run_input_kwargs = RunInput

    f_string_prompt = (
        "You are an AI-agent orchestration engine and your job is to evaluate the current AI agent's performance. "
        "The AI agent's background is:\n"
        "<background>\n"
        "{background_prompt}\n"
        "</background>\n\n"
        "The AI agent is defined by 3 attributes: instructions, state, and tools (i.e. functions).\n\n"
        "The instructions describe what the agent's conversation is supposed to be about and what they are expected to do.\n"
        "<instructions>\n"
        "{node_prompt}\n"
        "</instructions>\n\n"
        "The state keeps track of important data during the conversation.\n"
        "<state>\n"
        "{state_json_schema}\n"
        "</state>\n\n"
        "The tools represent explicit actions that the agent can perform.\n"
        "<tools>\n"
        "{tool_defs}\n"
        "</tools>\n\n"
        "Given a conversation between a customer and the current AI agent, determine if the"
        " conversation, especially given the last customer message, can continue to be fully handled by the current AI agent's <instructions>, <state>, or <tools> according to the guidelines defined in <guidelines>. Return true only if"
        " 100% certain, and return false if otherwise, meaning that we should at least explore letting another AI agent take over.\n\n"
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
        "{last_customer_msg}\n"
        "</last_customer_message>\n\n"
    )
    response_format = Response

    @classmethod
    def get_model_completion_args(
        cls, model_provider: ModelProvider, input: Any
    ) -> Dict[str, Any]:
        current_node_schema = input.current_node_schema
        tc = input.tc
        node_conv_msgs = copy.deepcopy(
            tc.model_provider_to_message_manager[model_provider].node_conversation_dicts
        )
        last_customer_msg = tc.get_user_message(content_only=True)

        prompt = cls(
            background_prompt=current_node_schema.node_system_prompt.BACKGROUND_PROMPT(),  # type: ignore
            node_prompt=current_node_schema.node_prompt,
            state_json_schema=str(
                current_node_schema.state_pydantic_model.model_json_schema()
            ),
            tool_defs=json.dumps(
                current_node_schema.tool_registry.get_tool_defs(
                    model_provider=model_provider
                )
            ),
            last_customer_msg=last_customer_msg,
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
        is_on_topic = chat_completion.get_message_prop("output")
        if model_provider == ModelProvider.OPENAI:
            prob = chat_completion.get_prob(-2)  # type: ignore
            logger.debug(f"IS_ON_TOPIC: {is_on_topic} with {prob}")
        else:
            logger.debug(f"IS_ON_TOPIC: {is_on_topic}")

        return is_on_topic