import copy
import json
from typing import Any, Dict, Set, Union

from pydantic import BaseModel, ConfigDict

from cashier.graph.conversation_node import ConversationNodeSchema
from cashier.graph.base.base_graph import BaseGraphSchema
from cashier.logger import logger
from cashier.model.model_completion import ModelOutput
from cashier.model.model_util import ModelProvider
from cashier.prompts.base_prompt import BasePrompt
from cashier.turn_container import TurnContainer


class Response(BaseModel):
    agent_id: int


class Input(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    current_node_schema: ConversationNodeSchema
    all_node_schemas: Set[Union[ConversationNodeSchema, BaseGraphSchema]]
    tc: TurnContainer
    is_wait: bool


class NodeSchemaSelectionPrompt(BasePrompt):
    run_input_kwargs = Input

    response_format = Response

    def dynamic_prompt(  # type: ignore
        self,
        background_prompt: str,
        all_node_schemas: Set[Union[ConversationNodeSchema, BaseGraphSchema]],
        model_provider: ModelProvider,
        last_customer_msg: str,
    ) -> str:
        prompt = (
            "You are an AI-agent orchestration engine and your job is to select the best AI agent. "
            "Each AI agent is defined by 3 attributes: instructions, state, and tools (i.e. functions). "
            "The instructions <instructions> describe what the agent's conversation is supposed to be about and what they are expected to do. "
            "The state <state> keeps track of important data during the conversation (some may not have a state). "
            "The tools <tools> represent explicit actions that the agent can perform.\n\n"
        )
        for node_schema in all_node_schemas:
            prompt += (
                f"<agent id={node_schema.id}>\n"
                "<instructions>\n"
                f"{node_schema.node_prompt}\n"
                "</instructions>\n\n"
                + (
                    "<state>\n"
                    f"{node_schema.state_schema.model_json_schema()}\n"
                    "</state>\n\n"
                    if node_schema.state_schema
                    else ""
                )
                + "<tools>\n"
                f"{json.dumps(node_schema.tool_registry.get_tool_defs(model_provider=model_provider))}\n"
                "</tools>\n"
                "</agent>\n\n"
            )

        prompt += (
            "All agents share the following background:\n"  # type: ignore
            "<background>\n"
            f"{background_prompt}\n"
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

    @classmethod
    def get_model_completion_args(
        cls, model_provider: ModelProvider, input: Any
    ) -> Dict[str, Any]:
        all_node_schemas = input.all_node_schemas
        tc = input.tc
        node_conv_msgs = copy.deepcopy(
            tc.model_provider_to_message_manager[model_provider].node_conversation_dicts
        )
        last_customer_msg = tc.get_user_message(content_only=True)

        prompt = NodeSchemaSelectionPrompt(
            background_prompt=input.current_node_schema.node_system_prompt.BACKGROUND_PROMPT(),
            all_node_schemas=all_node_schemas,
            model_provider=model_provider,
            last_customer_msg=last_customer_msg,
        )

        if model_provider == ModelProvider.ANTHROPIC:
            node_conv_msgs.append({"role": "user", "content": prompt})
        elif model_provider == ModelProvider.OPENAI:
            node_conv_msgs.append({"role": "system", "content": prompt})

        return {"message_dicts": node_conv_msgs, "logprobs": True, "temperature": 0}

    @classmethod
    def get_output(
        cls, model_provider: ModelProvider, chat_completion: ModelOutput, input: Input
    ) -> bool:
        current_node_schema = input.current_node_schema
        is_wait = input.is_wait
        agent_id = chat_completion.get_message_prop("agent_id")
        actual_agent_id = agent_id if agent_id != current_node_schema.id else None
        if model_provider == ModelProvider.OPENAI:
            prob = chat_completion.get_prob(-2)  # type: ignore
            logger.debug(
                f"{'SKIP_AGENT_ID' if not is_wait else 'WAIT_AGENT_ID'}: {actual_agent_id or 'current_id'} with {prob}"
            )
        else:
            logger.debug(
                f"{'SKIP_AGENT_ID' if not is_wait else 'WAIT_AGENT_ID'}: {actual_agent_id or 'current_id'}"
            )
        return actual_agent_id
