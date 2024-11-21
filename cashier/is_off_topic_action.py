import copy
import json
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict

from cashier.graph.node_schema import NodeSchema
from cashier.logger import logger
from cashier.model.model_completion import ModelOutput
from cashier.model.model_util import ModelProvider
from cashier.prompt_action_base import PromptActionBase
from cashier.prompts.off_topic import OffTopicPrompt
from cashier.turn_container import TurnContainer


class Input(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    current_node_schema: NodeSchema
    tc: TurnContainer


class IsOffTopicAction(PromptActionBase):
    prompt = OffTopicPrompt
    input_kwargs = Input

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

        prompt = cls.prompt(
            background_prompt=current_node_schema.node_system_prompt.BACKGROUND_PROMPT(),  # type: ignore
            node_prompt=current_node_schema.node_prompt,
            state_json_schema=current_node_schema.state_pydantic_model.model_json_schema(),
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
