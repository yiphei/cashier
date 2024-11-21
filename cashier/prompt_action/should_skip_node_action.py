import copy
import json
from typing import Any, Dict, Set

from pydantic import BaseModel, ConfigDict

from cashier.graph.node_schema import NodeSchema
from cashier.logger import logger
from cashier.model.model_completion import ModelOutput
from cashier.model.model_util import ModelProvider
from cashier.prompt_action.prompt_action_base import PromptActionBase
from cashier.prompts.node_schema_selection import NodeSchemaSelectionPrompt
from cashier.turn_container import TurnContainer


class Input(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    current_node_schema: NodeSchema
    all_node_schemas: Set[NodeSchema]
    tc: TurnContainer
    is_wait: bool


class SholdSkipNodeAction(PromptActionBase):
    prompt = NodeSchemaSelectionPrompt
    input_kwargs = Input

    @classmethod
    def get_model_completion_args(
        cls, model_provider: ModelProvider, input: Any
    ) -> Dict[str, Any]:
        all_node_schemas = input.all_node_schemas
        TM = input.tc
        node_conv_msgs = copy.deepcopy(
            TM.model_provider_to_message_manager[model_provider].node_conversation_dicts
        )
        last_customer_msg = TM.get_user_message(content_only=True)

        prompt = NodeSchemaSelectionPrompt(
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
        cls, model_provider: ModelProvider, chat_completion: ModelOutput, input: Any
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
        return agent_id