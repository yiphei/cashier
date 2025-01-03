from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from cashier.graph.conversation_node import ConversationNode

from cashier.model.message_list import MessageList
from cashier.model.model_turn import (
    AnthropicMessageManager,
    AssistantTurn,
    MessageManager,
    ModelTurn,
    NodeSystemTurn,
    OAIMessageManager,
    SystemTurn,
    UserTurn,
)
from cashier.model.model_util import FunctionCall, ModelProvider
from cashier.tool.tool_registry import ToolRegistry


class TurnContainer:
    model_provider_to_message_manager_cls = {
        ModelProvider.OPENAI: OAIMessageManager,
        ModelProvider.ANTHROPIC: AnthropicMessageManager,
    }

    def __init__(
        self,
        model_providers: List[ModelProvider] = [
            ModelProvider.OPENAI,
            ModelProvider.ANTHROPIC,
        ],
        remove_prev_tool_fn_return: Optional[bool] = None,
        remove_prev_tool_calls: bool = True,
    ):
        self.model_provider_to_message_manager: Dict[ModelProvider, MessageManager] = {}
        for provider in model_providers:
            mm = self.model_provider_to_message_manager_cls[provider]()
            self.model_provider_to_message_manager[provider] = mm

        self.turns: List[ModelTurn] = []
        self.remove_prev_tool_fn_return = remove_prev_tool_fn_return
        self.remove_prev_tool_calls = remove_prev_tool_calls

    def add_system_turn(self, msg_content: str) -> None:
        turn = SystemTurn(msg_content=msg_content)
        self.turns.append(turn)
        for mm in self.model_provider_to_message_manager.values():
            mm.add_system_turn(turn)

    def add_node_turn(
        self,
        node: ConversationNode,
        remove_prev_tool_fn_return: Optional[bool] = None,
        remove_prev_tool_calls: Optional[bool] = None,
        is_skip: bool = False,
    ) -> None:
        turn = NodeSystemTurn(node_id=node.id, msg_content=node.prompt)
        self.turns.append(turn)
        for mm in self.model_provider_to_message_manager.values():
            mm.add_node_turn(
                turn,
                remove_prev_tool_fn_return or self.remove_prev_tool_fn_return,
                remove_prev_tool_calls or self.remove_prev_tool_calls,
                is_skip,
            )

    def add_user_turn(self, msg_content: str) -> None:
        turn = UserTurn(msg_content=msg_content)
        self.turns.append(turn)
        for mm in self.model_provider_to_message_manager.values():
            mm.add_user_turn(turn)

    def add_assistant_turn(
        self,
        msg_content: Optional[str],
        model_provider: ModelProvider,
        tool_registry: Optional[ToolRegistry] = None,
        fn_calls: Optional[List[FunctionCall]] = None,
        fn_id_to_outputs: Optional[Dict[str, Any]] = None,
    ) -> None:
        turn = AssistantTurn(
            msg_content=msg_content,
            tool_registry=tool_registry,
            model_provider=model_provider,
            fn_calls=fn_calls,
            fn_call_id_to_fn_output=fn_id_to_outputs,
        )
        self.add_assistant_direct_turn(turn)

    def add_assistant_direct_turn(self, turn: AssistantTurn) -> None:
        self.turns.append(turn)
        for mm in self.model_provider_to_message_manager.values():
            mm.add_assistant_turn(turn)

    def get_message(
        self,
        item_type: MessageList.ItemType,
        idx: int = -1,
        model_provider: ModelProvider = ModelProvider.OPENAI,
        content_only: bool = False,
    ) -> Optional[str]:
        mm = self.model_provider_to_message_manager[model_provider]
        msg = mm.message_dicts.get_item_type_by_idx(item_type, idx)
        if content_only and msg:
            return msg["content"]
        else:
            return None

    def get_user_message(
        self,
        idx: int = -1,
        model_provider: ModelProvider = ModelProvider.OPENAI,
        content_only: bool = False,
    ) -> Optional[str]:
        return self.get_message(
            MessageList.ItemType.USER, idx, model_provider, content_only
        )

    def get_asst_message(
        self,
        idx: int = -1,
        model_provider: ModelProvider = ModelProvider.OPENAI,
        content_only: bool = False,
    ) -> Optional[str]:
        return self.get_message(
            MessageList.ItemType.ASSISTANT, idx, model_provider, content_only
        )
