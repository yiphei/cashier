from __future__ import annotations

import copy
import json
from abc import ABC, abstractmethod
from bisect import bisect_left
from collections import defaultdict
from enum import StrEnum
from typing import Any, ClassVar, Dict, List, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    constr,
    field_validator,
    model_validator,
)

from cashier.function_call_context import ToolExceptionWrapper
from cashier.model_util import CustomJSONEncoder, FunctionCall, ModelProvider
from cashier.tool_registry import ToolRegistry


class ModelTurn(BaseModel, ABC):
    should_strip_msg_content: ClassVar[bool] = True

    msg_content: constr(min_length=1)  # type: ignore

    @field_validator("msg_content")
    @classmethod
    def strip_whitespace(cls, value: str) -> str:
        return value.strip() if cls.should_strip_msg_content and value else value

    @abstractmethod
    def build_oai_messages(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def build_anthropic_messages(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def build_messages(self, model_provider: ModelProvider) -> List[Dict[str, Any]]:
        if model_provider == ModelProvider.OPENAI:
            return self.build_oai_messages()
        elif model_provider == ModelProvider.ANTHROPIC:
            return self.build_anthropic_messages()
        else:
            raise TypeError


class UserTurn(ModelTurn):
    def build_oai_messages(self) -> List[Dict[str, Any]]:
        return [{"role": "user", "content": self.msg_content}]

    def build_anthropic_messages(self) -> List[Dict[str, Any]]:
        return self.build_oai_messages()


class SystemTurn(ModelTurn):
    def build_oai_messages(self) -> List[Dict[str, Any]]:
        return [{"role": "system", "content": self.msg_content}]

    def build_anthropic_messages(self) -> None:
        return None


class NodeSystemTurn(SystemTurn):
    node_id: int


class AssistantTurn(ModelTurn):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_provider: ModelProvider
    tool_registry: Optional[ToolRegistry] = None
    msg_content: Optional[str]
    fn_calls: Optional[List[FunctionCall]] = Field(default_factory=list)
    fn_call_id_to_fn_output: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_function_args(self) -> AssistantTurn:
        if (
            self.fn_calls
            and self.fn_call_id_to_fn_output
            and len(self.fn_calls) != len(self.fn_call_id_to_fn_output.values())
        ):
            raise ValueError(
                "Mismatch between fn_calls' and fn_call_id_to_fn_output's lengths"
            )

        if self.fn_calls and self.tool_registry is None:
            raise ValueError()

        return self

    def build_oai_messages(self) -> List[Dict[str, Any]]:
        messages = []
        if self.msg_content and not (
            self.model_provider == ModelProvider.ANTHROPIC and self.fn_calls
        ):
            messages.append({"role": "assistant", "content": self.msg_content})
        if self.fn_calls:
            for fn_call in self.fn_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": fn_call.id,
                                "type": "function",
                                "function": {
                                    "arguments": fn_call.args_json,
                                    "name": fn_call.name,
                                },
                            }
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(
                            self.fn_call_id_to_fn_output[fn_call.id],
                            cls=CustomJSONEncoder,
                        ),
                        "tool_call_id": fn_call.id,
                    }
                )

                if (
                    fn_call.name in self.tool_registry.openai_tools_return_description
                    and not isinstance(
                        self.fn_call_id_to_fn_output[fn_call.id],
                        ToolExceptionWrapper,
                    )
                ):
                    json_schema = self.tool_registry.openai_tools_return_description[
                        fn_call.name
                    ]
                    system_msg = f"This is the JSON Schema of {fn_call.name}'s return type: {json.dumps(json_schema)}"

                    messages.append({"role": "system", "content": system_msg})

        return messages

    def build_anthropic_messages(self) -> List[Dict[str, Any]]:
        contents = []
        messages = []
        if self.msg_content and not (
            self.model_provider == ModelProvider.ANTHROPIC and self.fn_calls
        ):
            contents.append({"type": "text", "text": self.msg_content})
        if self.fn_calls:
            for fn_call in self.fn_calls:
                contents.append(
                    {
                        "type": "tool_use",
                        "id": fn_call.id,
                        "name": fn_call.name,
                        "input": fn_call.args,
                    }
                )

        if not self.fn_calls and self.msg_content:
            contents = contents[0]
            contents = contents["text"]

        messages.append({"role": "assistant", "content": contents})

        if self.fn_calls:
            return_contents = []
            for fn_call in self.fn_calls:
                return_contents.append(
                    {
                        "content": json.dumps(
                            self.fn_call_id_to_fn_output[fn_call.id],
                            cls=CustomJSONEncoder,
                        ),
                        "type": "tool_result",
                        "tool_use_id": fn_call.id,
                        "is_error": isinstance(
                            self.fn_call_id_to_fn_output[fn_call.id],
                            ToolExceptionWrapper,
                        ),
                    }
                )

            messages.append({"role": "user", "content": return_contents})

        return messages


class MessageManager(ABC):

    @property
    @abstractmethod
    def model_provider(self) -> ModelProvider:
        pass

    def __init__(self) -> None:
        self.message_dicts = MessageList(model_provider=self.model_provider)
        self.conversation_dicts = MessageList(model_provider=self.model_provider)
        self.node_conversation_dicts = MessageList(model_provider=self.model_provider)

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        if cls.model_provider is None:
            raise TypeError(f"{cls.__name__} must define 'model_provider'")

    def add_user_turn(self, turn: ModelTurn) -> None:
        user_msgs = turn.build_messages(self.model_provider)
        self.message_dicts.extend(user_msgs, MessageList.ItemType.USER)
        self.conversation_dicts.extend(user_msgs, MessageList.ItemType.USER)
        self.node_conversation_dicts.extend(user_msgs, MessageList.ItemType.USER)

    def add_node_turn(
        self,
        turn: ModelTurn,
        remove_prev_fn_return_schema: Optional[bool] = None,
        remove_prev_tool_calls: bool = False,
        is_skip: bool = False,
    ) -> None:
        if remove_prev_tool_calls:
            assert remove_prev_fn_return_schema is not False

        if remove_prev_fn_return_schema is True or remove_prev_tool_calls:
            self.message_dicts.clear(MessageList.ItemType.TOOL_OUTPUT_SCHEMA)

        if remove_prev_tool_calls:
            self.message_dicts.clear(
                [MessageList.ItemType.TOOL_CALL, MessageList.ItemType.TOOL_OUTPUT]
            )
        if is_skip:
            self.conversation_dicts.track_idx(
                MessageList.ItemType.NODE, len(self.conversation_dicts) - 2
            )
            self.node_conversation_dicts = self.node_conversation_dicts[-1:]
        else:
            self.conversation_dicts.track_idx(MessageList.ItemType.NODE)
            self.node_conversation_dicts.clear()

    @abstractmethod
    def parse_system_messages(self, msgs: List[Dict[str, Any]]):
        raise NotImplementedError

    @abstractmethod
    def parse_assistant_messages(self, msgs: List[Dict[str, Any]]):
        raise NotImplementedError

    def add_system_turn(self, turn: ModelTurn) -> None:
        self.parse_system_messages(turn.build_messages(self.model_provider))

    def add_assistant_turn(self, turn: AssistantTurn) -> None:
        # TODO: maybe move this logic to AssistantTurn
        if turn.msg_content and (
            turn.model_provider != ModelProvider.ANTHROPIC
            or (turn.model_provider == ModelProvider.ANTHROPIC and not turn.fn_calls)
        ):
            self.conversation_dicts.append(
                {"role": "assistant", "content": turn.msg_content},
                MessageList.ItemType.ASSISTANT,
            )
            self.node_conversation_dicts.append(
                {"role": "assistant", "content": turn.msg_content},
                MessageList.ItemType.ASSISTANT,
            )
        self.parse_assistant_messages(turn.build_messages(self.model_provider))


class OAIMessageManager(MessageManager):
    model_provider = ModelProvider.OPENAI

    def parse_system_messages(self, msgs: List[Dict[str, Any]]) -> None:
        self.message_dicts.extend(msgs)

    def add_node_turn(
        self,
        turn: ModelTurn,
        remove_prev_fn_return_schema: Optional[bool] = None,
        remove_prev_tool_calls: bool = False,
        is_skip: bool = False,
    ) -> None:
        super().add_node_turn(
            turn, remove_prev_fn_return_schema, remove_prev_tool_calls, is_skip
        )
        self.message_dicts.clear(MessageList.ItemType.NODE)
        [msg] = turn.build_oai_messages()
        if is_skip:
            self.message_dicts.insert(
                len(self.message_dicts) - 1, msg, MessageList.ItemType.NODE
            )
        else:
            self.message_dicts.append(msg, MessageList.ItemType.NODE)

    def parse_assistant_messages(self, msgs: List[Dict[str, Any]]) -> None:
        curr_fn_name = None
        for message in msgs:
            if message.get("tool_calls", None) is not None:
                tool_call_id = message["tool_calls"][0]["id"]
                curr_fn_name = message["tool_calls"][0]["function"]["name"]
                self.message_dicts.append(
                    message, MessageList.ItemType.TOOL_CALL, tool_call_id
                )
            elif message["role"] == "tool":
                tool_call_id = message["tool_call_id"]
                self.message_dicts.append(
                    message,
                    MessageList.ItemType.TOOL_OUTPUT,
                    MessageList.get_tool_output_uri_from_tool_id(tool_call_id),
                )
            elif message["role"] == "system" and curr_fn_name is not None:
                self.message_dicts.remove_by_uri(curr_fn_name, False)
                self.message_dicts.append(
                    message, MessageList.ItemType.TOOL_OUTPUT_SCHEMA, curr_fn_name
                )
                curr_fn_name = None
            else:
                self.message_dicts.append(message, MessageList.ItemType.ASSISTANT)


class AnthropicMessageManager(MessageManager):
    model_provider = ModelProvider.ANTHROPIC

    def __init__(self) -> None:
        super().__init__()
        self.system = None

    def parse_system_messages(self, msgs: List[Dict[str, Any]]) -> None:
        return None

    def add_node_turn(
        self,
        turn: ModelTurn,
        remove_prev_fn_return_schema: Optional[bool] = None,
        remove_prev_tool_calls: bool = False,
        is_skip: bool = False,
    ) -> None:
        super().add_node_turn(
            turn, remove_prev_fn_return_schema, remove_prev_tool_calls, is_skip
        )
        self.system = turn.msg_content

        if is_skip:
            self.message_dicts.track_idx(
                MessageList.ItemType.NODE, len(self.message_dicts) - 2
            )
        else:
            self.message_dicts.track_idx(MessageList.ItemType.NODE)

    def parse_assistant_messages(self, messages: List[Dict[str, Any]]) -> None:
        if len(messages) == 2:
            [message_1, message_2] = messages
        else:
            [message_1] = messages
            message_2 = None

        contents = message_1["content"]
        self.message_dicts.append(message_1)
        has_fn_calls = False
        if type(contents) == list:
            for content in contents:
                if content["type"] == "tool_use":
                    tool_call_id = content["id"]
                    self.message_dicts.track_idx(
                        MessageList.ItemType.TOOL_CALL, uri=tool_call_id
                    )
                    has_fn_calls = True

        if not has_fn_calls:
            self.message_dicts.track_idx(MessageList.ItemType.ASSISTANT)

        if message_2 is not None:
            self.message_dicts.append(message_2)
            for content in message_2["content"]:
                if content["type"] == "tool_result":
                    tool_id = content["tool_use_id"]
                    self.message_dicts.track_idx(
                        MessageList.ItemType.TOOL_OUTPUT,
                        uri=MessageList.get_tool_output_uri_from_tool_id(tool_id),
                    )


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
    ):
        self.model_provider_to_message_manager: Dict[ModelProvider, MessageManager] = {}
        for provider in model_providers:
            mm = self.model_provider_to_message_manager_cls[provider]()
            self.model_provider_to_message_manager[provider] = mm

        self.turns = []

    def add_system_turn(self, msg_content: str) -> None:
        turn = SystemTurn(msg_content=msg_content)
        self.turns.append(turn)
        for mm in self.model_provider_to_message_manager.values():
            mm.add_system_turn(turn)

    def add_node_turn(
        self,
        node: Any,  # TODO: fix this
        remove_prev_tool_fn_return: Optional[bool] = None,
        remove_prev_tool_calls: bool = False,
        is_skip: bool = False,
    ) -> None:
        turn = NodeSystemTurn(node_id=node.id, msg_content=node.prompt)
        self.turns.append(turn)
        for mm in self.model_provider_to_message_manager.values():
            mm.add_node_turn(
                turn, remove_prev_tool_fn_return, remove_prev_tool_calls, is_skip
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
        tool_registry: ToolRegistry,
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


class MessageList(list):
    class ItemType(StrEnum):
        USER = "USER"
        ASSISTANT = "ASSISTANT"
        TOOL_CALL = "TOOL_CALL"
        TOOL_OUTPUT = "TOOL_OUTPUT"
        TOOL_OUTPUT_SCHEMA = "TOOL_OUTPUT_SCHEMA"
        NODE = "NODE"

    item_type_to_uri_prefix = {
        ItemType.USER: "usr_",
        ItemType.TOOL_OUTPUT: "tout_",
        ItemType.NODE: "node_",
        ItemType.ASSISTANT: "asst_",
    }

    def __init__(self, *args: Any, model_provider: ModelProvider):
        super().__init__(*args)
        self.uri_to_list_idx: Dict[str, int] = {}
        self.list_idx_to_uris = defaultdict(set)
        self.list_idxs = []
        self.list_idx_to_track_idx = {}

        self.model_provider = model_provider
        self.item_type_to_uris = defaultdict(list)
        self.uri_to_item_type = {}
        self.item_type_to_count = {k: 0 for k in self.item_type_to_uri_prefix.keys()}

    def get_tool_id_from_tool_output_uri(self, uri: str) -> str:
        return uri[
            len(self.item_type_to_uri_prefix[MessageList.ItemType.TOOL_OUTPUT]) :
        ]

    @classmethod
    def get_tool_output_uri_from_tool_id(cls, tool_id: str) -> str:
        return cls.item_type_to_uri_prefix[MessageList.ItemType.TOOL_OUTPUT] + tool_id

    def pop_track_idx_ant(self, uri: str) -> None:
        track_idx = self.get_track_idx_from_uri(uri)
        item_type = self.uri_to_item_type[uri]
        message = self[track_idx]
        new_contents = []
        for content in message["content"]:
            if (
                item_type == MessageList.ItemType.TOOL_CALL
                and content["type"] == "tool_use"
                and content["id"] == uri
            ):
                continue
            elif (
                item_type == MessageList.ItemType.TOOL_OUTPUT
                and content["type"] == "tool_result"
                and content["tool_use_id"] == self.get_tool_id_from_tool_output_uri(uri)
            ):
                continue
            new_contents.append(content)

        if new_contents:
            if item_type == MessageList.ItemType.TOOL_CALL:
                if len(new_contents) == 1 and new_contents[0]["type"] == "text":
                    new_message = {
                        "role": "assistant",
                        "content": new_contents[0]["text"],
                    }
                else:
                    new_message = {"role": "assistant", "content": new_contents}
            elif item_type == MessageList.ItemType.TOOL_OUTPUT:
                new_message = {"role": "user", "content": new_contents}

            self[track_idx] = new_message
            self.pop_track_idx(uri, shift_idxs=False)
        else:
            self._remove_by_uri(uri, True)

    def track_idx(
        self,
        item_type: ItemType,
        list_idx: Optional[int] = None,
        uri: Optional[str] = None,
        is_insert: bool = False,
    ) -> None:
        if uri is None:
            self.item_type_to_count[item_type] += 1
            uri = self.item_type_to_uri_prefix[item_type] + str(
                self.item_type_to_count[item_type]
            )
        if list_idx is None:
            list_idx = len(self) - 1

        if uri in self.uri_to_list_idx:
            raise ValueError()

        self.uri_to_list_idx[uri] = list_idx
        self.item_type_to_uris[item_type].append(uri)
        self.uri_to_item_type[uri] = item_type
        if list_idx not in self.list_idxs or is_insert:
            if (self.list_idxs and self.list_idxs[-1] < list_idx) or not self.list_idxs:
                self.list_idxs.append(list_idx)
                self.list_idx_to_track_idx[list_idx] = len(self.list_idxs) - 1
            else:
                insert_idx = bisect_left(self.list_idxs, list_idx)

                self.list_idxs.insert(insert_idx, list_idx)
                self.shift_track_idxs(insert_idx + 1, 1)
                self.list_idx_to_track_idx[list_idx] = insert_idx

        self.list_idx_to_uris[list_idx].add(uri)

    def track_idxs(
        self,
        item_type: ItemType,
        start_list_idx: int,
        end_list_idx: Optional[int] = None,
        uris: Optional[List[str]] = None,
    ) -> None:
        if end_list_idx is None:
            end_list_idx = len(self) - 1
        if uris is None:
            range_idx = end_list_idx - start_list_idx + 1
            uris = [None] * range_idx

        for i, uri in zip(range(start_list_idx, end_list_idx + 1), uris):
            self.track_idx(item_type, i, uri)

    def get_track_idx_from_uri(self, uri: str) -> int:
        return self.uri_to_list_idx[uri]

    def get_track_idx_for_item_type(
        self, item_type: ItemType, idx: int = -1
    ) -> Optional[int]:
        order_validation = abs(idx) if idx < 0 else idx + 1
        target_uri = (
            self.item_type_to_uris[item_type][idx]
            if self.item_type_to_uris[item_type]
            and order_validation <= len(self.item_type_to_uris[item_type])
            else None
        )
        return self.uri_to_list_idx[target_uri] if target_uri else None

    def get_item_type_by_idx(self, item_type: ItemType, idx: int) -> Any:
        track_idx = self.get_track_idx_for_item_type(item_type, idx)
        if track_idx:
            return self[track_idx]
        else:
            return None

    def shift_track_idxs(self, start_track_idx: int, shift_direction: int) -> None:
        for i in range(start_track_idx, len(self.list_idxs)):
            curr_list_idx = self.list_idxs[i]
            self.list_idx_to_track_idx.pop(curr_list_idx)
            curr_uris = self.list_idx_to_uris[curr_list_idx]

            self.list_idxs[i] += shift_direction
            self.list_idx_to_track_idx[self.list_idxs[i]] = i

            for uri in curr_uris:
                self.uri_to_list_idx[uri] = self.list_idxs[i]
            self.list_idx_to_uris.pop(curr_list_idx)
            self.list_idx_to_uris[self.list_idxs[i]] = curr_uris

    def pop_track_idx(self, uri: str, shift_idxs: bool = True) -> Optional[int]:
        popped_list_idx = self.uri_to_list_idx.pop(uri)
        all_uris = self.list_idx_to_uris[popped_list_idx]

        item_type = self.uri_to_item_type.pop(uri)
        self.item_type_to_uris[item_type].remove(uri)

        all_uris.remove(uri)
        if not all_uris:
            popped_track_idx = self.list_idx_to_track_idx.pop(popped_list_idx)
            self.list_idx_to_uris.pop(popped_list_idx)
            del self.list_idxs[popped_track_idx]

            if shift_idxs:
                self.shift_track_idxs(popped_track_idx, -1)
            else:
                for i in range(popped_track_idx, len(self.list_idxs)):
                    curr_list_idx = self.list_idxs[i]
                    self.list_idx_to_track_idx.pop(curr_list_idx)
                    self.list_idx_to_track_idx[curr_list_idx] = i

            return popped_list_idx
        else:
            return None

    def append(
        self, item: Any, item_type: Optional[ItemType] = None, uri: Optional[str] = None
    ) -> None:
        super().append(item)
        if item_type is not None:
            self.track_idx(item_type, uri=uri)

    def insert(
        self,
        idx: int,
        item: Any,
        item_type: Optional[ItemType] = None,
        uri: Optional[str] = None,
    ) -> None:
        super().insert(idx, item)
        if item_type is not None:
            self.track_idx(item_type, idx, uri, is_insert=True)

    def extend(self, items: List[Any], item_type: Optional[ItemType] = None) -> None:
        curr_len = len(self) - 1
        super().extend(items)
        if items and item_type is not None:
            self.track_idxs(item_type, curr_len + 1)

    def _remove_by_uri(self, uri: str, raise_on_unpopped_idx: bool = False) -> None:
        popped_idx = self.pop_track_idx(uri)
        if popped_idx is not None:
            del self[popped_idx]
        else:
            if raise_on_unpopped_idx:
                raise ValueError

    def remove_by_uri(self, uri: str, raise_if_not_found: bool = True) -> None:
        if uri not in self.uri_to_item_type:
            if raise_if_not_found:
                raise ValueError()
            return

        item_type = self.uri_to_item_type[uri]
        if self.model_provider != ModelProvider.ANTHROPIC or (
            self.model_provider == ModelProvider.ANTHROPIC
            and not (
                item_type == MessageList.ItemType.TOOL_CALL
                or item_type == MessageList.ItemType.TOOL_OUTPUT
            )
        ):
            self._remove_by_uri(uri)
        else:
            self.pop_track_idx_ant(uri)

    def clear(
        self, item_type_or_types: Optional[Union[ItemType, List[ItemType]]] = None
    ) -> None:
        if item_type_or_types is None:
            super().clear()
        else:
            if not isinstance(item_type_or_types, list):
                item_type_or_types = [item_type_or_types]
            for item_type in item_type_or_types:
                uris = copy.copy(self.item_type_to_uris[item_type])
                for uri in uris:
                    self.remove_by_uri(uri)

    def __getitem__(self, index: Union[int, slice]) -> Any:
        if isinstance(index, slice):
            if index.step:
                raise ValueError()

            ml = MessageList(
                super().__getitem__(index), model_provider=self.model_provider
            )
            start_idx = index.start or 0
            if start_idx < 0:
                start_idx = len(self) + start_idx
            end_idx = index.stop or len(self)
            if end_idx < 0:
                end_idx = len(self) + end_idx
            ml.item_type_to_count = copy.deepcopy(self.item_type_to_count)
            for list_idx in self.list_idxs:
                if start_idx <= list_idx < end_idx:
                    for uri in self.list_idx_to_uris[list_idx]:
                        item_type = self.uri_to_item_type[uri]
                        ml.track_idx(item_type, list_idx - start_idx, uri)

            return ml
        else:
            return super().__getitem__(index)
