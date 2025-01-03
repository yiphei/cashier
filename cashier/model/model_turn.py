from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    constr,
    field_validator,
    model_validator,
)

from cashier.model.message_list import MessageList
from cashier.model.model_util import CustomJSONEncoder, FunctionCall, ModelProvider
from cashier.tool.function_call_context import ToolExceptionWrapper
from cashier.tool.tool_registry import ToolRegistry


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

    def build_anthropic_messages(self) -> List[Dict[str, Any]]:
        return [{"role": "system", "content": self.msg_content}]  # TODO: fix this


class NodeSystemTurn(SystemTurn):
    node_id: int


class AssistantTurn(ModelTurn):
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    model_provider: ModelProvider
    tool_registry: Optional[ToolRegistry] = Field(default=None, exclude=True)
    msg_content: Optional[str]
    fn_calls: Optional[List[FunctionCall]] = Field(default_factory=list)
    fn_call_id_to_fn_output: Optional[Dict[str, Any]] = Field(
        default_factory=dict, exclude=True
    )

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
        messages: List[Dict[str, Any]] = []
        if self.msg_content and not (
            self.model_provider == ModelProvider.ANTHROPIC and self.fn_calls
        ):
            messages.append({"role": "assistant", "content": self.msg_content})
        if self.fn_calls:
            for fn_call in self.fn_calls:
                assert self.fn_call_id_to_fn_output is not None
                assert self.tool_registry is not None
                api_id = fn_call.oai_api_id
                messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": api_id,
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
                        "tool_call_id": api_id,
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
        contents: List[Dict[str, Any]] = []
        assistant_msg = None
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
                        "id": fn_call.anthropic_api_id,
                        "name": fn_call.name,
                        "input": fn_call.args,
                    }
                )

        if not self.fn_calls and self.msg_content:
            assistant_msg = contents[0]["text"]

        messages.append({"role": "assistant", "content": assistant_msg or contents})

        if self.fn_calls:
            return_contents = []
            for fn_call in self.fn_calls:
                assert self.fn_call_id_to_fn_output is not None
                return_contents.append(
                    {
                        "content": json.dumps(
                            self.fn_call_id_to_fn_output[fn_call.id],
                            cls=CustomJSONEncoder,
                        ),
                        "type": "tool_result",
                        "tool_use_id": fn_call.anthropic_api_id,
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
        raise NotImplementedError

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
    def parse_system_messages(self, msgs: List[Dict[str, Any]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def parse_assistant_messages(self, msgs: List[Dict[str, Any]]) -> None:
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
        [msg] = msgs
        self.message_dicts.append(msg, MessageList.ItemType.NODE)  # TODO: refactor this

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
        [msg] = msgs
        self.system = msg["content"]
        self.message_dicts.track_idx(MessageList.ItemType.NODE)  # TODO: refactor this

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
        if type(contents) is list:
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
