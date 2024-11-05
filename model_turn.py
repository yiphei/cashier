from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json

from pydantic import BaseModel, Field, constr, model_validator

from function_call_context import ToolExceptionWrapper
from model import MessageList
from model_tool_decorator import ToolRegistry
from model_util import CustomJSONEncoder, FunctionCall, ModelProvider


class ModelTurn(BaseModel, ABC):
    msg_content: constr(min_length=1)  # type: ignore

    @abstractmethod
    def build_oai_messages(self):
        raise NotImplementedError

    @abstractmethod
    def build_anthropic_messages(self):
        raise NotImplementedError

    def build_messages(self, model_provider):
        if model_provider == ModelProvider.OPENAI:
            return self.build_oai_messages()
        elif model_provider == ModelProvider.ANTHROPIC:
            return self.build_anthropic_messages()


class UserTurn(ModelTurn):
    def build_oai_messages(self):
        return [{"role": "user", "content": self.msg_content}]

    def build_anthropic_messages(self):
        return self.build_oai_messages()


class SystemTurn(ModelTurn):
    def build_oai_messages(self):
        return [{"role": "system", "content": self.msg_content}]

    def build_anthropic_messages(self):
        return None


class NodeSystemTurn(SystemTurn):
    node_id: int


class AssistantTurn(ModelTurn):
    model_provider: ModelProvider
    msg_content: Optional[str]
    fn_calls: Optional[List[FunctionCall]] = Field(default_factory=list)
    fn_call_id_to_fn_output: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_function_args(self):
        if (
            self.fn_calls
            and self.fn_call_id_to_fn_output
            and len(self.fn_calls) != len(self.fn_call_id_to_fn_output.values())
        ):
            raise ValueError(
                "Mismatch between fn_calls' and fn_call_id_to_fn_output's lengths"
            )
        return self

    def build_oai_messages(self):
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
                                "id": fn_call.tool_call_id,
                                "type": "function",
                                "function": {
                                    "arguments": fn_call.function_args_json,
                                    "name": fn_call.function_name,
                                },
                            }
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(
                            self.fn_call_id_to_fn_output[fn_call.tool_call_id],
                            cls=CustomJSONEncoder,
                        ),
                        "tool_call_id": fn_call.tool_call_id,
                    }
                )

                if (
                    fn_call.function_name
                    in ToolRegistry.GLOBAL_OPENAI_TOOLS_RETUN_DESCRIPTION
                    and not isinstance(
                        self.fn_call_id_to_fn_output[fn_call.tool_call_id],
                        ToolExceptionWrapper,
                    )
                ):
                    json_schema = ToolRegistry.GLOBAL_OPENAI_TOOLS_RETUN_DESCRIPTION[
                        fn_call.function_name
                    ]
                    system_msg = f"This is the JSON Schema of {fn_call.function_name}'s return type: {json.dumps(json_schema)}"

                    messages.append({"role": "system", "content": system_msg})

        return messages

    def build_anthropic_messages(self):
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
                        "id": fn_call.tool_call_id,
                        "name": fn_call.function_name,
                        "input": fn_call.function_args,
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
                            self.fn_call_id_to_fn_output[fn_call.tool_call_id],
                            cls=CustomJSONEncoder,
                        ),
                        "type": "tool_result",
                        "tool_use_id": fn_call.tool_call_id,
                        "is_error": isinstance(
                            self.fn_call_id_to_fn_output[fn_call.tool_call_id],
                            ToolExceptionWrapper,
                        ),
                    }
                )

            messages.append({"role": "user", "content": return_contents})

        return messages


class MessageManager(ABC):
    model_provider = None

    def __init__(self):
        self.message_dicts = MessageList(model_provider=self.model_provider)
        self.conversation_dicts = MessageList(model_provider=self.model_provider)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.model_provider is None:
            raise TypeError(f"{cls.__name__} must define 'model_provider'")

    def add_user_turn(self, turn):
        user_msgs = turn.build_messages(self.model_provider)
        self.message_dicts.extend(user_msgs, MessageList.ItemType.USER)
        self.conversation_dicts.extend(user_msgs, MessageList.ItemType.USER)

    def add_node_turn(
        self,
        turn,
        remove_prev_fn_return_schema=None,
        remove_prev_tool_calls=False,
        is_skip=False,
    ):
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
        else:
            self.conversation_dicts.track_idx(MessageList.ItemType.NODE)

    @abstractmethod
    def parse_system_messages(self, msgs):
        raise NotImplementedError

    @abstractmethod
    def parse_assistant_messages(self, msgs):
        raise NotImplementedError

    def add_system_turn(self, turn):
        self.parse_system_messages(turn.build_messages(self.model_provider))

    def add_assistant_turn(self, turn):
        # TODO: maybe move this logic to AssistantTurn
        if turn.msg_content and (
            turn.model_provider != ModelProvider.ANTHROPIC
            or (turn.model_provider == ModelProvider.ANTHROPIC and not turn.fn_calls)
        ):
            self.conversation_dicts.append(
                {"role": "assistant", "content": turn.msg_content},
                MessageList.ItemType.ASSISTANT,
            )
        self.parse_assistant_messages(turn.build_messages(self.model_provider))

    def get_user_message(self, order=-1):
        idx = self.message_dicts.get_track_idx_for_item_type(
            MessageList.ItemType.USER, order
        )
        if idx:
            return self.message_dicts[idx]
        else:
            return None

    def get_asst_message(self, order=-1):
        idx = self.message_dicts.get_track_idx_for_item_type(
            MessageList.ItemType.ASSISTANT, order
        )
        if idx:
            return self.message_dicts[idx]
        else:
            return None

    def get_conversation_msgs_since_last_node(self):
        idx = self.conversation_dicts.get_track_idx_for_item_type(
            MessageList.ItemType.NODE
        )
        return self.conversation_dicts[idx + 1 :]


class OAIMessageManager(MessageManager):
    model_provider = ModelProvider.OPENAI

    def parse_system_messages(self, msgs):
        self.message_dicts.extend(msgs)

    def add_node_turn(
        self,
        turn,
        remove_prev_fn_return_schema=None,
        remove_prev_tool_calls=False,
        is_skip=False,
    ):
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

    def parse_assistant_messages(self, msgs):
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

    def __init__(self):
        super().__init__()
        self.system = None

    def parse_system_messages(self, msgs):
        return

    def add_node_turn(
        self,
        turn,
        remove_prev_fn_return_schema=None,
        remove_prev_tool_calls=False,
        is_skip=False,
    ):
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

    def parse_assistant_messages(self, messages):
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

    def __init__(self, model_providers=[ModelProvider.OPENAI, ModelProvider.ANTHROPIC]):
        self.model_provider_to_message_manager = {}
        for provider in model_providers:
            mm = self.model_provider_to_message_manager_cls[provider]()
            self.model_provider_to_message_manager[provider] = mm

        self.turns = []

    def add_system_turn(self, msg_content):
        turn = SystemTurn(msg_content=msg_content)
        self.turns.append(turn)
        for mm in self.model_provider_to_message_manager.values():
            mm.add_system_turn(turn)

    def add_node_turn(
        self,
        node,
        remove_prev_tool_fn_return=None,
        remove_prev_tool_calls=False,
        is_skip=False,
    ):
        turn = NodeSystemTurn(node_id=node.id, msg_content=node.prompt)
        self.turns.append(turn)
        for mm in self.model_provider_to_message_manager.values():
            mm.add_node_turn(
                turn, remove_prev_tool_fn_return, remove_prev_tool_calls, is_skip
            )

    def add_user_turn(self, msg_content):
        turn = UserTurn(msg_content=msg_content)
        self.turns.append(turn)
        for mm in self.model_provider_to_message_manager.values():
            mm.add_user_turn(turn)

    def add_assistant_turn(
        self,
        msg_content,
        model_provider,
        fn_calls=None,
        fn_id_to_outputs=None,
    ):
        turn = AssistantTurn(
            msg_content=msg_content,
            model_provider=model_provider,
            fn_calls=fn_calls,
            fn_call_id_to_fn_output=fn_id_to_outputs,
        )
        self.add_assistant_direct_turn(turn)

    def add_assistant_direct_turn(self, turn):
        self.turns.append(turn)
        for mm in self.model_provider_to_message_manager.values():
            mm.add_assistant_turn(turn)
