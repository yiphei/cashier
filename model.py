from __future__ import annotations

import itertools
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Union, overload

import anthropic
import numpy as np
from openai import OpenAI
from pydantic import BaseModel, Field, constr, model_validator

from function_call_context import ToolExceptionWrapper
from model_tool_decorator import ToolRegistry
from model_util import ModelProvider

OpenAIModels = Literal["gpt-4o-mini", "gpt-4o"]

AnthropicModels = Literal["claude-3.5", "claude-3-5-sonnet-latest"]


class Model:
    model_name_to_provider = {
        "gpt-4o-mini": ModelProvider.OPENAI,
        "gpt-4o": ModelProvider.OPENAI,
        "claude-3-5-sonnet-20240620": ModelProvider.ANTHROPIC,
    }
    alias_to_model_name = {"claude-3.5": "claude-3-5-sonnet-20240620"}

    def __init__(self):
        self.oai_client = OpenAI()
        self.anthropic_client = anthropic.Anthropic()

    @classmethod
    def get_model_provider(cls, model_name):
        if model_name in cls.alias_to_model_name:
            model_name = cls.alias_to_model_name[model_name]

        return cls.model_name_to_provider[model_name]

    @overload
    def chat(  # noqa: E704
        self,
        *,
        model_name: OpenAIModels,
        turn_container: Literal[None] = None,
        message_dicts: List[Dict[str, str]],
        system: Optional[str] = None,
        system_idx: int = -1,
        tool_names_or_tool_defs: List[Union[str, Dict]] = None,
        stream: bool = False,
        logprobs: bool = False,
        response_format: Optional[BaseModel] = None,
        extra_tool_registry: Optional[ToolRegistry] = None,
        **kwargs,
    ): ...

    @overload
    def chat(  # noqa: E704
        self,
        *,
        model_name: AnthropicModels,
        turn_container: Literal[None] = None,
        message_dicts: List[Dict[str, str]],
        system: Optional[str] = None,
        system_idx: Literal[None] = None,
        tool_names_or_tool_defs: List[Union[str, Dict]] = None,
        stream: bool = False,
        logprobs: bool = False,
        response_format: Optional[BaseModel] = None,
        extra_tool_registry: Optional[ToolRegistry] = None,
        **kwargs,
    ): ...

    @overload
    def chat(  # noqa: E704
        self,
        *,
        model_name: str,
        turn_container: TurnContainer,
        message_dicts: Literal[None] = None,
        system: Literal[None] = None,
        system_idx: Literal[None] = None,
        tool_names_or_tool_defs: List[Union[str, Dict]] = None,
        stream: bool = False,
        logprobs: bool = False,
        response_format: Optional[BaseModel] = None,
        extra_tool_registry: Optional[ToolRegistry] = None,
        **kwargs,
    ): ...

    def chat(
        self,
        *,
        model_name: str,
        turn_container: Optional[TurnContainer] = None,
        message_dicts: Optional[List[Dict[str, str]]] = None,
        system: Optional[str] = None,
        system_idx: int = -1,
        tool_names_or_tool_defs: Optional[List[Union[str, Dict]]] = None,
        stream: bool = False,
        logprobs: bool = False,
        response_format: Optional[BaseModel] = None,
        extra_tool_registry: Optional[ToolRegistry] = None,
        **kwargs,
    ):
        if model_name in self.alias_to_model_name:
            model_name = self.alias_to_model_name[model_name]
        model_provider = self.get_model_provider(model_name)

        tools = None
        if tool_names_or_tool_defs is not None:
            if type(tool_names_or_tool_defs[0]) is str:
                tools = ToolRegistry.get_tool_defs_from_names(
                    tool_names_or_tool_defs,
                    model_provider,
                    extra_tool_registry,
                )
            else:
                tools = tool_names_or_tool_defs
                if (
                    extra_tool_registry.model_provider_to_tool_def.get(
                        model_provider, None
                    )
                    is not None
                ):
                    tools += extra_tool_registry.model_provider_to_tool_def.get(
                        model_provider
                    ).values()

        message_manager = None
        if turn_container is not None:
            message_manager = turn_container.model_provider_to_message_manager[
                model_provider
            ]
            messages = message_manager.message_dicts
        elif message_dicts is not None:
            messages = message_dicts

        if model_provider == ModelProvider.OPENAI:
            if system is not None:
                system_dict = {"role": "system", "content": system}
                if system_idx == -1:
                    messages.append(system_dict)
                else:
                    messages.insert(system_idx, system_dict)

            return self.oai_chat(
                model_name, messages, tools, stream, logprobs, response_format, **kwargs
            )
        elif model_provider == ModelProvider.ANTHROPIC:
            if message_manager is not None:
                system = message_manager.system
            return self.ant_chat(
                model_name, messages, system, tools, stream, response_format, **kwargs
            )

    def oai_chat(
        self,
        model_name,
        messages,
        tools=None,
        stream=False,
        logprobs=False,
        response_format=None,
        **kwargs,
    ):
        chat_fn = (
            self.oai_client.chat.completions.create
            if response_format is None
            else self.oai_client.beta.chat.completions.parse
        )
        args = {
            "model": model_name,
            "messages": messages,
            "tools": tools,
            "stream": stream,
            "logprobs": logprobs,
            "response_format": response_format,
            **kwargs,
        }
        if response_format and stream:
            # Streaming with response_format is currently not supported (by me)
            raise Exception("cannot both have response_format and stream defined")
        if not tools:
            args.pop("tools")

        return OAIModelOutput(chat_fn(**args), stream, response_format)

    def ant_chat(
        self,
        model_name,
        messages,
        system=None,
        tools=None,
        stream=False,
        response_format=None,
        **kwargs,
    ):
        tool_choice = None
        if response_format is not None:
            if stream:
                # Streaming with response_format is currently not supported (by me)
                raise Exception("cannot both have response_format and stream defined")
            if tools:
                raise Exception("cannot both have response_format and tools defined")

            tools = [
                {
                    "name": "respond_fn",
                    "description": "provide your response by calling this function with the adequate args",
                    "input_schema": response_format.model_json_schema(),
                }
            ]
            tool_choice = {"type": "tool", "name": "respond_fn"}

        args = {
            "max_tokens": 8192,
            "model": model_name,
            "system": system,
            "messages": messages,
            "tools": tools,
            "stream": stream,
            **kwargs,
        }
        if tool_choice:
            args["tool_choice"] = tool_choice
        if not tools:
            args.pop("tools")
        if not system:
            args.pop("system")

        return AnthropicModelOutput(
            self.anthropic_client.messages.create(**args), stream, response_format
        )


class FunctionCall(BaseModel):
    function_name: str
    tool_call_id: str
    function_args_json: Optional[str] = None
    function_args: Optional[Dict] = None

    @model_validator(mode="after")
    def check_function_args(self):
        if self.function_args_json is None and self.function_args is None:
            raise ValueError(
                "One of [function_args_json, function_args] must be provided"
            )

        if self.function_args_json is not None and self.function_args is None:
            if self.function_args_json:
                self.function_args = json.loads(self.function_args_json)
            else:
                # This case always happens when claude models call inexistent functions.
                # We still want to construct the function call and let it error downstream.
                self.function_args = {}
                self.function_args_json = "{}"
        if self.function_args is not None and self.function_args_json is None:
            self.function_args_json = json.dumps(self.function_args)
        return self


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
        if self.msg_content and self.model_provider != ModelProvider.ANTHROPIC:
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
        self.message_dicts = []
        self.conversation_dicts = []
        self.last_node_id = None
        self.tool_call_ids = []
        self.tool_fn_return_names = set()
        self.index_tracker = ListIndexTracker()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.model_provider is None:
            raise TypeError(f"{cls.__name__} must define 'model_provider'")

    def add_user_turn(self, turn):
        user_msgs = turn.build_messages(self.model_provider)
        self.message_dicts.extend(user_msgs)
        self.conversation_dicts.extend(user_msgs)

    def add_node_turn(
        self,
        turn,
        remove_prev_fn_return_schema=None,
        remove_prev_tool_calls=False,
    ):
        if remove_prev_tool_calls:
            assert remove_prev_fn_return_schema is not False

        if remove_prev_fn_return_schema is True or remove_prev_tool_calls:
            for toll_return in self.tool_fn_return_names:
                self.remove_fn_output_schema(toll_return)

            self.tool_fn_return_names = set()

        if remove_prev_tool_calls:
            for tool_call_id in self.tool_call_ids:
                self.remove_fn_call(tool_call_id)
                self.remove_fn_output(tool_call_id)

            self.tool_call_ids = []

        self.last_node_id = turn.node_id

    @abstractmethod
    def remove_fn_call(self, tool_call_id):
        raise NotImplementedError

    @abstractmethod
    def remove_fn_output(self, tool_call_id):
        raise NotImplementedError

    @abstractmethod
    def remove_fn_output_schema(self, fn_name):
        raise NotImplementedError

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
                {"role": "assistant", "content": turn.msg_content}
            )
        self.parse_assistant_messages(turn.build_messages(self.model_provider))


class OAIMessageManager(MessageManager):
    model_provider = ModelProvider.OPENAI

    def parse_system_messages(self, msgs):
        self.message_dicts.extend(msgs)

    def remove_fn_call(self, tool_call_id):
        idx_to_remove = self.index_tracker.pop_idx(tool_call_id)
        del self.message_dicts[idx_to_remove]

    def remove_fn_output(self, tool_call_id):
        idx_to_remove = self.index_tracker.pop_idx(tool_call_id + "return")
        del self.message_dicts[idx_to_remove]

    def remove_fn_output_schema(self, fn_name):
        idx_to_remove = self.index_tracker.pop_idx(fn_name)
        del self.message_dicts[idx_to_remove]

    def add_node_turn(
        self,
        turn,
        remove_prev_fn_return_schema=None,
        remove_prev_tool_calls=False,
    ):
        if self.last_node_id is not None:
            idx_to_remove = self.index_tracker.pop_idx(self.last_node_id)
            del self.message_dicts[idx_to_remove]
        super().add_node_turn(
            turn, remove_prev_fn_return_schema, remove_prev_tool_calls
        )

        self.message_dicts.extend(turn.build_oai_messages())
        self.index_tracker.add_idx(turn.node_id, len(self.message_dicts) - 1)

    def parse_assistant_messages(self, msgs):
        curr_fn_name = None
        for message in msgs:
            if message.get("tool_calls", None) is not None:
                tool_call_id = message["tool_calls"][0]["id"]
                self.tool_call_ids.append(tool_call_id)
                curr_fn_name = message["tool_calls"][0]["function"]["name"]
                self.index_tracker.add_idx(tool_call_id, len(self.message_dicts))
            elif message["role"] == "tool":
                tool_call_id = message["tool_call_id"]
                self.index_tracker.add_idx(
                    tool_call_id + "return", len(self.message_dicts)
                )
            elif message["role"] == "system" and curr_fn_name is not None:
                if curr_fn_name in self.tool_fn_return_names:
                    idx_to_remove = self.index_tracker.get_idx(curr_fn_name)
                    del self.message_dicts[idx_to_remove]

                self.tool_fn_return_names.add(curr_fn_name)
                self.index_tracker.add_idx(curr_fn_name, len(self.message_dicts))
                curr_fn_name = None

            self.message_dicts.append(message)


class AnthropicMessageManager(MessageManager):
    model_provider = ModelProvider.ANTHROPIC

    def __init__(self):
        super().__init__()
        self.system = None

    def parse_system_messages(self, msgs):
        return

    def remove_fn_call(self, tool_call_id):
        idx_to_remove = self.index_tracker.get_idx(tool_call_id)
        message = self.message_dicts[idx_to_remove]
        new_contents = []
        for content in message["content"]:
            if content["type"] == "tool_use" and content["id"] == tool_call_id:
                continue
            new_contents.append(content)

        if new_contents:
            if len(new_contents) == 1 and new_contents[0]["type"] == "text":
                new_message = {"role": "assistant", "content": new_contents[0]["text"]}
            else:
                new_message = {"role": "assistant", "content": new_contents}
            self.message_dicts[idx_to_remove] = new_message
            self.index_tracker.pop_idx(tool_call_id, shift_idxs=False)
        else:
            del self.message_dicts[idx_to_remove]
            self.index_tracker.pop_idx(tool_call_id)

    def remove_fn_output(self, tool_call_id):
        idx_to_remove = self.index_tracker.get_idx(tool_call_id + "return")
        message = self.message_dicts[idx_to_remove]
        new_contents = []
        for content in message["content"]:
            if (
                content["type"] == "tool_result"
                and content["tool_use_id"] == tool_call_id
            ):
                continue
            new_contents.append(content)

        if new_contents:
            new_message = {"role": "user", "content": new_contents}
            self.message_dicts[idx_to_remove] = new_message
            self.index_tracker.pop_idx(tool_call_id + "return", shift_idxs=False)
        else:
            del self.message_dicts[idx_to_remove]
            self.index_tracker.pop_idx(tool_call_id + "return")

    def remove_fn_output_schema(self, fn_name):
        return

    def add_node_turn(
        self,
        turn,
        remove_prev_fn_return_schema=None,
        remove_prev_tool_calls=False,
    ):
        super().add_node_turn(
            turn, remove_prev_fn_return_schema, remove_prev_tool_calls
        )
        self.system = turn.msg_content

    def parse_assistant_messages(self, messages):
        if len(messages) == 2:
            [message_1, message_2] = messages
        else:
            [message_1] = messages
            message_2 = None

        contents = message_1["content"]
        if type(contents) == list:
            for content in contents:
                if content["type"] == "tool_use":
                    tool_call_id = content["id"]
                    self.tool_call_ids.append(tool_call_id)
                    self.index_tracker.add_idx(tool_call_id, len(self.message_dicts))

        self.message_dicts.append(message_1)

        if message_2 is not None:
            for content in message_2["content"]:
                if content["type"] == "tool_result":
                    tool_id = content["tool_use_id"]
                    self.index_tracker.add_idx(
                        tool_id + "return", len(self.message_dicts)
                    )
            self.message_dicts.append(message_2)


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
        node_id,
        node_prompt,
        remove_prev_tool_fn_return=None,
        remove_prev_tool_calls=False,
    ):
        turn = NodeSystemTurn(node_id=node_id, msg_content=node_prompt)
        self.turns.append(turn)
        for mm in self.model_provider_to_message_manager.values():
            mm.add_node_turn(turn, remove_prev_tool_fn_return, remove_prev_tool_calls)

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


class ModelOutput(ABC):
    def __init__(self, output_obj, is_stream, response_format=None):
        self.output_obj = output_obj
        self.is_stream = is_stream
        self.response_format = response_format
        self.parsed_msg = None
        self.msg_content = None
        self.last_chunk = None
        self.fn_calls = []

    @abstractmethod
    def get_message(self):
        raise NotImplementedError

    @abstractmethod
    def get_fn_calls(self):
        raise NotImplementedError

    def get_or_stream_message(self):
        if self.is_stream:
            return self.stream_message()
        else:
            return self.get_message()

    def get_or_stream_fn_calls(self):
        if self.is_stream:
            return self.stream_fn_calls()
        else:
            return self.get_fn_calls()

    @abstractmethod
    def is_message_start_chunk(self):
        raise NotImplementedError

    @abstractmethod
    def is_tool_start_chunk(self):
        raise NotImplementedError

    @abstractmethod
    def is_final_chunk(self):
        raise NotImplementedError

    @abstractmethod
    def has_function_call_id(self, chunk):
        raise NotImplementedError

    @abstractmethod
    def get_msg_from_chunk(self, chunk):
        raise NotImplementedError

    @abstractmethod
    def has_msg_content(self, chunk):
        raise NotImplementedError

    @abstractmethod
    def has_fn_args_json(self, chunk):
        raise NotImplementedError

    @abstractmethod
    def get_fn_call_id_from_chunk(self, chunk):
        raise NotImplementedError

    @abstractmethod
    def get_fn_name_from_chunk(self, chunk):
        raise NotImplementedError

    @abstractmethod
    def get_fn_args_json_from_chunk(self, chunk):
        raise NotImplementedError

    def stream_message(self):
        chunk = self.get_next_usable_chunk()
        self.last_chunk = chunk
        if self.is_message_start_chunk(chunk):
            return self._stream_message(chunk)
        else:
            return None

    def _stream_message(self, chunk):
        self.msg_content = ""
        while not self.has_msg_content(chunk):
            chunk = next(self.output_obj)
        first_msg = self.get_msg_from_chunk(chunk)
        self.msg_content += first_msg
        yield first_msg
        try:
            while True:
                chunk = next(self.output_obj)  # Get the next chunk
                if self.has_msg_content(chunk):
                    msg = self.get_msg_from_chunk(chunk)
                    self.msg_content += msg  # Append the message to full_msg
                    yield msg  # Return the message
                else:
                    self.last_chunk = chunk
                    raise StopIteration
        except StopIteration:
            pass  # Signal end of iteration

    def stream_fn_calls(self):
        self.last_chunk = self.get_next_usable_chunk()
        function_name = None
        tool_call_id = None
        function_args_json = None

        for chunk in itertools.chain([self.last_chunk], self.output_obj):
            if self.has_function_call_id(chunk):
                if tool_call_id is not None:
                    fn_call = FunctionCall(
                        function_name=function_name,
                        tool_call_id=tool_call_id,
                        function_args_json=function_args_json,
                    )
                    self.fn_calls.append(fn_call)
                    yield fn_call

                function_name = self.get_fn_name_from_chunk(chunk)
                tool_call_id = self.get_fn_call_id_from_chunk(chunk)
                function_args_json = ""
            elif tool_call_id is not None and self.has_fn_args_json(chunk):
                function_args_json += self.get_fn_args_json_from_chunk(chunk)

        if tool_call_id is not None:
            fn_call = FunctionCall(
                function_name=function_name,
                tool_call_id=tool_call_id,
                function_args_json=function_args_json,
            )

            self.fn_calls.append(fn_call)
            yield fn_call

    def get_next_usable_chunk(self):
        if self.last_chunk is None:
            chunk = next(self.output_obj)
        else:
            chunk = self.last_chunk
        while not (
            self.is_message_start_chunk(chunk)
            or self.is_tool_start_chunk(chunk)
            or self.is_final_chunk(chunk)
        ):
            chunk = next(self.output_obj)
        return chunk


class OAIModelOutput(ModelOutput):
    def _get_tool_call(self, chunk):
        return chunk.choices[0].delta.tool_calls[0]

    def get_fn_call_id_from_chunk(self, chunk):
        return self._get_tool_call(chunk).id

    def get_fn_name_from_chunk(self, chunk):
        return self._get_tool_call(chunk).function.name

    def get_fn_args_json_from_chunk(self, chunk):
        return self._get_tool_call(chunk).function.arguments

    def _has_tool_call(self, chunk):
        return bool(chunk.choices[0].delta.tool_calls)

    def has_function_call_id(self, chunk):
        return self._has_tool_call(chunk) and self._get_tool_call(chunk).id is not None

    def is_tool_start_chunk(self, chunk):
        return self.has_function_call_id(chunk)

    def has_fn_args_json(self, chunk):
        return (
            self._has_tool_call(chunk)
            and self._get_tool_call(chunk).function.arguments is not None
        )

    def is_message_start_chunk(self, chunk):
        return self.has_msg_content(chunk)

    def has_msg_content(self, chunk):
        return chunk.choices[0].delta.content is not None

    def get_msg_from_chunk(self, chunk):
        return chunk.choices[0].delta.content

    def is_final_chunk(self, chunk):
        return chunk.choices[0].finish_reason is not None

    def get_message(self):
        self.msg_content = self.output_obj.choices[0].message.content
        return self.msg_content

    def get_message_prop(self, prop_name):
        if self.parsed_msg is None:
            self.parsed_msg = self.output_obj.choices[0].message.parsed
        return getattr(self.parsed_msg, prop_name)

    def get_logprob(self, token_idx):
        return self.output_obj.choices[0].logprobs.content[token_idx].logprob

    def get_prob(self, token_idx):
        return np.exp(self.get_logprob(token_idx))

    def get_fn_calls(self):
        tool_calls = self.output_obj.choices[0].message.tool_calls or []
        for tool_call in tool_calls:
            fn_call = FunctionCall(
                function_name=tool_call.function.name,
                tool_call_id=tool_call.id,
                function_args_json=tool_call.function.arguments,
            )
            self.fn_calls.append(fn_call)
            yield fn_call


class AnthropicModelOutput(ModelOutput):
    def get_fn_call_id_from_chunk(self, chunk):
        return chunk.content_block.id

    def get_fn_name_from_chunk(self, chunk):
        return chunk.content_block.name

    def get_fn_args_json_from_chunk(self, chunk):
        return chunk.delta.partial_json

    def has_fn_args_json(self, chunk):
        return self._is_delta_chunk(chunk) and hasattr(chunk.delta, "partial_json")

    def _is_content_block(self, chunk):
        return hasattr(chunk, "content_block")

    def _is_delta_chunk(self, chunk):
        return hasattr(chunk, "delta")

    def is_message_start_chunk(self, chunk):
        return self._is_content_block(chunk) and chunk.content_block.type == "text"

    def is_tool_start_chunk(self, chunk):
        return self._is_content_block(chunk) and chunk.content_block.type == "tool_use"

    def is_end_block_chunk(self, chunk):
        return chunk.type == "content_block_stop"

    def is_final_chunk(self, chunk):
        return chunk.type == "message_stop"

    def has_msg_content(self, chunk):
        return (
            self._is_delta_chunk(chunk)
            and getattr(chunk.delta, "text", None) is not None
        )

    def has_function_call_id(self, chunk):
        return self._is_content_block(chunk) and hasattr(chunk.content_block, "id")

    def get_msg_from_chunk(self, chunk):
        return chunk.delta.text

    def get_message(self):
        content = self.output_obj.content[0]
        if content.type == "text":
            self.msg_content = content.text
            return self.msg_content
        else:
            return None

    def get_message_prop(self, prop_name):
        if self.parsed_msg is None:
            fn_call = next(self.get_fn_calls())
            self.parsed_msg = self.response_format(**fn_call.function_args)
        return getattr(self.parsed_msg, prop_name)

    def get_fn_calls(self):
        for content in self.output_obj.content:
            if content.type == "tool_use":
                fn_call = FunctionCall(
                    function_name=content.name,
                    tool_call_id=content.id,
                    function_args=content.input,
                )
                self.fn_calls.append(fn_call)
                yield fn_call


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        elif isinstance(obj, (defaultdict, dict)):
            return {self.default(k): self.default(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.default(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, ToolExceptionWrapper):
            return str(obj)
        return super().default(obj)


class ListIndexTracker:
    def __init__(self):
        self.named_idx_to_idx = {}
        self.idx_to_named_idx = defaultdict(set)
        self.idxs = []
        self.idx_to_pos = {}

    def add_idx(self, named_idx, idx):
        self.named_idx_to_idx[named_idx] = idx
        self.idx_to_named_idx[idx].add(named_idx)
        if idx not in self.idxs:
            self.idxs.append(idx)
            self.idx_to_pos[idx] = len(self.idxs) - 1

    def get_idx(self, named_idx):
        return self.named_idx_to_idx[named_idx]

    def pop_idx(self, named_idx, shift_idxs=True):
        popped_idx = self.named_idx_to_idx.pop(named_idx)
        named_idxs = self.idx_to_named_idx[popped_idx]

        named_idxs.remove(named_idx)
        if not named_idxs:
            popped_idx_pos = self.idx_to_pos.pop(popped_idx)
            self.idx_to_named_idx.pop(popped_idx)
            del self.idxs[popped_idx_pos]

            for i in range(popped_idx_pos, len(self.idxs)):
                curr_idx = self.idxs[i]
                self.idx_to_pos.pop(curr_idx)
                if shift_idxs:
                    curr_named_idxs = self.idx_to_named_idx[curr_idx]

                    self.idxs[i] -= 1
                    self.idx_to_pos[self.idxs[i]] = i

                    for curr_named_idx in curr_named_idxs:
                        self.named_idx_to_idx[curr_named_idx] = self.idxs[i]
                    self.idx_to_named_idx.pop(curr_idx)
                    self.idx_to_named_idx[self.idxs[i]] = curr_named_idxs
                else:
                    self.idx_to_pos[curr_idx] = i

            return popped_idx
