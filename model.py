import itertools
import json
from collections import defaultdict
from enum import StrEnum
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod, ABCMeta

import anthropic
import numpy as np
from openai import OpenAI
from pydantic import BaseModel, Field, constr

from model_tool_decorator import (
    ANTHROPIC_TOOL_NAME_TO_TOOL_DEF,
    OPENAI_TOOL_NAME_TO_TOOL_DEF,
    OPENAI_TOOLS_RETUN_DESCRIPTION,
)


class ModelProvider(StrEnum):
    OPENAI = "OPENAI"
    ANTHROPIC = "ANTHROPIC"


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

    def get_tool_defs_from_names(self, tool_names, tool_defs, extra_tool_defs):
        all_tool_defs = tool_defs
        if extra_tool_defs is not None:
            all_tool_defs |= extra_tool_defs

        return [all_tool_defs[tool_name] for tool_name in tool_names]

    def chat(
        self,
        model_name,
        turn_container,
        tool_names=None,
        tools=None,
        stream=False,
        logprobs=False,
        response_format=None,
        extra_oai_tool_defs=None,
        extra_anthropic_tool_defs=None,
        **kwargs,
    ):
        if model_name in self.alias_to_model_name:
            model_name = self.alias_to_model_name[model_name]

        model_provider = self.model_name_to_provider[model_name]
        message_manager = turn_container.model_provider_to_message_manager[
            model_provider
        ]
        if model_provider == ModelProvider.OPENAI:
            if tool_names:
                tools = self.get_tool_defs_from_names(
                    tool_names, OPENAI_TOOL_NAME_TO_TOOL_DEF, extra_oai_tool_defs
                )
            messages = message_manager.get_chat_input()
            return self.oai_chat(
                model_name, messages, tools, stream, logprobs, response_format, **kwargs
            )
        elif model_provider == ModelProvider.ANTHROPIC:
            if tool_names:
                tools = self.get_tool_defs_from_names(
                    tool_names,
                    ANTHROPIC_TOOL_NAME_TO_TOOL_DEF,
                    extra_anthropic_tool_defs,
                )

            messages, system_prompt = message_manager.get_chat_input()
            return self.ant_chat(
                model_name, messages, system_prompt, tools, stream, **kwargs
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
        if response_format is not None:
            args.pop("stream")
        if not tools:
            args.pop("tools")

        return OAIModelOutput(chat_fn(**args), stream)

    def ant_chat(
        self,
        model_name,
        messages,
        system=None,
        tools=None,
        stream=False,
        **kwargs,
    ):
        # TODO: remove adhoc code
        args = {
            "max_tokens": 8192,
            "model": model_name,
            "system": system,
            "messages": messages[1:],
            "tools": tools,
            "stream": stream,
            **kwargs,
        }
        if not tools:
            args.pop("tools")

        return AnthropicModelOutput(
            self.anthropic_client.messages.create(**args), stream
        )


class FunctionCall(BaseModel):
    function_name: str
    tool_call_id: str
    function_args_json: str


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
        return {"role": "user", "content": self.msg_content}

    def build_anthropic_messages(self):
        return self.build_oai_messages()


class SystemTurn(ModelTurn):
    def build_oai_messages(self):
        return {"role": "system", "content": self.msg_content}
    
    def build_anthropic_messages(self):
        return None


class NodeSystemTurn(SystemTurn):
    node_id: int

    def build_oai_messages(self):
        return {"role": "system", "content": self.msg_content}
    
    def build_anthropic_messages(self):
        return None


class AssistantTurn(ModelTurn):
    msg_content: Optional[str]
    fn_calls: Optional[List[FunctionCall]] = Field(default_factory=list)
    fn_call_id_to_fn_output: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def build_oai_messages(self):
        messages = []
        if self.msg_content:
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

                if fn_call.function_name in OPENAI_TOOLS_RETUN_DESCRIPTION:
                    json_schema = OPENAI_TOOLS_RETUN_DESCRIPTION[fn_call.function_name]
                    system_msg = f"This is the JSON Schema of {fn_call.function_name}'s return type: {json.dumps(json_schema)}"

                    messages.append({"role": "system", "content": system_msg})

        return messages

    def build_anthropic_messages(self):
        contents = []
        messages = []
        if self.msg_content:
            contents.append({"type": "text", "text": self.msg_content})
        if self.fn_calls:
            for fn_call in self.fn_calls:
                contents.append(
                    {
                        "type": "tool_use",
                        "id": fn_call.tool_call_id,
                        "name": fn_call.function_name,
                        "input": json.loads(fn_call.function_args_json),
                    }
                )

        if len(contents) == 1 and self.msg_content:
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
                    }
                )

            messages.append({"role": "user", "content": return_contents})

        return messages


class MessageManager(ABC):
    model_provider = None

    def __init__(self):
        self.message_dicts = []
        self.last_node_id = None
        self.tool_call_ids = []
        self.tool_fn_return_names = set()
        self.index_tracker = ListIndexTracker()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.model_provider is None:
            raise TypeError(f"{cls.__name__} must define 'model_provider'")
        
    def add_user_turn(self, turn):
        self.message_dicts.append(turn.build_messages(self.model_provider))

    def add_node_turn(
        self,
        turn,
        remove_prev_tool_fn_return=None,
        remove_prev_tool_calls=False,
    ):
        if remove_prev_tool_calls:
            assert remove_prev_tool_fn_return is not False

        if remove_prev_tool_fn_return is True or remove_prev_tool_calls:
            for toll_return in self.tool_fn_return_names:
                self.remove_fn_output_schema(toll_return)

            self.tool_fn_return_names = set()

        if remove_prev_tool_calls:
            for tool_call_id in self.tool_call_ids:
                self.remove_fn_call(tool_call_id)
                self.remove_fn_output(tool_call_id)

            self.tool_call_ids = []

        self.last_node_id = turn.node_id


class OAIMessageManager(MessageManager):
    model_provider = ModelProvider.OPENAI

    def add_system_turn(self, turn):
        self.message_dicts.append(turn.build_oai_messages())

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
        remove_prev_tool_fn_return=None,
        remove_prev_tool_calls=False,
    ):
        if self.last_node_id is not None:
            idx_to_remove = self.index_tracker.pop_idx(self.last_node_id)
            del self.message_dicts[idx_to_remove]
        super().add_node_turn(turn, remove_prev_tool_fn_return, remove_prev_tool_calls)

        self.message_dicts.append(turn.build_oai_messages())
        self.index_tracker.add_idx(turn.node_id, len(self.message_dicts) - 1)

    def add_assistant_turn(self, turn):
        messages = turn.build_oai_messages()
        last_fn_name = None
        for message in messages:
            if message.get("tool_calls", None) is not None:
                tool_call_id = message["tool_calls"][0]["id"]
                self.tool_call_ids.append(tool_call_id)
                last_fn_name = message["tool_calls"][0]["function"]["name"]
                self.index_tracker.add_idx(tool_call_id, len(self.message_dicts))
            elif message["role"] == "tool":
                tool_call_id = message["tool_call_id"]
                self.index_tracker.add_idx(
                    tool_call_id + "return", len(self.message_dicts)
                )
            elif message["role"] == "system" and last_fn_name is not None:
                if last_fn_name in self.tool_fn_return_names:
                    idx_to_remove = self.index_tracker.get_idx(last_fn_name)
                    del self.message_dicts[idx_to_remove]

                self.tool_fn_return_names.add(last_fn_name)
                self.index_tracker.add_idx(last_fn_name, len(self.message_dicts))
                last_fn_name = None

            self.message_dicts.append(message)

    def get_chat_input(self):
        return self.message_dicts


class AnthropicMessageManager(MessageManager):
    model_provider = ModelProvider.ANTHROPIC

    def add_system_turn(self, turn):
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
            if new_contents[0]["type"] == "text":
                # TODO: i prob also want to remove these texts because they are usually internal reflections
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
            new_message = {"role": "assistant", "content": new_contents}
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
        remove_prev_tool_fn_return=None,
        remove_prev_tool_calls=False,
    ):
        super().add_node_turn(turn, remove_prev_tool_fn_return, remove_prev_tool_calls)
        self.system = turn.msg_content

    def add_assistant_turn(self, turn):
        messages = turn.build_anthropic_messages()
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

    def get_chat_input(self):
        return self.message_dicts, self.system


class TurnContainer:
    model_provider_to_message_manager_cls = {
        ModelProvider.OPENAI: OAIMessageManager,
        ModelProvider.ANTHROPIC: AnthropicMessageManager,
    }

    def __init__(self, model_providers=[ModelProvider.OPENAI, ModelProvider.ANTHROPIC]):
        self.message_managers = []
        self.model_provider_to_message_manager = {}
        for provider in model_providers:
            mm = self.model_provider_to_message_manager_cls[provider]()
            self.message_managers.append(mm)
            self.model_provider_to_message_manager[provider] = mm

        self.turns = []

    def add_system_turn(self, msg_content):
        turn = SystemTurn(msg_content=msg_content)
        self.turns.append(turn)
        for mm in self.message_managers:
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
        for mm in self.message_managers:
            mm.add_node_turn(turn, remove_prev_tool_fn_return, remove_prev_tool_calls)

    def add_user_turn(self, msg_content):
        turn = UserTurn(msg_content=msg_content)
        self.turns.append(turn)
        for mm in self.message_managers:
            mm.add_user_turn(turn)

    def add_assistant_turn(self, msg_content, fn_calls=None, fn_id_to_outputs=None):
        turn = AssistantTurn(
            msg_content=msg_content,
            fn_calls=fn_calls,
            fn_call_id_to_fn_output=fn_id_to_outputs,
        )
        self.turns.append(turn)
        for mm in self.message_managers:
            mm.add_assistant_turn(turn)


class ModelOutput:
    def __init__(self, output_obj, is_stream):
        self.output_obj = output_obj
        self.is_stream = is_stream
        self.msg_content = None
        self.current_chunk = None
        self.fn_calls = []

    def stream_message(self):
        raise NotImplementedError

    def get_message(self):
        raise NotImplementedError

    def stream_fn_calls(self):
        raise NotImplementedError

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


class OAIModelOutput(ModelOutput):

    def _has_function_call_id(self, chunk):
        return (
            chunk.choices[0].delta.tool_calls is not None
            and chunk.choices[0].delta.tool_calls[0].id is not None
        )

    def _has_msg_content(self, chunk):
        return chunk.choices[0].delta.content is not None

    def _get_first_usable_chunk(self):
        chunk = next(self.output_obj)
        while not (self._has_function_call_id(chunk) or self._has_msg_content(chunk)):
            chunk = next(self.output_obj)
        return chunk

    def stream_message(self):
        first_chunk = self._get_first_usable_chunk()
        self.current_chunk = first_chunk
        if self._has_msg_content(first_chunk):
            return self._stream_message()
        else:
            return None

    def _stream_message(self):
        self.msg_content = ""
        first_msg = self.current_chunk.choices[0].delta.content
        self.msg_content += first_msg
        yield first_msg

        try:
            while True:
                chunk = next(self.output_obj)  # Get the next chunk
                msg = chunk.choices[0].delta.content
                finish_reason = chunk.choices[0].finish_reason
                if finish_reason is not None:
                    raise StopIteration
                if msg is None:
                    self.current_chunk = chunk
                    raise StopIteration
                self.msg_content += msg  # Append the message to full_msg
                yield msg  # Return the message
        except StopIteration:
            pass  # Signal end of iteration

    def stream_fn_calls(self):
        function_name = None
        tool_call_id = None
        function_args_json = None

        for chunk in itertools.chain([self.current_chunk], self.output_obj):
            finish_reason = chunk.choices[0].finish_reason
            if finish_reason is not None:
                break
            elif self._has_function_call_id(chunk):
                if self.current_chunk != chunk:
                    fn_call = FunctionCall(
                        function_name=function_name,  # noqa
                        tool_call_id=tool_call_id,  # noqa
                        function_args_json=function_args_json,  # noqa
                    )
                    self.fn_calls.append(fn_call)
                    yield fn_call

                function_name = chunk.choices[0].delta.tool_calls[0].function.name
                tool_call_id = chunk.choices[0].delta.tool_calls[0].id
                function_args_json = ""
            elif tool_call_id is not None:
                function_args_json += (
                    chunk.choices[0].delta.tool_calls[0].function.arguments
                )

        if tool_call_id is not None:
            fn_call = FunctionCall(
                function_name=function_name,
                tool_call_id=tool_call_id,
                function_args_json=function_args_json,
            )

            self.fn_calls.append(fn_call)
            yield fn_call

    def get_message(self):
        self.msg_content = self.output_obj.choices[0].message.content
        return self.msg_content

    def get_message_prop(self, prop_name):
        return getattr(self.output_obj.choices[0].message.parsed, prop_name)

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
    def is_message_start_chunk(self, chunk):
        content_block = getattr(chunk, "content_block", None)
        return content_block is not None and content_block.type == "text"

    def is_tool_start_chunk(self, chunk):
        content_block = getattr(chunk, "content_block", None)
        return content_block is not None and content_block.type == "tool_use"

    def is_end_block_chunk(self, chunk):
        return chunk.type == "content_block_stop"

    def is_message_end_chunk(self, chunk):
        return chunk.type == "message_stop"

    def get_next_usable_chunk(self):
        if self.current_chunk is None:
            chunk = next(self.output_obj)
        else:
            chunk = self.current_chunk
        while not (
            self.is_message_start_chunk(chunk)
            or self.is_tool_start_chunk(chunk)
            or self.is_message_end_chunk(chunk)
        ):
            chunk = next(self.output_obj)
        return chunk

    def get_message(self):
        self.msg_content = self.output_obj.content[0].text
        return self.msg_content

    def stream_message(self):
        first_chunk = self.get_next_usable_chunk()
        self.current_chunk = first_chunk
        if self.is_message_start_chunk(first_chunk):
            self.current_chunk = next(self.output_obj)
            return self._stream_message()
        else:
            return None

    def _stream_message(self):
        self.msg_content = ""
        first_msg = self.current_chunk.delta.text
        self.msg_content += first_msg
        yield first_msg

        try:
            while True:
                chunk = next(self.output_obj)  # Get the next chunk
                chunk_type = chunk.type
                if chunk_type == "content_block_stop":
                    self.current_chunk = next(self.output_obj)
                    raise StopIteration

                msg = chunk.delta.text
                self.msg_content += msg  # Append the message to full_msg
                yield msg  # Return the message
        except StopIteration:
            pass  # Signal end of iteration

    def stream_fn_calls(self):
        first_chunk = self.get_next_usable_chunk()

        function_name = None
        tool_call_id = None
        function_args_json = None

        for chunk in itertools.chain([first_chunk], self.output_obj):
            if self.is_tool_start_chunk(chunk):
                function_name = chunk.content_block.name
                tool_call_id = chunk.content_block.id
                function_args_json = ""
            elif self.is_end_block_chunk(chunk):
                fn_call = FunctionCall(
                    function_name=function_name,  # noqa
                    tool_call_id=tool_call_id,  # noqa
                    function_args_json=function_args_json,  # noqa
                )
                self.fn_calls.append(fn_call)
                yield fn_call
                function_name = None
                tool_call_id = None
                function_args_json = None
            elif tool_call_id is not None:
                function_args_json += chunk.delta.partial_json

        if tool_call_id is not None:
            fn_call = FunctionCall(
                function_name=function_name,
                tool_call_id=tool_call_id,
                function_args_json=function_args_json,
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
