from __future__ import annotations

import copy
import itertools
from abc import ABC, abstractmethod
from bisect import bisect_left
from collections import defaultdict
from enum import StrEnum
from typing import Dict, List, Literal, Optional, Union, overload

import anthropic
import numpy as np
from openai import OpenAI
from pydantic import BaseModel

from model_tool_decorator import ToolRegistry
from model_turn import TurnContainer
from model_util import FunctionCall, ModelProvider

OpenAIModels = Literal["gpt-4o-mini", "gpt-4o"]

AnthropicModels = Literal["claude-3.5", "claude-3-5-sonnet-latest"]


class Model:
    model_name_to_provider = {
        "gpt-4o-mini": ModelProvider.OPENAI,
        "gpt-4o": ModelProvider.OPENAI,
        "claude-3-5-sonnet-latest": ModelProvider.ANTHROPIC,
    }
    alias_to_model_name = {"claude-3.5": "claude-3-5-sonnet-latest"}

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

    def get_tool_choice_arg(self, args, model_provider):
        if "force_tool_choice" in args:
            if args["force_tool_choice"] is not None:
                fn_name = args["force_tool_choice"]
                if model_provider == ModelProvider.ANTHROPIC:
                    args["tool_choice"] = {"type": "tool", "name": fn_name}
                elif model_provider == ModelProvider.OPENAI:
                    args["tool_choice"] = {
                        "type": "function",
                        "function": {"name": fn_name},
                    }

            args.pop("force_tool_choice")

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
        if not stream:
            args.pop("stream")

        self.get_tool_choice_arg(args, ModelProvider.OPENAI)

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

        self.get_tool_choice_arg(args, ModelProvider.ANTHROPIC)

        return AnthropicModelOutput(
            self.anthropic_client.messages.create(**args), stream, response_format
        )


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

    def __init__(self, *args, model_provider):
        super().__init__(*args)
        self.uri_to_list_idx = {}
        self.list_idx_to_uris = defaultdict(set)
        self.list_idxs = []
        self.list_idx_to_track_idx = {}

        self.model_provider = model_provider
        self.item_type_to_uris = defaultdict(list)
        self.uri_to_item_type = {}
        self.item_type_to_count = {k: 0 for k in self.item_type_to_uri_prefix.keys()}

    def get_tool_id_from_tool_output_uri(self, uri):
        return uri[
            len(self.item_type_to_uri_prefix[MessageList.ItemType.TOOL_OUTPUT]) :
        ]

    @classmethod
    def get_tool_output_uri_from_tool_id(cls, tool_id):
        return cls.item_type_to_uri_prefix[MessageList.ItemType.TOOL_OUTPUT] + tool_id

    def pop_track_idx_ant(self, uri):
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

    def track_idx(self, item_type, list_idx=None, uri=None, is_insert=False):
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

    def track_idxs(self, item_type, start_list_idx, end_list_idx=None, uris=None):
        if end_list_idx is None:
            end_list_idx = len(self) - 1
        if uris is None:
            range_idx = end_list_idx - start_list_idx + 1
            uris = [None] * range_idx

        for i, uri in zip(range(start_list_idx, end_list_idx + 1), uris):
            self.track_idx(item_type, i, uri)

    def get_track_idx_from_uri(self, uri):
        return self.uri_to_list_idx[uri]

    def get_track_idx_for_item_type(self, item_type, order=-1):
        order_validation = abs(order) if order < 0 else order + 1
        target_uri = (
            self.item_type_to_uris[item_type][order]
            if self.item_type_to_uris[item_type]
            and order_validation <= len(self.item_type_to_uris[item_type])
            else None
        )
        return self.uri_to_list_idx[target_uri] if target_uri else None

    def shift_track_idxs(self, start_track_idx, shift_direction):
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

    def pop_track_idx(self, uri, shift_idxs=True):
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

    def append(self, item, item_type=None, uri=None):
        super().append(item)
        if item_type is not None:
            self.track_idx(item_type, uri=uri)

    def insert(self, idx, item, item_type=None, uri=None):
        super().insert(idx, item)
        if item_type is not None:
            self.track_idx(item_type, idx, uri, is_insert=True)

    def extend(self, items, item_type=None):
        curr_len = len(self) - 1
        super().extend(items)
        if items and item_type is not None:
            self.track_idxs(item_type, curr_len + 1)

    def _remove_by_uri(self, uri, raise_on_unpopped_idx=False):
        popped_idx = self.pop_track_idx(uri)
        if popped_idx is not None:
            del self[popped_idx]
        else:
            if raise_on_unpopped_idx:
                raise ValueError

    def remove_by_uri(self, uri, raise_if_not_found=True):
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

    def clear(self, item_type_or_types=None):
        if item_type_or_types is None:
            super().clear()
        else:
            if not isinstance(item_type_or_types, list):
                item_type_or_types = [item_type_or_types]
            for item_type in item_type_or_types:
                uris = copy.copy(self.item_type_to_uris[item_type])
                for uri in uris:
                    self.remove_by_uri(uri)
