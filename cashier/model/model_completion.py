from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
from anthropic.types.raw_message_stream_event import RawMessageStreamEvent
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDeltaToolCall,
)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from pydantic import BaseModel

from cashier.model.message_list import MessageList
from cashier.model.model_client import ModelClient
from cashier.model.model_turn import AnthropicMessageManager
from cashier.model.model_util import FunctionCall, ModelProvider
from cashier.tool.tool_registry import ToolRegistry
from cashier.turn_container import TurnContainer

OpenAIModel = Literal["gpt-4o-mini", "gpt-4o"]

AnthropicModel = Literal[
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
]

AnthropicModelAlias = Literal[
    "claude-3.5",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
]

ModelName = Union[OpenAIModel, AnthropicModelAlias]

ModelResponseChunkType = TypeVar(
    "ModelResponseChunkType", ChatCompletionChunk, RawMessageStreamEvent
)


class Model:
    model_name_to_provider = {
        "gpt-4o-mini": ModelProvider.OPENAI,
        "gpt-4o": ModelProvider.OPENAI,
        "claude-3-5-sonnet-latest": ModelProvider.ANTHROPIC,
        "claude-3-5-haiku-latest": ModelProvider.ANTHROPIC,
        "claude-3-5-sonnet-20241022": ModelProvider.ANTHROPIC,
        "claude-3-5-haiku-20241022": ModelProvider.ANTHROPIC,
    }
    alias_to_model_name: Dict[ModelName, Union[OpenAIModel, AnthropicModel]] = {
        "claude-3.5": "claude-3-5-sonnet-latest"
    }

    @classmethod
    def get_model_provider(cls, model_name: ModelName) -> ModelProvider:
        if model_name in cls.alias_to_model_name:
            model_name = cls.alias_to_model_name[model_name]

        return cls.model_name_to_provider[model_name]

    @classmethod
    @overload
    def chat(  # noqa: E704
        cls,
        *,
        model_name: OpenAIModel,
        turn_container: Literal[None] = None,
        message_dicts: Union[List[Dict[str, Any]], MessageList],
        system: Optional[str] = None,
        system_idx: int = -1,
        tool_registry: Optional[ToolRegistry] = None,
        stream: bool = False,
        logprobs: bool = False,
        response_format: Optional[Type[BaseModel]] = None,
        exclude_update_state_fns: bool=False,
        **kwargs: Any,
    ) -> ModelOutput: ...

    @classmethod
    @overload
    def chat(  # noqa: E704
        cls,
        *,
        model_name: AnthropicModelAlias,
        turn_container: Literal[None] = None,
        message_dicts: Union[List[Dict[str, Any]], MessageList],
        system: Optional[str] = None,
        system_idx: Literal[None] = None,
        tool_registry: Optional[ToolRegistry] = None,
        stream: bool = False,
        logprobs: bool = False,
        response_format: Optional[Type[BaseModel]] = None,
        exclude_update_state_fns: bool=False,
        **kwargs: Any,
    ) -> ModelOutput: ...

    @classmethod
    @overload
    def chat(  # noqa: E704
        cls,
        *,
        model_name: ModelName,
        turn_container: TurnContainer,
        message_dicts: Literal[None] = None,
        system: Literal[None] = None,
        system_idx: Literal[None] = None,
        tool_registry: Optional[ToolRegistry] = None,
        stream: bool = False,
        logprobs: bool = False,
        response_format: Optional[Type[BaseModel]] = None,
        exclude_update_state_fns: bool=False,
        **kwargs: Any,
    ) -> ModelOutput: ...

    @classmethod
    def chat(
        cls,
        *,
        model_name: ModelName,
        turn_container: Optional[TurnContainer] = None,
        message_dicts: Optional[Union[List[Dict[str, Any]], MessageList]] = None,
        system: Optional[str] = None,
        system_idx: Optional[int] = -1,
        tool_registry: Optional[ToolRegistry] = None,
        stream: bool = False,
        logprobs: bool = False,
        response_format: Optional[Type[BaseModel]] = None,
        exclude_update_state_fns: bool=False,
        **kwargs: Any,
    ) -> ModelOutput:
        if model_name in cls.alias_to_model_name:
            model_name = cls.alias_to_model_name[model_name]
        model_provider = cls.get_model_provider(model_name)

        tools = None
        if tool_registry is not None:
            tools = tool_registry.get_tool_defs(model_provider=model_provider, exclude_update_state_fns=exclude_update_state_fns)

        messages: Union[MessageList, List]
        message_manager = None
        if turn_container is not None:
            message_manager = turn_container.model_provider_to_message_manager[
                model_provider
            ]
            messages = message_manager.message_dicts
        else:
            assert message_dicts is not None
            messages = message_dicts

        if model_provider == ModelProvider.OPENAI:
            if system is not None:
                system_dict = {"role": "system", "content": system}
                assert system_idx is not None
                if system_idx == -1:
                    messages.append(system_dict)
                else:
                    messages.insert(system_idx, system_dict)

            return cls.oai_chat(
                cast(OpenAIModel, model_name),
                messages,
                tools,  # type: ignore
                stream,
                logprobs,
                response_format,
                **kwargs,
            )
        elif model_provider == ModelProvider.ANTHROPIC:
            if message_manager is not None:
                assert isinstance(message_manager, AnthropicMessageManager)
                system = message_manager.system
            return cls.ant_chat(
                cast(AnthropicModel, model_name),
                messages,
                system,
                tools,  # type: ignore
                stream,
                response_format,
                **kwargs,
            )
        else:
            raise ValueError()

    @classmethod
    def get_tool_choice_arg(
        cls, args: Dict[str, Any], model_provider: ModelProvider
    ) -> None:
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

    @classmethod
    def oai_chat(
        cls,
        model_name: OpenAIModel,
        messages: Union[List[Dict[str, Any]], MessageList],
        tools: Optional[List[ChatCompletionToolParam]] = None,
        stream: bool = False,
        logprobs: bool = False,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> OAIModelOutput:
        oai_client = ModelClient.get_client(ModelProvider.OPENAI)

        chat_fn = (
            oai_client.chat.completions.create
            if response_format is None
            else oai_client.beta.chat.completions.parse
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

        cls.get_tool_choice_arg(args, ModelProvider.OPENAI)

        return OAIModelOutput(chat_fn(**args), stream, response_format)  # type: ignore

    @classmethod
    def ant_chat(
        cls,
        model_name: AnthropicModel,
        messages: Union[List[Dict[str, Any]], MessageList],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> AnthropicModelOutput:
        anthropic_client = ModelClient.get_client(ModelProvider.ANTHROPIC)
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

        cls.get_tool_choice_arg(args, ModelProvider.ANTHROPIC)

        return AnthropicModelOutput(
            anthropic_client.messages.create(**args), stream, response_format
        )


class ModelOutput(ABC, Generic[ModelResponseChunkType]):

    def __init__(
        self,
        output_obj: Any,
        is_stream: bool,
        response_format: Optional[Type[BaseModel]] = None,
    ):
        self.output_obj = output_obj
        self.is_stream = is_stream
        self.response_format = response_format
        self.parsed_msg: Optional[BaseModel] = None
        self.msg_content: Optional[str] = None
        self.last_chunk: Optional[ModelResponseChunkType] = None
        self.fn_calls: List[FunctionCall] = []
        self.fn_api_id_to_fn_call: Dict[str, FunctionCall] = {}

    @property
    @abstractmethod
    def model_provider(self) -> ModelProvider:
        raise NotImplementedError

    @abstractmethod
    def get_message(self) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def get_fn_calls(self) -> Iterator[FunctionCall]:
        raise NotImplementedError

    @abstractmethod
    def get_message_prop(self, prop_name: str) -> Any:
        raise NotImplementedError

    def get_or_stream_message(self) -> Union[str, None, Iterator[str]]:
        if self.is_stream:
            return self.stream_message()
        else:
            return self.get_message()

    def get_or_stream_fn_calls(self) -> Iterator[FunctionCall]:
        if self.is_stream:
            return self.stream_fn_calls()
        else:
            return self.get_fn_calls()

    @abstractmethod
    def is_message_start_chunk(self, chunk: ModelResponseChunkType) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_tool_start_chunk(self, chunk: ModelResponseChunkType) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_final_chunk(self, chunk: ModelResponseChunkType) -> bool:
        raise NotImplementedError

    @abstractmethod
    def has_function_call_id(self, chunk: ModelResponseChunkType) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_msg_from_chunk(self, chunk: ModelResponseChunkType) -> str:
        raise NotImplementedError

    @abstractmethod
    def has_msg_content(self, chunk: ModelResponseChunkType) -> bool:
        raise NotImplementedError

    @abstractmethod
    def has_fn_args_json(self, chunk: ModelResponseChunkType) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_fn_call_id_from_chunk(self, chunk: ModelResponseChunkType) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_fn_name_from_chunk(self, chunk: ModelResponseChunkType) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_fn_args_json_from_chunk(self, chunk: ModelResponseChunkType) -> str:
        raise NotImplementedError

    def stream_message(self) -> Optional[Iterator[str]]:
        chunk = self.get_next_usable_chunk()
        self.last_chunk = chunk
        if self.is_message_start_chunk(chunk):
            return self._stream_message(chunk)
        else:
            return None

    def _stream_message(self, chunk: ModelResponseChunkType) -> Iterator[str]:
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

    def stream_fn_calls(self) -> Iterator[FunctionCall]:
        self.last_chunk = self.get_next_usable_chunk()
        function_name = None
        tool_call_id = None
        function_args_json = ""

        for chunk in itertools.chain([self.last_chunk], self.output_obj):
            if self.has_function_call_id(chunk):
                if tool_call_id is not None:
                    fn_call = FunctionCall.create(
                        name=function_name,
                        api_id=tool_call_id,
                        api_id_model_provider=self.model_provider,
                        args_json=function_args_json,
                    )
                    self.fn_calls.append(fn_call)
                    yield fn_call

                function_name = self.get_fn_name_from_chunk(chunk)
                tool_call_id = self.get_fn_call_id_from_chunk(chunk)
                function_args_json = ""
            elif tool_call_id is not None and self.has_fn_args_json(chunk):
                function_args_json += self.get_fn_args_json_from_chunk(chunk)

        if tool_call_id is not None:
            assert function_name is not None
            fn_call = FunctionCall.create(
                name=function_name,
                api_id_model_provider=self.model_provider,
                api_id=tool_call_id,
                args_json=function_args_json,
            )

            self.fn_calls.append(fn_call)
            yield fn_call

    def get_next_usable_chunk(self) -> ModelResponseChunkType:
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


class OAIModelOutput(ModelOutput[ChatCompletionChunk]):
    model_provider = ModelProvider.OPENAI

    def _get_tool_call(self, chunk: ChatCompletionChunk) -> ChoiceDeltaToolCall:
        return chunk.choices[0].delta.tool_calls[0]  # type: ignore

    def get_fn_call_id_from_chunk(self, chunk: ChatCompletionChunk) -> str:
        return self._get_tool_call(chunk).id  # type: ignore

    def get_fn_name_from_chunk(self, chunk: ChatCompletionChunk) -> str:
        return self._get_tool_call(chunk).function.name  # type: ignore

    def get_fn_args_json_from_chunk(self, chunk: ChatCompletionChunk) -> str:
        return self._get_tool_call(chunk).function.arguments  # type: ignore

    def _has_tool_call(self, chunk: ChatCompletionChunk) -> bool:
        return bool(chunk.choices[0].delta.tool_calls)

    def has_function_call_id(self, chunk: ChatCompletionChunk) -> bool:
        return self._has_tool_call(chunk) and self._get_tool_call(chunk).id is not None

    def is_tool_start_chunk(self, chunk: ChatCompletionChunk) -> bool:
        return self.has_function_call_id(chunk)

    def has_fn_args_json(self, chunk: ChatCompletionChunk) -> bool:
        return (
            self._has_tool_call(chunk)
            and self._get_tool_call(chunk).function.arguments is not None  # type: ignore
        )

    def is_message_start_chunk(self, chunk: ChatCompletionChunk) -> bool:
        return self.has_msg_content(chunk)

    def has_msg_content(self, chunk: ChatCompletionChunk) -> bool:
        return chunk.choices[0].delta.content is not None

    def get_msg_from_chunk(self, chunk: ChatCompletionChunk) -> str:
        return chunk.choices[0].delta.content  # type: ignore

    def is_final_chunk(self, chunk: ChatCompletionChunk) -> bool:
        return chunk.choices[0].finish_reason is not None

    def get_message(self) -> Optional[str]:
        self.msg_content = self.output_obj.choices[0].message.content
        return self.msg_content

    def get_message_prop(self, prop_name: str) -> Any:
        if self.parsed_msg is None:
            self.parsed_msg = self.output_obj.choices[0].message.parsed
        return getattr(self.parsed_msg, prop_name)

    def get_logprob(self, token_idx: int) -> float:
        return self.output_obj.choices[0].logprobs.content[token_idx].logprob

    def get_prob(self, token_idx: int) -> float:
        return np.exp(self.get_logprob(token_idx))

    def get_fn_calls(self) -> Iterator[FunctionCall]:
        tool_calls = self.output_obj.choices[0].message.tool_calls or []
        for tool_call in tool_calls:
            if tool_call.id not in self.fn_api_id_to_fn_call:
                fn_call = FunctionCall.create(
                    name=tool_call.function.name,
                    api_id_model_provider=self.model_provider,
                    api_id=tool_call.id,
                    args_json=tool_call.function.arguments,
                )
                self.fn_calls.append(fn_call)
                self.fn_api_id_to_fn_call[tool_call.id] = fn_call
            else:
                fn_call = self.fn_api_id_to_fn_call[tool_call.id]
            yield fn_call


class AnthropicModelOutput(ModelOutput[RawMessageStreamEvent]):
    model_provider = ModelProvider.ANTHROPIC

    def get_fn_call_id_from_chunk(self, chunk: RawMessageStreamEvent) -> str:
        return chunk.content_block.id  # type: ignore

    def get_fn_name_from_chunk(self, chunk: RawMessageStreamEvent) -> str:
        return chunk.content_block.name  # type: ignore

    def get_fn_args_json_from_chunk(self, chunk: RawMessageStreamEvent) -> str:
        return chunk.delta.partial_json  # type: ignore

    def has_fn_args_json(self, chunk: RawMessageStreamEvent) -> bool:
        return self._is_delta_chunk(chunk) and hasattr(chunk.delta, "partial_json")  # type: ignore

    def _is_content_block(self, chunk: RawMessageStreamEvent) -> bool:
        return hasattr(chunk, "content_block")  # type: ignore

    def _is_delta_chunk(self, chunk: RawMessageStreamEvent) -> bool:
        return hasattr(chunk, "delta")

    def is_message_start_chunk(self, chunk: RawMessageStreamEvent) -> bool:
        return self._is_content_block(chunk) and chunk.content_block.type == "text"  # type: ignore

    def is_tool_start_chunk(self, chunk: RawMessageStreamEvent) -> bool:
        return self._is_content_block(chunk) and chunk.content_block.type == "tool_use"  # type: ignore

    def is_end_block_chunk(self, chunk: RawMessageStreamEvent) -> bool:
        return chunk.type == "content_block_stop"

    def is_final_chunk(self, chunk: RawMessageStreamEvent) -> bool:
        return chunk.type == "message_stop"

    def has_msg_content(self, chunk: RawMessageStreamEvent) -> bool:
        return (
            self._is_delta_chunk(chunk)
            and getattr(chunk.delta, "text", None) is not None  # type: ignore
        )

    def has_function_call_id(self, chunk: RawMessageStreamEvent) -> bool:
        return self._is_content_block(chunk) and hasattr(chunk.content_block, "id")  # type: ignore

    def get_msg_from_chunk(self, chunk: RawMessageStreamEvent) -> str:
        return chunk.delta.text  # type: ignore

    def get_message(self) -> Optional[str]:
        content = self.output_obj.content[0]
        if content.type == "text":
            self.msg_content = content.text
            return self.msg_content
        else:
            return None

    def get_message_prop(self, prop_name: str) -> Any:
        if self.response_format is None:
            raise ValueError()

        if self.parsed_msg is None:
            fn_call = next(self.get_fn_calls())
            self.parsed_msg = self.response_format(**fn_call.args)
        return getattr(self.parsed_msg, prop_name)

    def get_fn_calls(self) -> Iterator[FunctionCall]:
        for content in self.output_obj.content:
            if content.type == "tool_use":
                if content.id not in self.fn_api_id_to_fn_call:
                    fn_call = FunctionCall.create(
                        name=content.name,
                        api_id_model_provider=self.model_provider,
                        api_id=content.id,
                        args=content.input,
                    )
                    self.fn_calls.append(fn_call)
                    self.fn_api_id_to_fn_call[content.id] = fn_call
                else:
                    fn_call = self.fn_api_id_to_fn_call[content.id]
                yield fn_call
