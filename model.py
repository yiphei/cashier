import itertools
from enum import StrEnum

import anthropic
import numpy as np
from openai import OpenAI
from pydantic import BaseModel

from model_tool_decorator import (
    ANTHROPIC_TOOL_NAME_TO_TOOL_DEF,
    OPENAI_TOOL_NAME_TO_TOOL_DEF,
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
        messages,
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
        if model_provider == ModelProvider.OPENAI:
            if tool_names:
                tools = self.get_tool_defs_from_names(
                    tool_names, OPENAI_TOOL_NAME_TO_TOOL_DEF, extra_oai_tool_defs
                )

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

            return self.ant_chat(model_name, messages, tools, stream, **kwargs)

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
        tools=None,
        stream=False,
        **kwargs,
    ):
        # TODO: remove adhoc code
        args = {
            "max_tokens": 8192,
            "model": model_name,
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
