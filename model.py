import itertools
from enum import StrEnum

import anthropic
import numpy as np
from openai import OpenAI
from pydantic import BaseModel

from logger import logger


class ModelProvider(StrEnum):
    OPENAI = "OPENAI"
    ANTHROPIC = "ANTHROPIC"


class Model:
    model_name_to_provider = {
        "gpt-4o-mini": ModelProvider.OPENAI,
        "gpt-4o": ModelProvider.OPENAI,
        "claude-3-5-sonnet-20240620": ModelProvider.ANTHROPIC,
    }

    def __init__(self):
        self.oai_client = OpenAI()
        self.anthropic_client = anthropic.Anthropic()

    def chat(
        self,
        model_name,
        messages,
        tools=None,
        stream=False,
        logprobs=False,
        response_format=None,
        **kwargs,
    ):
        model_provider = self.model_name_to_provider[model_name]
        if model_provider == ModelProvider.OPENAI:
            return self.oai_chat(
                model_name, messages, tools, stream, logprobs, response_format, **kwargs
            )
        elif model_provider == ModelProvider.ANTHROPIC:
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

        return ModelOutput(chat_fn(**args), stream, ModelProvider.OPENAI)

    def ant_chat(
        self,
        model_name,
        messages,
        tools=None,
        stream=False,
        **kwargs,
    ):
        args = {
            "max_tokens": 8192,
            "model": model_name,
            "messages": messages,
            "tools": tools,
            "stream": stream,
            **kwargs,
        }
        if not tools:
            args.pop("tools")

        return self.anthropic_client.messages.create(**args)


class FunctionCall(BaseModel):
    function_name: str
    tool_call_id: str
    function_args_json: str


class ModelOutput:
    def __init__(self, output_obj, is_stream, model_provider):
        self.output_obj = output_obj
        self.is_stream = is_stream
        self.model_provider = model_provider
        self.msg_content = None
        self.current_chunk = None
        self._has_tool_call = None

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

    def has_tool_call(self):
        if self._has_tool_call is not None:
            return self._has_tool_call

        if self.is_stream:
            first_chunk = self._get_first_usable_chunk()
            self.current_chunk = first_chunk
            self._has_tool_call = self._has_function_call_id(first_chunk)
        else:
            self._has_tool_call = bool(self.output_obj.choices[0].message.tool_calls)

        return self._has_tool_call

    def stream_message(self):
        self.msg_content = ""
        try:
            while True:
                chunk = next(self.output_obj)  # Get the next chunk
                msg = chunk.choices[0].delta.content
                finish_reason = chunk.choices[0].finish_reason
                if finish_reason is not None:
                    raise StopIteration
                if msg is None:
                    logger.warning(f"msg is None with chunk {chunk}")
                    raise StopIteration
                self.msg_content += msg  # Append the message to full_msg
                yield msg  # Return the message
        except StopIteration:
            pass  # Signal end of iteration

    def get_message(self):
        self.msg_content = self.output_obj.choices[0].message.content
        return self.msg_content
    
    def get_or_stream_message(self):
        if self.is_stream:
            return self.stream_message()
        else:
            return self.get_message()

    def get_message_prop(self, prop_name):
        return getattr(self.output_obj.choices[0].message.parsed, prop_name)

    def get_logprob(self, token_idx):
        return self.output_obj.choices[0].logprobs.content[token_idx].logprob

    def get_prob(self, token_idx):
        return np.exp(self.get_logprob(token_idx))

    def extract_fn_calls(self):
        function_calls = []
        if self.is_stream:
            for chunk in itertools.chain([self.current_chunk], self.output_obj):
                finish_reason = chunk.choices[0].finish_reason
                if finish_reason is not None:
                    break
                elif self._has_function_call_id(chunk):
                    if self.current_chunk != chunk:
                        function_calls.append(
                            FunctionCall(
                                function_name=function_name,  # noqa
                                tool_call_id=tool_call_id,  # noqa
                                function_args_json=function_args_json,  # noqa
                            )
                        )

                    function_name = chunk.choices[0].delta.tool_calls[0].function.name
                    tool_call_id = chunk.choices[0].delta.tool_calls[0].id
                    function_args_json = ""
                else:
                    function_args_json += (
                        chunk.choices[0].delta.tool_calls[0].function.arguments
                    )

            function_calls.append(
                FunctionCall(
                    function_name=function_name,
                    tool_call_id=tool_call_id,
                    function_args_json=function_args_json,
                )
            )
        else:
            tool_calls = self.output_obj.choices[0].message.tool_calls
            for tool_call in tool_calls:
                function_calls.append(
                    FunctionCall(
                        function_name=tool_call.function.name,
                        tool_call_id=tool_call.id,
                        function_args_json=tool_call.function.arguments,
                    )
                )
        return function_calls
