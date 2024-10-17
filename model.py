import itertools
from enum import StrEnum

import anthropic
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

        return ModelCompletion(chat_fn(**args), stream, ModelProvider.OPENAI)

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


class ModelCompletion:
    def __init__(self, completion_obj, is_stream, model_provider):
        self.completion_obj = completion_obj
        self.is_stream = is_stream
        self.model_provider = model_provider
        self.full_msg = None
        self.current_chunk = None

    def _has_function_call_id(self, chunk):
        return (
            chunk.choices[0].delta.tool_calls is not None
            and chunk.choices[0].delta.tool_calls[0].id is not None
        )

    def _has_msg_content(self, chunk):
        return chunk.choices[0].delta.content is not None

    def _get_first_usable_chunk(self):
        chunk = next(self.completion_obj)
        while not (self._has_function_call_id(chunk) or self._has_msg_content(chunk)):
            chunk = next(self.completion_obj)
        return chunk

    def is_tool_call(self):
        if self.is_stream:
            first_chunk = self._get_first_usable_chunk()
            self.current_chunk = first_chunk
            return self._has_function_call_id(first_chunk)

    def iter_messages(self):
        self.full_msg = ""
        try:
            while True:
                chunk = next(self.completion_obj)  # Get the next chunk
                msg = chunk.choices[0].delta.content
                finish_reason = chunk.choices[0].finish_reason
                if finish_reason is not None:
                    raise StopIteration
                if msg is None:
                    logger.warning(f"msg is None with chunk {chunk}")
                    raise StopIteration
                self.full_msg += msg  # Append the message to full_msg
                yield msg  # Return the message
        except StopIteration:
            pass  # Signal end of iteration

    def extract_fns_from_chat_stream(self):
        function_calls = []

        for chunk in itertools.chain([self.current_chunk], self.completion_obj):
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
        return function_calls
