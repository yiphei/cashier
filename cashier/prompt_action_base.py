from __future__ import annotations

from cashier.model.model_completion import ModelName, Model
from abc import ABC, abstractmethod

class PromptActionBase(ABC):
    # TODO: fix these to raise NotImplemenetedError
    prompt = None 
    input_kwargs=None

    @classmethod
    @abstractmethod
    def get_model_completion_args(cls, model_provider, input):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_output(cls, model_provider,chat_completion, input):
        raise NotImplementedError

    @classmethod
    def run(cls, model_name: ModelName, **kwargs):
        input = cls.input_kwargs(**kwargs)
        model_provider = Model.get_model_provider(model_name)
        args = cls.get_model_completion_args(model_provider, input)

        if "response_format" not in args and cls.prompt.response_format is not None:
            args["response_format"] = cls.prompt.response_format

        chat_completion = Model.chat(
            model_name=model_name,
            **args
        )
        return cls.get_output(model_provider, chat_completion, input)