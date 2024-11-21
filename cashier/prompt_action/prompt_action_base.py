from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Generic, Type, TypeVar

from pydantic import BaseModel

from cashier.model.model_completion import Model, ModelName, ModelOutput
from cashier.model.model_util import ModelProvider
from cashier.prompts.base_prompt import BasePrompt

T = TypeVar("T", bound=BaseModel)


class PromptActionBase(ABC, Generic[T]):
    prompt: ClassVar[Type[BasePrompt]]
    input_kwargs: ClassVar[Type[T]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "prompt") or cls.prompt is None:
            raise NotImplementedError("Subclasses must define 'prompt' class attribute")
        if not hasattr(cls, "input_kwargs") or cls.input_kwargs is None:
            raise NotImplementedError(
                "Subclasses must define 'input_kwargs' class attribute"
            )

    @classmethod
    @abstractmethod
    def get_model_completion_args(
        cls, model_provider: ModelProvider, input: T
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_output(
        cls, model_provider: ModelProvider, chat_completion: ModelOutput, input: T
    ) -> Any:
        raise NotImplementedError

    @classmethod
    def run(cls, model_name: ModelName, **kwargs: Any) -> Any:
        input = cls.input_kwargs(**kwargs)
        model_provider = Model.get_model_provider(model_name)
        args = cls.get_model_completion_args(model_provider, input)

        if "response_format" not in args and cls.prompt.response_format is not None:
            args["response_format"] = cls.prompt.response_format

        chat_completion = Model.chat(model_name=model_name, **args)
        return cls.get_output(model_provider, chat_completion, input)
