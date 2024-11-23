import inspect
from string import Formatter
from typing import Any, Callable, Dict, Optional, Set, Type

from pydantic import BaseModel, ConfigDict, Field, create_model

from cashier.model.model_completion import Model, ModelName, ModelOutput
from cashier.model.model_util import ModelProvider


class CallableMeta(type):
    def __call__(cls, strict_kwargs_check: bool = True, **kwargs: Any) -> str:
        instance = super().__call__(**kwargs)
        if not strict_kwargs_check:
            kwargs = {k: v for k, v in kwargs.items() if k in set(cls.prompt_kwargs.model_fields.keys())}  # type: ignore

        cls.prompt_kwargs.model_validate(kwargs)
        return (
            instance.f_string_prompt.format(**kwargs)
            if instance.f_string_prompt is not None
            else instance.dynamic_prompt(**kwargs)
        )


class BasePrompt(metaclass=CallableMeta):
    f_string_prompt: Optional[str] = None
    response_format: Optional[Type[BaseModel]] = None
    prompt_kwargs: Optional[Type[BaseModel]] = None
    run_input_kwargs: Optional[Type[BaseModel]] = None

    def __init__(self, **kwargs: Any):
        pass

    def dynamic_prompt(self, **kwargs: Any) -> Optional[str]:
        return None

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        has_fstring = cls.f_string_prompt is not None
        has_dynamic = cls.dynamic_prompt != BasePrompt.dynamic_prompt

        if not has_fstring and not has_dynamic:
            raise NotImplementedError(
                f"Class {cls.__name__} must override either f_string_prompt or dynamic_prompt"
            )
        if has_fstring and has_dynamic:
            raise ValueError(
                f"Class {cls.__name__} should not override both f_string_prompt and dynamic_prompt"
            )

        prompt_kwargs_fields = (
            BasePrompt.extract_fstring_args(cls.f_string_prompt)  # type: ignore
            if has_fstring
            else BasePrompt.extract_dynamic_args(cls.dynamic_prompt)
        )
        cls.prompt_kwargs = create_model(
            cls.__name__ + "_prompt_kwargs",
            __config__=ConfigDict(extra="forbid", arbitrary_types_allowed=True),
            **prompt_kwargs_fields,
        )
        if cls.run_input_kwargs is None:
            cls.run_input_kwargs = cls.prompt_kwargs

    @staticmethod
    def extract_fstring_args(f_string: str) -> Set[str]:
        """
        Extract argument names from an f-string format using string.Formatter.

        Args:
            f_string: A string with format placeholders like "hello {arg1}"

        Returns:
            Set of argument names
        """
        formatter = Formatter()
        # parse() returns tuples of (literal_text, field_name, format_spec, conversion)
        # We only need the field_name (second element)
        fields = {
            field_name: (str, Field())
            for _, field_name, _, _ in formatter.parse(f_string)
            if field_name is not None
        }
        return fields

    @staticmethod
    def extract_dynamic_args(dynamic_func: Callable) -> Set[str]:
        """
        Extract argument names from a dynamic_prompt method using inspection.

        Args:
            dynamic_func: The dynamic_prompt method

        Returns:
            Set of argument names
        """
        # Get the signature of the function
        sig = inspect.signature(dynamic_func)

        # Extract parameter names, excluding 'self'
        params = {
            param_name: (param.annotation, Field())
            for param_name, param in sig.parameters.items()
            if param_name != "self"
        }

        return params

    @classmethod
    def get_model_completion_args(
        cls, model_provider: ModelProvider, input: Any
    ) -> Dict[str, Any]:
        prompt = cls({field: getattr(input, field) for field in input.model_fields})
        return {"message_dicts": [{"role": "user", "content": prompt}]}

    @classmethod
    def get_output(
        cls, model_provider: ModelProvider, chat_completion: ModelOutput, input: Any
    ) -> Any:
        return chat_completion

    @classmethod
    def run(cls, model_name: ModelName, **kwargs: Any) -> Any:
        input = cls.run_input_kwargs(**kwargs)
        model_provider = Model.get_model_provider(model_name)
        args = cls.get_model_completion_args(model_provider, input)

        if "response_format" not in args and cls.response_format is not None:
            args["response_format"] = cls.response_format

        chat_completion = Model.chat(model_name=model_name, **args)
        return cls.get_output(model_provider, chat_completion, input)
