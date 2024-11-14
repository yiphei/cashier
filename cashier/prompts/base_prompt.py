import inspect
from string import Formatter
from typing import Callable, Set

from pydantic import BaseModel


class CallableMeta(type):
    def __call__(cls, strict_kwargs_check: bool = True, **kwargs) -> str:
        instance = super().__call__(**kwargs)
        if not strict_kwargs_check:
            kwargs = {k: v for k, v in kwargs.items() if k in cls.kwargs}

        return (
            instance.f_string_prompt.format(**kwargs)
            if instance.f_string_prompt is not None
            else instance.dynamic_prompt(**kwargs)
        )


class BasePrompt(metaclass=CallableMeta):
    f_string_prompt: str = None
    response_format: BaseModel = None
    kwargs: Set[str] = None

    def __init__(self, **kwargs):
        pass

    def dynamic_prompt(self, **kwargs) -> str:
        return None

    def __init_subclass__(cls, **kwargs):
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

        cls.kwargs = (
            BasePrompt.extract_fstring_args(cls.f_string_prompt)
            if has_fstring
            else BasePrompt.extract_dynamic_args(cls.dynamic_prompt)
        )

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
            field_name
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
            param_name
            for param_name, _ in sig.parameters.items()
            if param_name != "self"
        }

        return params
