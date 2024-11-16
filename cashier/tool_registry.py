from __future__ import annotations

import copy
import inspect
import re
from functools import wraps
from inspect import Signature
from types import FunctionType
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import pydantic_function_tool
from pydantic import Field, create_model
from pydantic.fields import FieldInfo

from cashier.model_util import ModelProvider


# got this from: https://stackoverflow.com/questions/28237955/same-name-for-classmethod-and-instancemethod
class class_or_instance_method(classmethod):
    def __get__(self, instance: Any, type_: Any) -> Any:  # type: ignore
        descr_get = super().__get__ if instance is None else self.__func__.__get__
        return descr_get(instance, type_)


def get_return_description_from_docstring(docstring: str) -> str:
    return_description = ""
    returns_pattern = re.compile(r"Returns:\n(.*)", re.DOTALL)
    returns_match = returns_pattern.search(docstring)
    if returns_match:
        return_description = returns_match.group(1).strip()
    return return_description


def get_field_map_from_docstring(
    docstring: str, func_signature: Signature
) -> Dict[str, Tuple[Any, FieldInfo]]:
    field_name_to_field_info: Dict[str, FieldInfo] = {}
    field_name_to_field_type_annotation: Dict[str, Any] = {}

    # Simplified regex pattern that captures everything between "Args:" and the first empty line
    args_regex_pattern = re.compile(r"Args:\n((?:(?!\n\s*\n).)*)", re.DOTALL)

    # Find args section
    args_match = args_regex_pattern.search(docstring)
    if args_match:
        args_section = args_match.group(1).strip()
        for line in args_section.splitlines():
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            # Split by the first colon to separate the argument name from its description
            parts = line.split(":", 1)
            if len(parts) == 2:
                arg_name, arg_description = parts
                field_name_to_field_info[arg_name.strip()] = Field(
                    description=arg_description.strip()
                )

    for param_name, param in func_signature.parameters.items():
        if param_name in field_name_to_field_info:
            # Update type annotations if available
            if param.annotation == inspect.Parameter.empty:
                raise Exception("Type annotation is required for all parameters")
            field_name_to_field_type_annotation[param_name] = param.annotation
        else:
            raise Exception(f"Parameter {param_name} is not found in the docstring")

    assert field_name_to_field_info.keys() == field_name_to_field_type_annotation.keys()

    return {
        k: (field_name_to_field_type_annotation[k], field_name_to_field_info[k])
        for k in field_name_to_field_type_annotation.keys()
    }


def get_description_from_docstring(docstring: str) -> str:
    if "Args:" in docstring:
        description = docstring.split("Args:")[0].strip()
    else:
        description = docstring.strip()
    return description


def get_anthropic_tool_def_from_oai(oai_tool_def: Dict) -> Dict:
    anthropic_tool_def_body = copy.deepcopy(oai_tool_def["function"]["parameters"])
    return {
        "name": oai_tool_def["function"]["name"],
        "description": oai_tool_def["function"]["description"],
        "input_schema": anthropic_tool_def_body,
    }


class ToolRegistry:
    GLOBAL_OPENAI_TOOL_NAME_TO_TOOL_DEF: Dict[str, Dict] = {}
    GLOBAL_ANTHROPIC_TOOL_NAME_TO_TOOL_DEF: Dict[str, Dict] = {}
    GLOBAL_FN_NAME_TO_FN: Dict[str, Callable] = {}
    GLOBAL_OPENAI_TOOLS_RETURN_DESCRIPTION: Dict[str, Dict] = {}

    def __init_subclass__(cls):
        super().__init_subclass__()
        for base in cls.__bases__:
            for key, value in base.__dict__.items():
                if not key.startswith("__") and not isinstance(
                    value, (FunctionType, classmethod, staticmethod, property)
                ):
                    setattr(cls, key, copy.deepcopy(value))

    def __init__(self, oai_tool_defs: Optional[List[Dict]] = None):
        self.openai_tool_name_to_tool_def = copy.copy(
            self.GLOBAL_OPENAI_TOOL_NAME_TO_TOOL_DEF
        )
        self.anthropic_tool_name_to_tool_def = copy.copy(
            self.GLOBAL_ANTHROPIC_TOOL_NAME_TO_TOOL_DEF
        )
        self.fn_name_to_fn = copy.copy(self.GLOBAL_FN_NAME_TO_FN)
        self.openai_tools_return_description = copy.copy(
            self.GLOBAL_OPENAI_TOOLS_RETURN_DESCRIPTION
        )
        self.model_provider_to_tool_def = {
            ModelProvider.OPENAI: self.openai_tool_name_to_tool_def,
            ModelProvider.ANTHROPIC: self.anthropic_tool_name_to_tool_def,
        }
        if oai_tool_defs:
            for tool_def in oai_tool_defs:
                tool_name = tool_def["function"]["name"]
                self.add_tool_def_w_oai_def(tool_name, tool_def)

    @property
    def tool_names(self) -> List[str]:
        return list(self.openai_tool_name_to_tool_def.keys())

    @classmethod
    def create_from_tool_registry(
        cls, tool_registry: ToolRegistry, tool_names: Optional[List[str]] = None
    ) -> ToolRegistry:
        if tool_names is None:
            return copy.deepcopy(tool_registry)
        else:
            new_tool_registry = cls()
            for tool_name in tool_names:
                new_tool_registry.openai_tool_name_to_tool_def[tool_name] = (
                    tool_registry.openai_tool_name_to_tool_def[tool_name]
                )
                new_tool_registry.anthropic_tool_name_to_tool_def[tool_name] = (
                    tool_registry.anthropic_tool_name_to_tool_def[tool_name]
                )
                if tool_name in tool_registry.openai_tools_return_description:
                    new_tool_registry.openai_tools_return_description[tool_name] = (
                        tool_registry.openai_tools_return_description[tool_name]
                    )

                if tool_name in tool_registry.fn_name_to_fn:
                    new_tool_registry.fn_name_to_fn[tool_name] = (
                        tool_registry.fn_name_to_fn[tool_name]
                    )

            return new_tool_registry

    def add_tool_def(self, tool_name: str, description: str, field_args: Any) -> None:
        fn_pydantic_model = create_model(tool_name, **field_args)
        fn_json_schema = pydantic_function_tool(
            fn_pydantic_model,
            name=tool_name,
            description=description,
        )
        remove_default(fn_json_schema)
        self.add_tool_def_w_oai_def(tool_name, fn_json_schema)

    def get_tool_defs(
        self,
        tool_names: Optional[List[str]] = None,
        model_provider: ModelProvider = ModelProvider.OPENAI,
    ) -> List[Dict]:
        if tool_names:
            return [
                self.model_provider_to_tool_def[model_provider][tool_name]
                for tool_name in tool_names
            ]
        else:
            return list(self.model_provider_to_tool_def[model_provider].values())

    def add_tool_def_w_oai_def(self, tool_name: str, oai_tool_def: Dict) -> None:
        self.openai_tool_name_to_tool_def[tool_name] = oai_tool_def
        self.anthropic_tool_name_to_tool_def[tool_name] = (
            get_anthropic_tool_def_from_oai(oai_tool_def)
        )

    @classmethod
    def _add_tool_def_w_oai_def_cls(cls, tool_name: str, oai_tool_def: Dict) -> None:
        cls.GLOBAL_OPENAI_TOOL_NAME_TO_TOOL_DEF[tool_name] = oai_tool_def
        cls.GLOBAL_ANTHROPIC_TOOL_NAME_TO_TOOL_DEF[tool_name] = (
            get_anthropic_tool_def_from_oai(oai_tool_def)
        )

    @class_or_instance_method
    def model_tool_decorator(
        self_or_cls, tool_instructions: Optional[str] = None
    ) -> Callable:
        is_class = isinstance(self_or_cls, type)
        if is_class:
            fn_name_to_fn_attr = self_or_cls.GLOBAL_FN_NAME_TO_FN
            oai_tools_return_map_attr = (
                self_or_cls.GLOBAL_OPENAI_TOOLS_RETURN_DESCRIPTION
            )
        else:
            fn_name_to_fn_attr = self_or_cls.fn_name_to_fn
            oai_tools_return_map_attr = self_or_cls.openai_tools_return_description

        def decorator_fn(func: Callable):
            docstring = inspect.getdoc(func)
            fn_signature = inspect.signature(func)

            # Generate function args schemas
            description = get_description_from_docstring(docstring)
            if tool_instructions is not None:
                if description[-1] != ".":
                    description += "."
                description += " " + tool_instructions.strip()

            field_map = get_field_map_from_docstring(docstring, fn_signature)
            fn_signature_pydantic_model = create_model(
                func.__name__ + "_parameters", **field_map
            )
            func.pydantic_model = fn_signature_pydantic_model
            oai_tool_def = pydantic_function_tool(
                fn_signature_pydantic_model, name=func.__name__, description=description
            )

            if is_class:
                self_or_cls._add_tool_def_w_oai_def_cls(func.__name__, oai_tool_def)
            else:
                self_or_cls.add_tool_def_w_oai_def(func.__name__, oai_tool_def)

            # Generate function return type schema
            return_description = get_return_description_from_docstring(docstring)
            return_annotation = fn_signature.return_annotation
            if return_annotation == inspect.Signature.empty:
                raise Exception("Type annotation is required for return type")
            if return_annotation is None:
                return_annotation = type(None)
            fn_return_type_model = create_model(
                func.__name__ + "_return",
                return_obj=(return_annotation, Field(description=return_description)),
            )
            return_type_json_schema = fn_return_type_model.model_json_schema()

            # Remove extra fields
            actual_return_json_schema = return_type_json_schema["properties"][
                "return_obj"
            ]
            if "$defs" in return_type_json_schema:
                actual_return_json_schema["$defs"] = return_type_json_schema["$defs"]
            if "title" in actual_return_json_schema:
                actual_return_json_schema.pop("title")

            oai_tools_return_map_attr[func.__name__] = actual_return_json_schema

            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                bound_args = fn_signature.bind(*args, **kwargs)
                bound_args.apply_defaults()

                pydantic_obj = func.pydantic_model(**bound_args.arguments)
                for field_name in pydantic_obj.model_fields.keys():
                    bound_args.arguments[field_name] = getattr(pydantic_obj, field_name)

                # Call the original function with the modified arguments
                return func(*bound_args.args, **bound_args.kwargs)

            fn_name_to_fn_attr[func.__name__] = wrapper

            return wrapper

        return decorator_fn


def remove_default(schema: Dict) -> None:
    found_key = False
    for key, value in schema.items():
        if key == "default":
            found_key = True
        elif isinstance(value, dict):
            remove_default(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    remove_default(item)
    if found_key:
        schema.pop("default")
