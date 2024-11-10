import copy
import inspect
import re
from collections import defaultdict
from functools import wraps

from openai import pydantic_function_tool
from pydantic import Field, create_model

from cashier.model_util import ModelProvider


def get_return_description_from_docstring(docstring):
    return_description = ""
    returns_pattern = re.compile(r"Returns:\n(.*)", re.DOTALL)
    returns_match = returns_pattern.search(docstring)
    if returns_match:
        return_description = returns_match.group(1).strip()
    return return_description


def get_field_map_from_docstring(docstring, func_signature):
    field_name_to_field = defaultdict(lambda: [None, Field()])

    # Regex patterns to capture Args and Returns sections
    args_regex_pattern = re.compile(r"Args:\n(.*?)\n\n", re.DOTALL)

    # Find args section
    args_match = args_regex_pattern.search(docstring)
    if args_match:
        args_section = args_match.group(1).strip()
        for line in args_section.splitlines():
            # Split by the first colon to separate the argument name from its description
            arg_name, arg_description = line.split(":", 1)
            field_name_to_field[arg_name.strip()][
                1
            ].description = arg_description.strip()

    for param_name, param in func_signature.parameters.items():
        if param_name in field_name_to_field:
            # Update type annotations if available
            if param.annotation == inspect.Parameter.empty:
                raise Exception("Type annotation is required for all parameters")
            field_name_to_field[param_name][0] = param.annotation
        else:
            raise Exception(f"Parameter {param_name} is not found in the docstring")

    return {k: tuple(v) for k, v in field_name_to_field.items()}


def get_description_from_docstring(docstring):
    if "Args:" in docstring:
        description = docstring.split("Args:")[0].strip()
    else:
        description = docstring.strip()
    return description


def get_anthropic_tool_def_from_oai(oai_tool_def):
    anthropic_tool_def_body = copy.deepcopy(oai_tool_def["function"]["parameters"])
    return {
        "name": oai_tool_def["function"]["name"],
        "description": oai_tool_def["function"]["description"],
        "input_schema": anthropic_tool_def_body,
    }


class ToolRegistry:
    def __init__(self, oai_tool_defs_map=None):
        self.openai_tool_name_to_tool_def = {}
        self.anthropic_tool_name_to_tool_def = {}
        self.fn_name_to_fn = {}
        self.openai_tools_return_description = {}
        self.model_provider_to_tool_def = {
            ModelProvider.OPENAI: self.openai_tool_name_to_tool_def,
            ModelProvider.ANTHROPIC: self.anthropic_tool_name_to_tool_def,
        }
        if oai_tool_defs_map:
            for tool_name, tool_def in oai_tool_defs_map.items():
                self.openai_tool_name_to_tool_def[tool_name] = tool_def
                self.anthropic_tool_name_to_tool_def[tool_name] = tool_def

    @property
    def tool_names(self):
        return list(self.openai_tool_name_to_tool_def.keys())

    @classmethod
    def create_from_tool_registry(cls, tool_registry, tool_names=None):
        if tool_names is None:
            return copy.deepcopy(tool_registry)
        else:
            new_tool_registry = ToolRegistry()
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

    def add_tool_def(self, tool_name, description, field_args):
        fn_pydantic_model = create_model(tool_name, **field_args)
        fn_json_schema = pydantic_function_tool(
            fn_pydantic_model,
            name=tool_name,
            description=description,
        )
        remove_default(fn_json_schema)
        self.add_tool_def_w_oai_def(tool_name, fn_json_schema)

    def get_tool_defs(self, tool_names=None, model_provider=ModelProvider.OPENAI):
        if tool_names:
            return [
                self.model_provider_to_tool_def[model_provider][tool_name]
                for tool_name in tool_names
            ]
        else:
            return list(self.model_provider_to_tool_def[model_provider].values())

    def add_tool_def_w_oai_def(self, tool_name, oai_tool_def):
        self.openai_tool_name_to_tool_def[tool_name] = oai_tool_def
        self.anthropic_tool_name_to_tool_def[tool_name] = (
            get_anthropic_tool_def_from_oai(oai_tool_def)
        )

    def model_tool_decorator(self, tool_instructions=None):
        def decorator_fn(func):
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

            self.add_tool_def_w_oai_def(func.__name__, oai_tool_def)

            # Generate function return type schema
            return_description = get_return_description_from_docstring(docstring)
            return_annotation = fn_signature.return_annotation
            if return_annotation == inspect.Signature.empty:
                raise Exception("Type annotation is required for return type")
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

            self.openai_tools_return_description[func.__name__] = (
                actual_return_json_schema
            )

            @wraps(func)
            def wrapper(*args, **kwargs):
                bound_args = fn_signature.bind(*args, **kwargs)
                bound_args.apply_defaults()

                pydantic_obj = func.pydantic_model(**bound_args.arguments)
                for field_name in pydantic_obj.model_fields.keys():
                    bound_args.arguments[field_name] = getattr(pydantic_obj, field_name)

                # Call the original function with the modified arguments
                return func(*bound_args.args, **bound_args.kwargs)

            self.fn_name_to_fn[func.__name__] = wrapper

            return wrapper

        return decorator_fn


def remove_default(schema):
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
