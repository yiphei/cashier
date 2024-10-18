import inspect
import re
from collections import defaultdict
from functools import wraps

from openai import pydantic_function_tool
from pydantic import Field, create_model


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


OPENAI_TOOL_NAME_TO_TOOL_DEF = {}
FN_NAME_TO_FN = {}
OPENAI_TOOLS_RETUN_DESCRIPTION = {}


def openai_tool_decorator(tool_instructions=None):
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
        global OPENAI_TOOL_NAME_TO_TOOL_DEF
        OPENAI_TOOL_NAME_TO_TOOL_DEF[func.__name__] = pydantic_function_tool(
            fn_signature_pydantic_model, name=func.__name__, description=description
        )

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
        actual_return_json_schema = return_type_json_schema["properties"]["return_obj"]
        if "$defs" in return_type_json_schema:
            actual_return_json_schema["$defs"] = return_type_json_schema["$defs"]
        if "title" in actual_return_json_schema:
            actual_return_json_schema.pop("title")

        global OPENAI_TOOLS_RETUN_DESCRIPTION
        OPENAI_TOOLS_RETUN_DESCRIPTION[func.__name__] = actual_return_json_schema

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = fn_signature.bind(*args, **kwargs)
            bound_args.apply_defaults()

            pydantic_obj = func.pydantic_model(**bound_args.arguments)
            for field_name in pydantic_obj.model_fields.keys():
                bound_args.arguments[field_name] = getattr(pydantic_obj, field_name)

            # Call the original function with the modified arguments
            return func(*bound_args.args, **bound_args.kwargs)

        global FN_NAME_TO_FN
        FN_NAME_TO_FN[func.__name__] = wrapper

        return wrapper

    return decorator_fn
