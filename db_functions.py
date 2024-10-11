import inspect
import os
import re
from collections import defaultdict
from functools import wraps
from typing import Dict, List, Optional, get_args, get_origin

from openai import pydantic_function_tool
from pydantic import BaseModel, Field, create_model
from supabase import Client
from supabase import create_client as create_supabase_client

supabase: Client = None

OPENAI_TOOL_NAME_TO_TOOL_DEF = {}
FN_NAME_TO_FN = {}
OPENAI_TOOLS_RETUN_DESCRIPTION = {}


def create_client():
    global supabase
    supabase = create_supabase_client(
        os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")
    )

def obj_to_dict(obj):
    obj_type = type(obj)
    if issubclass(obj_type, BaseModel):
        return obj.model_dump()
    elif isinstance(obj, (dict, defaultdict)):
        return {obj_to_dict(k): obj_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [obj_to_dict(item) for item in obj]
    else:
        return obj  # Default to "object" if type not found

def get_description_from_docstring(docstring):
    if "Args:" in docstring:
        description = docstring.split("Args:")[0].strip()
    else:
        description = docstring.strip()
    return description

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
        
    return {
            k: tuple(v) for k, v in field_name_to_field.items()
        }

def get_return_description_from_docstring(docstring):
    return_description = ""
    returns_pattern = re.compile(r"Returns:\n(.*)", re.DOTALL)
    returns_match = returns_pattern.search(docstring)
    if returns_match:
        return_description = returns_match.group(1).strip()
    return return_description

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

            # Iterate over the parameters and check for dict-to-BaseModel conversion
            for param_name, param_value in bound_args.arguments.items():
                param = fn_signature.parameters[param_name]
                # Check if the type annotation is a subclass of pydantic.BaseModel
                if isinstance(param.annotation, type):
                    # Case 1: Single BaseModel
                    if issubclass(param.annotation, BaseModel) and isinstance(
                        param_value, dict
                    ):
                        # Convert the dict to a BaseModel instance
                        bound_args.arguments[param_name] = param.annotation(
                            **param_value
                        )

                # Case 2: List[BaseModel]
                elif get_origin(param.annotation) is list:
                    list_type = get_args(param.annotation)[0]
                    if issubclass(list_type, BaseModel) and isinstance(
                        param_value, list
                    ):
                        # Convert each dict in the list to a BaseModel instance
                        bound_args.arguments[param_name] = [
                            list_type(**item) if isinstance(item, dict) else item
                            for item in param_value
                        ]

            # Call the original function with the modified arguments
            return func(*bound_args.args, **bound_args.kwargs)

        global FN_NAME_TO_FN
        FN_NAME_TO_FN[func.__name__] = wrapper

        return wrapper

    return decorator_fn


class Option(BaseModel):
    name: str
    type: str
    value_type: str
    num_unit: Optional[str]
    default_value: int | bool | str
    str_option_values: Optional[List[str]] = None


@openai_tool_decorator(
    "Most customers either don't provide a complete order (i.e. not specifying required options like size)"
    "or are not aware of all the options available for a menu item. It is your job to help them with both cases."
)
def get_menu_items_options(menu_item_id: int) -> Dict[str, List[Option]]:
    """
    Get all the options available for the menu item.

    Args:
        menu_item_id: The menu item id.

    Returns:
        A mapping of cup size to the list of options available for that size.
    """

    response = (
        supabase.table("menu_item_to_options_map")
        .select(
            "*, option(name, type, value_type, num_unit), option_type_config(default_num_value, default_bool_value, default_discrete_value_id, discrete_option_value(name)), menu_item_to_option_values_map(discrete_option_value(name))"
        )
        .eq("menu_item_id", menu_item_id)
        .execute()
    )
    data = response.data

    size_to_options_map = defaultdict(list)
    for item in data:
        option = item["option"]
        option_type_config = item["option_type_config"]
        str_option_values_map = item["menu_item_to_option_values_map"]
        option = Option(
            name=option["name"],
            type=option["type"],
            value_type=option["value_type"],
            num_unit=option["num_unit"],
            default_value=(
                option_type_config["default_num_value"]
                or option_type_config["default_bool_value"]
                or (
                    option_type_config["discrete_option_value"]["name"]
                    if option_type_config["discrete_option_value"] is not None
                    else None
                )
            ),
            str_option_values=(
                [
                    map_obj["discrete_option_value"]["name"]
                    for map_obj in str_option_values_map
                ]
                if str_option_values_map
                else None
            ),
        )

        size_to_options_map[item["cup_size"]].append(option)

    return size_to_options_map


class MenuItem(BaseModel):
    id: int = Field(description="db id")
    name: str
    description: str
    group: str = Field(description="the menu category it belongs to")


@openai_tool_decorator()
def get_menu_item_from_name(menu_item_name: str) -> MenuItem:
    """
    Get the menu item given the name of the menu item.

    Args:
        menu_item_name: The menu item name.

    Returns:
        A MenuItem object.
    """
    formatted_menu_item_name = menu_item_name.replace(" ", "&")
    response = (
        supabase.table("menu_item")
        .select("id, name, description, group")
        .text_search("name", formatted_menu_item_name)
        .execute()
    )
    item = response.data[0]
    return MenuItem(
        id=item["id"],
        name=item["name"],
        description=item["description"],
        group=item["group"],
    )


class OptionOrder(BaseModel):
    name: str
    value: str | int | bool


class ItemOrder(BaseModel):
    name: str
    options: List[OptionOrder]


class Order(BaseModel):
    item_orders: List[ItemOrder] = []


order = Order()


# @openai_tool_decorator()
# def get_current_order() -> Order:
#     """
#     Get the current order.

#     Returns:
#         The current order.
#     """
#     return order

# @openai_tool_decorator(
#     "As soon as all the required options have been provided for a single item, add it to the order."
# )
# def add_to_order(item_order: ItemOrder) -> None:
#     """
#     Add an item order to the current order.

#     Args:
#         item_order: the ItemOrder to add.

#     Returns:
#         None
#     """
#     global order
#     order.item_orders.append(item_order)


# @openai_tool_decorator()
# def upsert_to_order(item_name: str, new_options: List[OptionOrder]) -> None:
#     """
#     Update the options of an item order in the current order.

#     Args:
#         item_name: the name of the item to update.
#         new_options: the new options to set.

#     Returns:
#         None
#     """
#     global order
#     item_order = next(
#         item_order for item_order in order.item_orders if item_order.name == item_name
#     )
#     item_order.options = new_options


# @openai_tool_decorator()
# def remove_from_order(item_name: str) -> None:
#     """
#     Remove an item order from the current order.

#     Args:
#         item_name: the name of the item to remove.

#     Returns:
#         None
#     """
#     global order
#     item_order = next(
#         item_order for item_order in order.item_orders if item_order.name == item_name
#     )
#     order.item_orders.remove(item_order)
