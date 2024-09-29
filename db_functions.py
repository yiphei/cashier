import os
from supabase import create_client as create_supabase_client, Client
from dataclasses import dataclass
from collections import defaultdict
from typing import List
import re
import inspect

supabase: Client = None

OPENAI_TOOLS = []

def create_client():
    global supabase
    supabase = create_supabase_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))

@dataclass
class OptionValue:
    default_num_value: int
    default_bool_value: bool 
    discrete_discrete_value: str

@dataclass
class DefaultOption:
    option_name: str
    option_type: str
    value_type: str
    num_unit: str
    default_value: OptionValue

@dataclass
class Option:
    name: str
    option_values: List[str]


def python_type_to_json_schema(py_type):
    """Map Python types to JSON Schema types."""
    mapping = {
        int: "integer",
        float: "number",
        str: "string",
        bool: "boolean",
        list: "array",
        dict: "object",
        None: "null",
    }

    # Handle Optional types (Union[SomeType, None])
    if hasattr(py_type, "__origin__") and py_type.__origin__ is list:
        return "array"
    elif hasattr(py_type, "__origin__") and py_type.__origin__ is dict:
        return "object"
    elif hasattr(py_type, "__args__") and type(None) in py_type.__args__:
        return "null"
    else:
        return mapping.get(py_type, "object")  # Default to "object" if type not found

def openai_tool_decorator(tool_instructions=None):
    def decorator_fn(func):
        docstring = inspect.getdoc(func)

        description = ''
        args = defaultdict(dict)
        returns = ''

        if 'Args:' in docstring:
            description = docstring.split('Args:')[0].strip()
        else:
            description = docstring.strip()
        
        # Regex patterns to capture Args and Returns sections
        args_pattern = re.compile(r"Args:\n(.*?)\n\n", re.DOTALL)
        returns_pattern = re.compile(r"Returns:\n(.*)", re.DOTALL)

    # Find args section
        args_match = args_pattern.search(docstring)
        if args_match:
            args_section = args_match.group(1).strip()
            for line in args_section.splitlines():
                # Split by the first colon to separate the argument name from its description
                arg_name, arg_description = line.split(":", 1)
                args[arg_name.strip()]['description'] = arg_description.strip()

        signature = inspect.signature(func)
        for param_name, param in signature.parameters.items():
            if param_name in args:
                # Update type annotations if available
                if param.annotation == inspect.Parameter.empty:
                    assert False, "Type annotation is required for all parameters"
                args[param_name]['type_annotation'] = python_type_to_json_schema(param.annotation)

        # Find return section
        returns_match = returns_pattern.search(docstring)
        if returns_match:
            returns = returns_match.group(1).strip()

        func_params = {
                        arg_name: {
                            "type": arg_dict['type_annotation'],
                            "description": arg_dict['description']
                        }

                        for arg_name, arg_dict in args.items()
                    }

        full_description = description
        if tool_instructions is not None:
            if description[-1] != ".":
                full_description += "."
            full_description += " " + tool_instructions.strip()
        
        global OPENAI_TOOLS
        OPENAI_TOOLS.append({
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": full_description,
                "parameters": {
                    "type": "object",
                    "properties": func_params,
                    "required": list(args.keys()),
                    "additionalProperties": False
                }
            }
        })

        return func
    return decorator_fn

@openai_tool_decorator("Most customers either don't provide a complete order (i.e. not specifying required options like size)" \
                        "or are not aware of all the options available for a menu item. It is your job to help them with both cases.")
def get_menu_items_options(menu_item_id):
    """
    Get all the options available for the menu item.
    
    Args:
        menu_item_id: The menu item id used in the db.

    Returns:
        A MenuItem object.
    """

    response = (    
                supabase.table("menu_item_to_options_map")
                .select("*, option(name, type, value_type, num_unit), option_type_config(default_num_value, default_bool_value, default_discrete_value_id, discrete_option_value(name))")
                .eq("menu_item_id", menu_item_id)
                .execute()
            )
    data = response.data

    size_to_default_options_map = defaultdict(list)
    for item in data:
        option = item["option"]
        option_type_config = item["option_type_config"]
        do = DefaultOption(option["name"], option["type"], option["value_type"], option["num_unit"], 
                        OptionValue(option_type_config["default_num_value"], option_type_config["default_bool_value"], option_type_config["discrete_option_value"]["name"] if option_type_config["discrete_option_value"] is not None else None))

        size_to_default_options_map[item["cup_size"]].append(do)

    response = (    
                supabase.table("menu_item_to_options_map")
                .select("*, option(name), menu_item_to_option_values_map!inner(discrete_option_value(name))")
                .eq("menu_item_id", menu_item_id)
                .order('id')
                .execute()
            )

    # currently, this only supports discrete options that have default value.
    # Therefore, assume that other discrete options are not available for this menu item.
    size_to_available_options = defaultdict(list)
    for item in response.data:
        option = item["option"]
        option_values = [obj["discrete_option_value"]["name"] for obj in item["menu_item_to_option_values_map"]]
        size_to_available_options[item["cup_size"]].append(Option(option["name"], option_values))

    return size_to_default_options_map, size_to_available_options

@dataclass
class MenuItem:
    id: int
    name: str
    description: str
    group: str

@openai_tool_decorator()
def get_menu_item_from_name(menu_item_name:str):
    """
    Get the menu item given the name string of the menu item.
    
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
    item =  response.data[0]
    return MenuItem(item['id'], item["name"], item["description"], item['group'])
