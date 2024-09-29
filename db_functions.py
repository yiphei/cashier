import os
from supabase import create_client as create_supabase_client, Client
from dataclasses import dataclass
from collections import defaultdict
from typing import List

supabase: Client = None

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

def get_menu_items_options(menu_item_id):
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
    name: str
    description: str
    group: str

def get_menu_item_from_name(menu_item_name):
    formatted_menu_item_name = menu_item_name.replace(" ", "&")
    response = (    
            supabase.table("menu_item")
            .select("id, name, description, group")
            .text_search("name", formatted_menu_item_name)
            .execute()
        )
    item =  response.data[0]
    return MenuItem(item["name"], item["description"], item['group'])
