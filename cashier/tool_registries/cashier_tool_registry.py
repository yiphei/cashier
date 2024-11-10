import os
from collections import defaultdict
from enum import StrEnum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from supabase import Client
from supabase import create_client as create_supabase_client

from cashier.model_tool_decorator import ToolRegistry

supabase: Client = None


def create_db_client():
    global supabase
    supabase = create_supabase_client(
        os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")
    )


class Option(BaseModel):
    name: str
    type: str
    value_type: str
    num_unit: Optional[str]
    default_value: int | bool | str
    str_option_values: Optional[List[str]] = None


@ToolRegistry.model_tool_decorator(
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


@ToolRegistry.model_tool_decorator()
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


class CupSize(StrEnum):
    SHORT = "SHORT"
    TALL = "TALL"
    GRANDE = "GRANDE"
    VENTI = "VENTI"


class OptionOrder(BaseModel):
    name: str
    value: str | int | bool


class ItemOrder(BaseModel):
    name: str
    size: CupSize
    options: List[OptionOrder] = Field(
        description="Even if the order only has default options, they must be included here"
    )


class Order(BaseModel):
    item_orders: List[ItemOrder] = []
