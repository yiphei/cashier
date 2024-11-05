from typing import Optional

from pydantic import Field

from db_functions import Order
from graph import BaseStateModel, EdgeSchema, NodeSchema
from model import AssistantTurn
from model_util import ModelProvider


class TakeOrderState(BaseStateModel):
    order: Optional[Order] = None
    has_finished_ordering: bool = Field(
        description=(
            "whether the customer has finished ordering. This can only be true after"
            " you have explicitly confirmed with them that they have finished ordering,"
            " by asking questions like 'Anything else?'."
        ),
        default=False,
        resettable=True,
    )


take_order_node_schema = NodeSchema(
    node_prompt=(
        "First, greet the customer. Then, your main job is to take their orders, which"
        " also includes answering reasonable questions about the shop & menu only and"
        " assisting them with any issues they may have about their orders. Things you"
        " must keep in mind:\n"
        "- never assume the drink size\n"
        "- given a size, always assume the default option values of that size, unless the customer"
        " explicitly specifies otherwise. This also means that you do not need to ask if they are ok with the default option values,"
        " and you do not need to spell them out (not even say e.g. 'grande size with the default options'), unless the customer asks otherwise."
        " But if an option does not have a default value, do not make any assumptions.\n"
        "- as soon as a single drink order is completed (e.g. all the required options are specified),"
        " you must restate the order back to the customer to ensure that it is correct. For the option values,"
        " only state those that were explicitly specified by the customer, so do not state default options"
        " assumed for the size but do state any other option, even default option values that were specified by the customer.\n\n"
        "Lastly, if they are not immediately ready to order after the greeting, you can also engage in some"
        " small talk about any topic, but you need to steer the conversation back to ordering after some"
        " back-and-forths."
    ),
    tool_fn_names=[
        "get_menu_items_options",
        "get_menu_item_from_name",
    ],
    input_pydantic_model=None,
    state_pydantic_model=TakeOrderState,
    first_turn=AssistantTurn(
        msg_content="hi, welcome to Heaven Coffee", model_provider=ModelProvider.NONE
    ),
)


class ConfirmOrderState(BaseStateModel):
    has_confirmed_order: bool = Field(
        description="whether the customer has confirmed their order",
        default=False,
        resettable=True,
    )


confirm_order_node_schema = NodeSchema(
    node_prompt=(
        "Confirm the order with the customer. You do this by"
        " repeating the order back to them and get their confirmation."
    ),
    tool_fn_names=[],
    input_pydantic_model=Order,
    state_pydantic_model=ConfirmOrderState,
)
take_to_confirm_edge_schema = EdgeSchema(
    from_node_schema=take_order_node_schema,
    to_node_schema=confirm_order_node_schema,
    state_condition_fn=lambda state: state.has_finished_ordering
    and state.order is not None,
    new_input_from_state_fn=lambda state: state.order,
)


class TerminalOrderState(BaseStateModel):
    has_said_goodbye: bool = Field(
        description="whether the customer has said goodbye",
        default=False,
        resettable=True,
    )


terminal_order_node_schema = NodeSchema(
    node_prompt=("Order has been successfully placed. Thank the customer."),
    tool_fn_names=[],
    input_pydantic_model=None,
    state_pydantic_model=TerminalOrderState,
)
confirm_to_terminal_edge_schema = EdgeSchema(
    from_node_schema=confirm_order_node_schema,
    to_node_schema=terminal_order_node_schema,
    state_condition_fn=lambda state: state.has_confirmed_order,
    new_input_from_state_fn=lambda state: None,
)
FROM_NODE_SCHEMA_ID_TO_EDGE_SCHEMA = {
    take_order_node_schema.id: [take_to_confirm_edge_schema],
    confirm_order_node_schema.id: [confirm_to_terminal_edge_schema],
}
NODE_SCHEMA_ID_TO_NODE_SCHEMA = {
    take_order_node_schema.id: take_order_node_schema,
    confirm_order_node_schema.id: confirm_order_node_schema,
    terminal_order_node_schema.id: terminal_order_node_schema,
}
EDGE_SCHEMA_ID_TO_EDGE_SCHEMA = {
    take_to_confirm_edge_schema.id: take_to_confirm_edge_schema,
    confirm_to_terminal_edge_schema.id: confirm_to_terminal_edge_schema,
}
