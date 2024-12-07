from typing import Optional

from pydantic import Field

from cashier.graph.base.base_edge_schema import StateTransitionConfig
from cashier.graph.conversation_node import ConversationNodeSchema
from cashier.graph.edge_schema import EdgeSchema
from cashier.graph.graph_schema import GraphSchema
from cashier.graph.base.base_state import BaseStateModel
from cashier.graph.request_graph import RequestGraphSchema
from cashier.model.model_turn import AssistantTurn
from cashier.model.model_util import ModelProvider
from cashier.prompts.node_system import NodeSystemPrompt
from data.prompt.cashier_background import CashierBackgroundPrompt
from data.tool_registry.cashier_tool_registry import CASHIER_TOOL_REGISTRY, Order


class CashierNodeSystemPrompt(NodeSystemPrompt):
    BACKGROUND_PROMPT = CashierBackgroundPrompt


class TakeOrderState(BaseStateModel):
    resettable_fields = ["has_finished_ordering"]

    order: Optional[Order] = None
    has_finished_ordering: bool = Field(  # type: ignore
        description=(
            "whether the customer has finished ordering. This can only be true after"
            " you have explicitly confirmed with them that they have finished ordering,"
            " by asking questions like 'Anything else?'."
        ),
        default=False,
    )


take_order_node_schema = ConversationNodeSchema(
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
    node_system_prompt=CashierNodeSystemPrompt,
    tool_names=[
        "get_menu_items_options",
        "get_menu_item_from_name",
    ],
    tool_registry_or_tool_defs=CASHIER_TOOL_REGISTRY,
    input_schema=None,
    state_schema=TakeOrderState,
    first_turn=AssistantTurn(
        msg_content="hi, welcome to Heaven Coffee", model_provider=ModelProvider.NONE
    ),
)


class ConfirmOrderState(BaseStateModel):
    resettable_fields = ["has_confirmed_order"]

    has_confirmed_order: bool = Field(  # type: ignore
        description="whether the customer has confirmed their order",
        default=False,
    )


confirm_order_node_schema = ConversationNodeSchema(
    node_prompt=(
        "Confirm the order with the customer. You do this by"
        " repeating the order back to them and get their confirmation."
    ),
    node_system_prompt=CashierNodeSystemPrompt,
    tool_names=None,
    tool_registry_or_tool_defs=None,
    input_schema=Order,
    state_schema=ConfirmOrderState,
)
take_to_confirm_edge_schema = EdgeSchema(
    from_node_schema=take_order_node_schema,
    to_node_schema=confirm_order_node_schema,
    transition_config=StateTransitionConfig(
        need_user_msg=True,
        state_check_fn_map={
            "has_finished_ordering": lambda val: val,
            "order": lambda val: val is not None,
        },
    ),
    new_input_fn=lambda state: state.order,  # type: ignore
)


class TerminalOrderState(BaseStateModel):
    resettable_fields = ["has_said_goodbye"]

    has_said_goodbye: bool = Field(  # type: ignore
        description="whether the customer has said goodbye",
        default=False,
    )


terminal_order_node_schema = ConversationNodeSchema(
    node_prompt=("Order has been successfully placed. Thank the customer."),
    node_system_prompt=CashierNodeSystemPrompt,
    tool_names=None,
    tool_registry_or_tool_defs=None,
    input_schema=None,
    state_schema=TerminalOrderState,
)
confirm_to_terminal_edge_schema = EdgeSchema(
    from_node_schema=confirm_order_node_schema,
    to_node_schema=terminal_order_node_schema,
    transition_config=StateTransitionConfig(
        need_user_msg=True, state_check_fn_map={"has_confirmed_order": lambda val: val}
    ),
    new_input_fn=lambda state: None,
)


class GraphState(BaseStateModel):
    order: Optional[Order] = None


cashier_graph_schema = GraphSchema(
    description="Help the customer place a coffee order",
    output_schema=Order,
    start_node_schema=take_order_node_schema,
    last_node_schema=terminal_order_node_schema,
    edge_schemas=[take_to_confirm_edge_schema, confirm_to_terminal_edge_schema],
    node_schemas=[
        take_order_node_schema,
        confirm_order_node_schema,
        terminal_order_node_schema,
    ],
    state_schema=GraphState,
    completion_config=StateTransitionConfig(
        need_user_msg=True,
        state_check_fn_map={"has_said_goodbye": lambda val: bool(val)},
    ),
)


REQUEST_GRAPH_SCHEMA = RequestGraphSchema(
    node_schemas=[cashier_graph_schema],
    edge_schemas=[],
    node_prompt="You work at Heaven Coffee and help customers with anything they need.",
    node_system_prompt=CashierNodeSystemPrompt,
    description="todo",
)
