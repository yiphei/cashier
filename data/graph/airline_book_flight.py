from typing import List, Optional

from pydantic import BaseModel, Field

from cashier.graph.base.base_edge_schema import (
    FunctionState,
    FunctionTransitionConfig,
    StateTransitionConfig,
)
from cashier.graph.base.base_state import BaseStateModel
from cashier.graph.conversation_node import ConversationNodeSchema
from cashier.graph.edge_schema import EdgeSchema
from cashier.graph.graph_schema import GraphSchema
from data.prompt.airline import AirlineNodeSystemPrompt
from data.tool_registry.airline_tool_registry import AIRLINE_TOOL_REGISTRY
from data.types.airline import (
    CabinType,
    FlightInfo,
    FlightReservationInfo,
    FlightType,
    InsuranceValue,
    PassengerInfo,
    PaymentDetails,
    PaymentMethod,
    UserDetails,
)

## book flight graph

PREAMBLE = "You are helping the customer to book a flight. "


class UserState(BaseStateModel):
    user_details: Optional[UserDetails] = None


get_user_id_node_schema = ConversationNodeSchema(
    node_prompt=PREAMBLE + "Right now, you need to get their user details.",
    node_system_prompt=AirlineNodeSystemPrompt,
    state_schema=UserState,
    tool_registry_or_tool_defs=AIRLINE_TOOL_REGISTRY,
    tool_names=["get_user_details", "calculate"],
)

# ---------------------------------------------------------


class FlightOrder(BaseStateModel):
    flight_infos: List[FlightInfo] = Field(default_factory=list)


find_flight_node_schema = ConversationNodeSchema(
    node_prompt=PREAMBLE + "Right now, you need to help find flights for them.",
    node_system_prompt=AirlineNodeSystemPrompt,
    state_schema=FlightOrder,
    tool_registry_or_tool_defs=AIRLINE_TOOL_REGISTRY,
    tool_names=[
        "search_direct_flight",
        "search_onestop_flight",
        "list_all_airports",
        "calculate",
        "get_reservation_details",
    ],
)


# ---------------------------------------------------------


class PassengerState(BaseStateModel):
    passengers: List[PassengerInfo] = Field(default_factory=list)


get_passanger_info_schema = ConversationNodeSchema(
    node_prompt=(
        PREAMBLE
        + (
            "Right now, you need to get the passenger info of all the passengers. "
            "Each reservation can have at most five passengers. All passengers must fly the same flights in the same cabin."
        )
    ),
    node_system_prompt=AirlineNodeSystemPrompt,
    state_schema=PassengerState,
    tool_registry_or_tool_defs=AIRLINE_TOOL_REGISTRY,
    tool_names=["calculate"],
)


# ---------------------------------------------------------


class InsuranceState(BaseStateModel):
    add_insurance: Optional[InsuranceValue] = Field(
        default=None, description="whether to add insurance for all passengers"
    )


ask_for_insurance_node_schema = ConversationNodeSchema(
    node_prompt=PREAMBLE
    + "Right now, you need to ask if they want to add insurance, which is 30 dollars per passenger and enables full refund if the user needs to cancel the flight given health or weather reasons.",
    node_system_prompt=AirlineNodeSystemPrompt,
    state_schema=InsuranceState,
    tool_registry_or_tool_defs=AIRLINE_TOOL_REGISTRY,
    tool_names=["calculate"],
)

# ------------------------------------------


class LuggageState(BaseStateModel):
    total_baggages: Optional[int] = None
    nonfree_baggages: Optional[int] = None


luggage_node_schema = ConversationNodeSchema(
    node_prompt=PREAMBLE
    + (
        "Right now, you need to ask how many luggages to check. "
        "If the booking user is a regular member, 0 free checked bag for each basic economy passenger, 1 free checked bag for each economy passenger, and 2 free checked bags for each business passenger. If the booking user is a silver member, 1 free checked bag for each basic economy passenger, 2 free checked bag for each economy passenger, and 3 free checked bags for each business passenger. If the booking user is a gold member, 2 free checked bag for each basic economy passenger, 3 free checked bag for each economy passenger, and 3 free checked bags for each business passenger. Each extra baggage is 50 dollars."
    ),
    node_system_prompt=AirlineNodeSystemPrompt,
    state_schema=LuggageState,
    tool_registry_or_tool_defs=AIRLINE_TOOL_REGISTRY,
    tool_names=["calculate"],
)

# ---------------------------------------------------------


class PaymentState(BaseStateModel):
    resettable_fields = [
        "has_explained_payment_policy_to_customer",
        "is_payment_finalized",
    ]

    payments: List[PaymentMethod] = Field(default_factory=list)
    has_explained_payment_policy_to_customer: bool = Field(
        default=False,
        description="There are very important payment policies, and these must be clearly communicated to the customer. Most importantly, the customer must understand that any left-over balance on a travel certificate will be forfeited.",
    )
    is_payment_finalized: bool = Field(
        default=False,
        description="This can only be true after payment policy has been communicated and payment method collected",
    )


payment_node_schema = ConversationNodeSchema(
    node_prompt=PREAMBLE
    + (
        "Right now, you need to get the payment information. "
        "IMPORTANT: Each reservation can use AT MOST one travel certificate, AT MOST one credit card, and AT MOST three gift cards. The remaining unused amount of a travel certificate is not refundable (i.e. forfeited). All payment methods must already be in user profile for safety reasons."
    ),
    node_system_prompt=AirlineNodeSystemPrompt,
    state_schema=PaymentState,
    tool_registry_or_tool_defs=AIRLINE_TOOL_REGISTRY,
    tool_names=["calculate"],
)


# ---------------------------------------------------------

book_flight_node_schema = ConversationNodeSchema(
    node_prompt=PREAMBLE
    + "Right now, you have all the data necessary to place the booking.",
    node_system_prompt=AirlineNodeSystemPrompt,
    state_schema=None,
    tool_registry_or_tool_defs=AIRLINE_TOOL_REGISTRY,
    tool_names=[
        "book_reservation",
        "calculate",
    ],
)

# ---------------------------------------------------------

edge_1 = EdgeSchema(
    from_node_schema=get_user_id_node_schema,
    to_node_schema=find_flight_node_schema,
    transition_config=StateTransitionConfig(
        need_user_msg=False,
        state_check_fn_map={"user_details": lambda val: val is not None},
    ),
)

edge_2 = EdgeSchema(
    from_node_schema=find_flight_node_schema,
    to_node_schema=get_passanger_info_schema,
    transition_config=StateTransitionConfig(
        need_user_msg=False,
        state_check_fn_map={"flight_infos": lambda val: val and len(val) > 0},
    ),
)

edge_3 = EdgeSchema(
    from_node_schema=get_passanger_info_schema,
    to_node_schema=ask_for_insurance_node_schema,
    transition_config=StateTransitionConfig(
        need_user_msg=False,
        state_check_fn_map={"passengers": lambda val: val and len(val) > 0},
    ),
)

edge_4 = EdgeSchema(
    from_node_schema=ask_for_insurance_node_schema,
    to_node_schema=luggage_node_schema,
    transition_config=StateTransitionConfig(
        need_user_msg=False,
        state_check_fn_map={"add_insurance": lambda val: val is not None},
    ),
)

edge_5 = EdgeSchema(
    from_node_schema=luggage_node_schema,
    to_node_schema=payment_node_schema,
    transition_config=StateTransitionConfig(
        need_user_msg=False,
        state_check_fn_map={
            "total_baggages": lambda val: val is not None,
            "nonfree_baggages": lambda val: val is not None,
        },
    ),
)

edge_6 = EdgeSchema(
    from_node_schema=payment_node_schema,
    to_node_schema=book_flight_node_schema,
    transition_config=StateTransitionConfig(
        need_user_msg=True,
        state_check_fn_map={
            "is_payment_finalized": lambda val: val is True,
            "has_explained_payment_policy_to_customer": lambda val: val is True,
            "payments": lambda val: val and len(val) > 0,
        },
    ),
)
# --------------------


class GraphOutputSchema(BaseModel):
    reservation_id: str
    user_id: str
    origin: str
    destination: str
    flight_type: FlightType
    cabin: CabinType
    flights: List[FlightReservationInfo]
    passengers: List[PassengerInfo]
    payment_history: List[PaymentDetails]
    created_at: str
    total_baggages: int
    nonfree_baggages: int
    insurance: InsuranceValue


class StateSchema(BaseStateModel):
    user_details: Optional[UserDetails] = None
    flight_infos: List[FlightInfo] = Field(default_factory=list)
    passengers: List[PassengerInfo] = Field(default_factory=list)
    add_insurance: Optional[InsuranceValue] = None
    total_baggages: Optional[int] = None
    nonfree_baggages: Optional[int] = None
    payments: List[PaymentMethod] = Field(default_factory=list)


BOOK_FLIGHT_GRAPH_SCHEMA = GraphSchema(
    description="Help customers books flights",
    start_node_schema=get_user_id_node_schema,
    output_schema=GraphOutputSchema,
    end_node_schema=book_flight_node_schema,
    edge_schemas=[edge_1, edge_2, edge_3, edge_4, edge_5, edge_6],
    node_schemas=[
        get_user_id_node_schema,
        find_flight_node_schema,
        get_passanger_info_schema,
        ask_for_insurance_node_schema,
        luggage_node_schema,
        payment_node_schema,
        book_flight_node_schema,
    ],
    completion_config=FunctionTransitionConfig(
        need_user_msg=False,
        fn_name="book_reservation",
        state=FunctionState.CALLED_AND_SUCCEEDED,
    ),
    state_schema=StateSchema,
)
