from typing import List, Optional

from pydantic import BaseModel

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
    FlightReservationInfo,
    FlightType,
    InsuranceValue,
    PassengerInfo,
    PaymentDetails,
    ReservationDetails,
    UserDetails,
)

## book flight graph

PREAMBLE = (
    "You are helping the customer to change the baggage information for a reservation. "
)


class UserState(BaseStateModel):
    user_details: UserDetails = None


get_user_id_node_schema = ConversationNodeSchema(
    node_prompt=PREAMBLE + "Right now, you need to get their user details.",
    node_system_prompt=AirlineNodeSystemPrompt,
    state_schema=UserState,
    tool_registry_or_tool_defs=AIRLINE_TOOL_REGISTRY,
    tool_names=["get_user_details", "calculate"],
)

# ---------------------------------------------------------


class ReservationDetailsState(BaseStateModel):
    reservation_details: Optional[ReservationDetails] = None


get_reservation_details_node_schema = ConversationNodeSchema(
    node_prompt=PREAMBLE
    + "Right now, you need to get the reservation details by asking for the reservation id. If they don't know the id, lookup each reservation in their user details and find the one that best matches their description .",
    node_system_prompt=AirlineNodeSystemPrompt,
    state_schema=ReservationDetailsState,
    tool_registry_or_tool_defs=AIRLINE_TOOL_REGISTRY,
    tool_names=[
        "get_reservation_details",
        "calculate",
        "list_all_airports",
    ],
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
    payment_id: Optional[str] = None


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
    + "Right now, you have all the data necessary to change the baggage for the reservation.",
    node_system_prompt=AirlineNodeSystemPrompt,
    state_schema=None,
    tool_registry_or_tool_defs=AIRLINE_TOOL_REGISTRY,
    tool_names=[
        "update_reservation_baggages",
        "calculate",
    ],
)

# ---------------------------------------------------------

edge_1 = EdgeSchema(
    from_node_schema=get_user_id_node_schema,
    to_node_schema=get_reservation_details_node_schema,
    transition_config=StateTransitionConfig(
        need_user_msg=True,
        state_check_fn_map={"user_details": lambda val: val is not None},
    ),
)

edge_2 = EdgeSchema(
    from_node_schema=get_reservation_details_node_schema,
    to_node_schema=luggage_node_schema,
    transition_config=StateTransitionConfig(
        need_user_msg=True,
        state_check_fn_map={"reservation_details": lambda val: bool(val)},
    ),
)

edge_3 = EdgeSchema(
    from_node_schema=luggage_node_schema,
    to_node_schema=payment_node_schema,
    transition_config=StateTransitionConfig(
        need_user_msg=True,
        state_check_fn_map={
            "total_baggages": lambda val: val is not None,
            "nonfree_baggages": lambda val: val is not None,
        },
    ),
)

edge_4 = EdgeSchema(
    from_node_schema=payment_node_schema,
    to_node_schema=book_flight_node_schema,
    transition_config=StateTransitionConfig(
        need_user_msg=True,
        state_check_fn_map={
            "payment_id": lambda val: val is not None,
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


class ChangeBaggageGraphStateSchema(BaseStateModel):
    user_details: Optional[UserDetails] = None
    reservation_details: Optional[ReservationDetails] = None
    total_baggages: Optional[int] = None
    nonfree_baggages: Optional[int] = None
    payment_id: Optional[str] = None


CHANGE_BAGGAGE_GRAPH_SCHEMA = GraphSchema(
    description="Help customers update baggage information for a reservation",
    start_node_schema=get_user_id_node_schema,
    output_schema=GraphOutputSchema,
    end_node_schema=book_flight_node_schema,
    edge_schemas=[edge_1, edge_2, edge_3, edge_4],
    node_schemas=[
        get_user_id_node_schema,
        get_reservation_details_node_schema,
        luggage_node_schema,
        payment_node_schema,
        book_flight_node_schema,
    ],
    completion_config=FunctionTransitionConfig(
        need_user_msg=False,
        fn_name="update_reservation_baggages",
        state=FunctionState.CALLED_AND_SUCCEEDED,
    ),
    state_schema=ChangeBaggageGraphStateSchema,
)
