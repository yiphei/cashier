from cashier.graph.request_graph import GraphEdgeSchema, RequestGraphSchema
from data.graph.airline_book_flight_and_graph import BOOK_FLIGHT_AND_GRAPH_SCHEMA
from data.graph.airline_change_baggage import (
    CHANGE_BAGGAGE_GRAPH_SCHEMA,
    ChangeBaggageGraphStateSchema,
)
from data.graph.airline_change_flight import CHANGE_FLIGHT_GRAPH_SCHEMA
from data.prompt.airline import AirlineNodeSystemPrompt

GRAPH_EDGE_SCHEMA_1 = GraphEdgeSchema(
    from_node_schema=CHANGE_FLIGHT_GRAPH_SCHEMA,
    to_node_schema=CHANGE_BAGGAGE_GRAPH_SCHEMA,
    new_input_fn=lambda state: ChangeBaggageGraphStateSchema(
        user_details=state.user_details, reservation_details=state.reservation_details
    ).model_dump(include={"user_details", "reservation_details"}),
)

AIRLINE_REQUEST_SCHEMA = RequestGraphSchema(
    node_schemas=[
        BOOK_FLIGHT_AND_GRAPH_SCHEMA,
        CHANGE_FLIGHT_GRAPH_SCHEMA,
        CHANGE_BAGGAGE_GRAPH_SCHEMA,
    ],
    edge_schemas=[GRAPH_EDGE_SCHEMA_1],
    node_prompt="You are a helpful assistant that helps customers with flight-related requests.",
    node_system_prompt=AirlineNodeSystemPrompt,
    description="Help customers change flights and baggage information for a reservation.",
)
