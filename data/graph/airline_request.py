from cashier.graph.request_graph import RequestGraphSchema
from data.graph.airline_book_flight import BOOK_FLIGHT_GRAPH_SCHEMA
from data.prompt.airline import AirlineNodeSystemPrompt

AIRLINE_REQUEST_SCHEMA = RequestGraphSchema(
    node_schemas=[BOOK_FLIGHT_GRAPH_SCHEMA],
    edge_schemas=[],
    node_prompt="You are a helpful assistant that helps customers with flight-related requests.",
    node_system_prompt=AirlineNodeSystemPrompt,
    description="Help customers change flights and baggage information for a reservation.",
)
