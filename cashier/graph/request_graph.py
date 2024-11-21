from cashier.prompt_action.graph_selection_action import GraphSelectionAction


class RequestGraph:
    def __init__(self, graph_schemas):
        self.graph_schemas = graph_schemas

    def get_graph_schemas(self, request):
        graph_schema_ids = GraphSelectionAction.run(
            "claude-3.5", request, self.graph_schemas
        )
        return [
            graph_schema
            for graph_schema in self.graph_schemas
            if graph_schema.id in graph_schema_ids
        ]
