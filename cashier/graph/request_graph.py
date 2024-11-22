from cashier.prompt_action.graph_selection_action import GraphSelectionAction


class RequestGraphSchema:
    graph_schemas = None
    system_prompt = None

    @classmethod
    def get_graph_schemas(cls, request):
        graph_schema_ids = GraphSelectionAction.run(
            "claude-3.5", request, cls.graph_schemas
        )
        return [
            graph_schema
            for graph_schema in cls.graph_schemas
            if graph_schema.id in graph_schema_ids
        ]
