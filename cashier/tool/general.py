from cashier.tool_registry import ToolRegistry


class GeneralToolRegistry(ToolRegistry):
    pass


@GeneralToolRegistry.model_tool_decorator()
def think(thought: str) -> None:
    """
    Use the tool to think about something or when complex reasoning is needed. This is a pure function (i.e. has no side effect).

    Args:
        thought: A thought to think about.


    """
    return None
