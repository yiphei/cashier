from cashier.tool.tool_registry import ToolRegistry


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


@GeneralToolRegistry.model_tool_decorator()
def think_deep(thought: str) -> None:
    """
    Use the tool to think very deeply about something very important and difficult. This is a pure function (i.e. has no side effect).

    Args:
        thought: A long and deep thought to think about. There is no limit on the length of the thought, so longer and deeper thoughts are better.


    """
    return None