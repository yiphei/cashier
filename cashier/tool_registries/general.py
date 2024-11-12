from cashier.tool_registry import ToolRegistry

class GeneralToolRegistry(ToolRegistry):
    pass

GeneralToolRegistry.model_tool_decorator()
def think(thought: str) -> None:
    """
    Use the tool to think about something. It will not obtain new information or change the database, but just append the thought to the log. Use it when complex reasoning is needed.

    Args:
        thought: A thought to think about.


    """
    return None