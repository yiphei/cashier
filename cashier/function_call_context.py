from __future__ import annotations

from typing import Any


class InexistentFunctionError(Exception):
    def __init__(self, fn_name: str):
        self.fn_name = fn_name

    def __str__(self) -> str:
        return f"the tool {self.fn_name} does not exist"


class StateUpdateError(Exception):
    pass


class ToolExceptionWrapper:
    def __init__(self, exception: Exception):
        self.exception = exception

    def __str__(self) -> str:
        return f"{self.exception.__class__.__name__}: {str(self.exception)}"


class FunctionCallContext:
    def __init__(self):
        self.exception = None

    def has_exception(self) -> bool:
        return self.exception is not None

    def __enter__(self) -> FunctionCallContext:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if exc_val:
            self.exception = ToolExceptionWrapper(exc_val)
        return True
