class InexistentFunctionError(Exception):
    def __init__(self, fn_name):
        self.fn_name = fn_name

    def __str__(self):
        return f"the tool {self.fn_name} does not exist"


class ExceptionWrapper:
    def __init__(self, exception):
        self.exception = exception

    def __str__(self):
        return f"{self.exception.__class__.__name__}: {str(self.exception)}"


class FunctionCallContext:
    def __init__(self):
        self.exception = None

    def has_exception(self):
        return self.exception is not None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            self.exception = ExceptionWrapper(exc_val)
        return True
