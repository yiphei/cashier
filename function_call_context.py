class InexistentFunctionError(Exception):
    def __init__(self, fn_name):
        self.fn_name = fn_name

    def __str__(self):
        return f"InexistentToolException: the tool {self.fn_name} does not exist"

class FunctionCallContext:
    def __init__(self):
        self.exception = None

    def has_exception(self):
        return self.exception is not None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exception = exc_val
        return True
