from enum import StrEnum
from pydantic import ValidationError

class FnCallError(StrEnum):
    INEXISTENT_FUNCTION = "INEXISTENT_FUNCTION"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    OTHER = "OTHER"

class FunctionCallContext:
    def __init__(self):
        self.error_type = None
        self.error_msg = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type == Exception:
            self.error_type = FnCallError.OTHER
            self.error_msg = str(exc_val)
        elif exc_type == ValidationError:
            self.error_type = FnCallError.VALIDATION_ERROR
            self.error_msg = str()
        return True