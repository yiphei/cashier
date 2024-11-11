import json
from collections import defaultdict
from enum import StrEnum
from typing import Dict, Optional

from pydantic import BaseModel, model_validator

from cashier.function_call_context import ToolExceptionWrapper


class ModelProvider(StrEnum):
    OPENAI = "OPENAI"
    ANTHROPIC = "ANTHROPIC"
    NONE = "NONE"


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        elif isinstance(obj, (defaultdict, dict)):
            return {self.default(k): self.default(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.default(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, ToolExceptionWrapper):
            return str(obj)
        return super().default(obj)


class FunctionCall(BaseModel):
    function_name: str
    tool_call_id: str
    function_args_json: Optional[str] = None
    function_args: Optional[Dict] = None

    @model_validator(mode="after")
    def check_function_args(self):
        if self.function_args_json is None and self.function_args is None:
            raise ValueError(
                "One of [function_args_json, function_args] must be provided"
            )

        if self.function_args_json is not None and self.function_args is None:
            if self.function_args_json:
                self.function_args = json.loads(self.function_args_json)
            else:
                # This case always happens when claude models call inexistent functions.
                # We still want to construct the function call and let it error downstream.
                self.function_args = {}
                self.function_args_json = "{}"
        if self.function_args is not None and self.function_args_json is None:
            self.function_args_json = json.dumps(self.function_args)
        return self
    
    @classmethod
    def create_fake_call(cls, fn_name, fn_args_json, fn_args):
        id = "toolu_01CmtofC946qXZNABne7Lobb"
        return FunctionCall(function_name=fn_name, tool_call_id=id,  function_args_json = fn_args_json, function_args= fn_args)