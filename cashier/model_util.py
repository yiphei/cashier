from __future__ import annotations

import json
import random
import string
from collections import defaultdict
from enum import StrEnum
from typing import Any, Dict, Optional, cast

from pydantic import BaseModel, Field, model_validator

from cashier.function_call_context import ToolExceptionWrapper


class ModelProvider(StrEnum):
    OPENAI = "OPENAI"
    ANTHROPIC = "ANTHROPIC"
    NONE = "NONE"


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
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


def generate_random_string(length: int) -> str:
    """
    Generate a random string of specified length using alphanumeric characters
    (both uppercase and lowercase).

    Args:
        length (int): The desired length of the random string

    Returns:
        str: A random alphanumeric string
    """
    # Define the character set: uppercase + lowercase + digits
    charset = string.ascii_letters + string.digits

    # Generate random string using random.choices
    # choices() is preferred over choice() in a loop as it's more efficient
    return "".join(random.choices(charset, k=length))


MODEL_PROVIDER_TO_TOOL_CALL_ID_PREFIX = {
    ModelProvider.ANTHROPIC: "toolu_",
    ModelProvider.OPENAI: "call_",
}


class FunctionCall(BaseModel):
    name: str
    id: str
    # when using model_dump, must add by_alias=True to get the alias names
    input_args_json: Optional[str] = Field(default=None, alias="args_json")
    input_args: Optional[Dict] = Field(default=None, alias="args")

    @model_validator(mode="after")
    def check_function_args(self) -> FunctionCall:
        if self.input_args_json is None and self.input_args is None:
            raise ValueError("One of [args_json, args] must be provided")

        if self.input_args_json is not None and self.input_args is None:
            if self.input_args_json:
                self.input_args = json.loads(self.input_args_json)
            else:
                # This case always happens when claude models call inexistent functions.
                # We still want to construct the function call and let it error downstream.
                self.input_args = {}
                self.input_args_json = "{}"
        if self.input_args is not None and self.input_args_json is None:
            self.input_args_json = json.dumps(self.input_args)
        return self

    @classmethod
    def create_fake_fn_call(
        cls,
        model_provider: ModelProvider,
        name: str,
        args_json: Optional[str] = None,
        args: Optional[Dict] = None,
    ) -> FunctionCall:
        id_prefix = MODEL_PROVIDER_TO_TOOL_CALL_ID_PREFIX[model_provider]
        fake_id = id_prefix + generate_random_string(24)

        return FunctionCall(
            name=name,
            id=fake_id,
            args_json=args_json,
            args=args,
        )

    # the following is to pass mypy checks

    @property
    def args_json(self) -> str:
        return cast(str, self.input_args_json)

    @property
    def args(self) -> Dict:
        return cast(Dict, self.input_args_json)
