from __future__ import annotations

import copy
from typing import Any, Type, Optional

from pydantic import BaseModel, ConfigDict

from cashier.tool.function_call_context import StateUpdateError


class BaseStateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def copy_resume(self) -> BaseStateModel:
        new_data = copy.deepcopy(dict(self))

        # Iterate through fields and reset those marked as resettable
        for field_name, field_info in self.model_fields.items():
            # Check if field has the resettable marker in its metadata
            if field_info.json_schema_extra and field_info.json_schema_extra.get(
                "resettable"
            ):
                new_data[field_name] = field_info.default

        return self.__class__(**new_data)


class HasStateSchemaMixin:
    def __init__(self, state_pydantic_model: Optional[Type[BaseStateModel]]):
        self.state_pydantic_model = state_pydantic_model


class HasStateMixin:
    def __init__(
        self,
        state: BaseStateModel,
    ):
        self.state = state
        self.first_user_message = False

    def update_state(self, **kwargs: Any) -> None:
        if self.first_user_message:
            old_state = self.state.model_dump()
            new_state = old_state | kwargs
            self.state = self.state.__class__(**new_state)
        else:
            raise StateUpdateError(
                "cannot update any state field until you get the first customer message in the current conversation. Remember, the current conversation starts after <cutoff_msg>"
            )

    def get_state(self) -> BaseStateModel:
        return self.state

    def update_first_user_message(self) -> None:
        self.first_user_message = True
