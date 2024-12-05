from __future__ import annotations

import copy
from typing import Any, ClassVar, List, Optional, Type

from pydantic import BaseModel, ConfigDict

from cashier.tool.function_call_context import StateUpdateError


class BaseStateModel(BaseModel):
    resettable_fields: ClassVar[Optional[List[str]]] = None
    model_config = ConfigDict(extra="forbid")

    def copy_resume(self) -> BaseStateModel:
        new_data = copy.deepcopy(dict(self))

        # Iterate through fields and reset those marked as resettable
        if self.resettable_fields:
            for field_name, field_info in self.model_fields.items():
                # Check if field has the resettable marker in its metadata
                if field_name in self.resettable_fields:
                    new_data[field_name] = field_info.default

        return self.__class__(**new_data)