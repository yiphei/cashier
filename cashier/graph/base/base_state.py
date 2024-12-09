from __future__ import annotations

import copy
from typing import ClassVar, List, Optional, Dict

from pydantic import BaseModel, ConfigDict


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

    def copy_data(self) -> Dict:
        return copy.deepcopy(dict(self))
