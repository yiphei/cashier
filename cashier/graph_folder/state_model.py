from __future__ import annotations

from pydantic import BaseModel, ConfigDict

import copy


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