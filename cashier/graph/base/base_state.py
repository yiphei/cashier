from __future__ import annotations

import copy
from typing import ClassVar, List, Optional

from pydantic import BaseModel, ConfigDict, create_model


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

    def get_set_schema_and_fields(self):
        field_kwargs = {
            field_name: field_info
            for field_name, field_info in self.model_fields.items()
            if field_name in self.model_fields_set
        }
        if len(field_kwargs.keys()) == 0:
            return None, None
        new_model = create_model(self.__class__.__name__ + "_sub", **field_kwargs)
        new_instance = new_model(**self.model_dump(include=self.model_fields_set))
        return new_model, new_instance
