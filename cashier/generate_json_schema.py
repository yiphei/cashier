from pydantic.json_schema import GenerateJsonSchema

def create_schema_generator(init_fields_set=None):
    init_fields_set = init_fields_set or []
    class NewGenerateJsonSchema(GenerateJsonSchema):
        fields_set = init_fields_set
        
        def model_fields_schema(self, schema):
            """Generates a JSON schema that matches a schema that defines a model's fields.

            Args:
                schema: The core schema.

            Returns:
                The generated JSON schema.
            """
            named_required_fields = [
                (name, self.field_is_required(field, total=True), field)
                for name, field in schema['fields'].items()
                if name in self.fields_set and self.field_is_present(field)
            ]
            if self.mode == 'serialization':
                named_required_fields.extend(self._name_required_computed_fields(schema.get('computed_fields', [])))
            json_schema = self._named_required_fields_schema(named_required_fields)
            extras_schema = schema.get('extras_schema', None)
            if extras_schema is not None:
                schema_to_update = self.resolve_schema_to_update(json_schema)
                schema_to_update['additionalProperties'] = self.generate_inner(extras_schema)
            return json_schema
        
    return NewGenerateJsonSchema



