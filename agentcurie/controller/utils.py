from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue

def resolve_schema(schema: JsonSchemaValue) -> JsonSchemaValue:
    """Recursively resolve $ref references in a schema."""
    if not isinstance(schema, dict):
        return schema

    defs = schema.get("$defs", {})
    
    def _resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                ref = node["$ref"]
                if ref.startswith("#/$defs/"):
                    name = ref.split("/")[-1]
                    return _resolve(defs.get(name, {}))
            return {k: _resolve(v) for k, v in node.items()}
        elif isinstance(node, list):
            return [_resolve(v) for v in node]
        return node

    return _resolve(schema) #type:ignore