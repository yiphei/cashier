from enum import StrEnum


class ModelProvider(StrEnum):
    OPENAI = "OPENAI"
    ANTHROPIC = "ANTHROPIC"
    NONE = "NONE"