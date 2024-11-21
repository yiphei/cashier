from typing import Optional, Union
from openai import OpenAI
import anthropic

from cashier.model.model_util import ModelProvider


class ModelClient:
    _oai_client: Optional[OpenAI] = None
    _anthropic_client: Optional[anthropic.Anthropic] = None
    model_provider_to_client = {}

    @classmethod
    def initialize(cls) -> None:
        if cls._oai_client is None:
            cls._oai_client = OpenAI()
            cls.model_provider_to_client[ModelProvider.OPENAI] = cls._oai_client
        if cls._anthropic_client is None:
            cls._anthropic_client = anthropic.Anthropic()
            cls.model_provider_to_client[ModelProvider.ANTHROPIC] = cls._anthropic_client

    @classmethod
    def get_client(cls, model_provider: ModelProvider) -> Union[OpenAI, anthropic.Anthropic]:
        if cls.model_provider_to_client[model_provider] is None:
            raise RuntimeError("Client not initialized. Call initialize() first.")
        return cls.model_provider_to_client[model_provider]