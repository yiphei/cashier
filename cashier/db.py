import os
from supabase import Client
from supabase import create_client as create_supabase_client
from typing import Optional


class DBClient:
    _client: Optional[Client] = None

    @classmethod
    def initialize(cls) -> None:
        if cls._client is None:
            cls._client = create_supabase_client(
        os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")  # type: ignore
    )

    @classmethod
    def get_client(
        cls
    ) -> Client:
        if cls._client is None:
            raise RuntimeError("Client not initialized. Call initialize() first.")
        return cls._client
