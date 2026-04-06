from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import chromadb
from chromadb.api.models.Collection import Collection


class MemoryLayer:
    def __init__(self, persist_directory: str = "./chroma_db") -> None:
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.script_history = self.client.get_or_create_collection(name="script_history")
        self.character_metadata = self.client.get_or_create_collection(name="character_metadata")
        self.image_references = self.client.get_or_create_collection(name="image_references")

        self._collections: Dict[str, Collection] = {
            "script_history": self.script_history,
            "character_metadata": self.character_metadata,
            "image_references": self.image_references,
        }

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def add_character(
        self,
        name: str,
        personality_traits: Optional[List[str]] = None,
        appearance_description: str = "",
        reference_style: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        record_id: Optional[str] = None,
    ) -> str:
        character_id = record_id or str(uuid4())
        traits = personality_traits or []

        document = (
            f"Name: {name}\n"
            f"Traits: {', '.join(traits)}\n"
            f"Appearance: {appearance_description}\n"
            f"Reference Style: {reference_style}"
        )

        record_metadata: Dict[str, Any] = {
            "name": name,
            "personality_traits": ",".join(traits),
            "appearance_description": appearance_description,
            "reference_style": reference_style,
            "created_at": self._utc_now_iso(),
        }
        if metadata:
            record_metadata.update(metadata)

        self.character_metadata.add(
            ids=[character_id],
            documents=[document],
            metadatas=[record_metadata],
        )
        return character_id

    def add_script_segment(
        self,
        scene_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        record_id: Optional[str] = None,
    ) -> str:
        segment_id = record_id or str(uuid4())

        record_metadata: Dict[str, Any] = {
            "scene_id": scene_id,
            "created_at": self._utc_now_iso(),
        }
        if metadata:
            record_metadata.update(metadata)

        self.script_history.add(
            ids=[segment_id],
            documents=[content],
            metadatas=[record_metadata],
        )
        return segment_id

    def query_memory(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if collection_name not in self._collections:
            raise ValueError(
                "collection_name must be one of: script_history, character_metadata, image_references"
            )

        collection = self._collections[collection_name]
        query_kwargs: Dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": n_results,
        }
        if where:
            query_kwargs["where"] = where

        return collection.query(**query_kwargs)
