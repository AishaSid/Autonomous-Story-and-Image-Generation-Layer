from __future__ import annotations

from typing import Any, Dict, Literal
from uuid import uuid4

from memory import MemoryLayer


def commit_memory(
    data: Dict[str, Any],
    collection_name: Literal["script_history", "character_metadata", "image_references"],
) -> Dict[str, Any]:
    memory = MemoryLayer()

    if collection_name == "script_history":
        record_id = memory.add_script_segment(
            scene_id=str(data.get("scene_id", "scene_mock")),
            content=str(data.get("content", "")),
            metadata=data.get("metadata") if isinstance(data.get("metadata"), dict) else None,
            record_id=str(data.get("record_id")) if data.get("record_id") else None,
        )
    elif collection_name == "character_metadata":
        record_id = memory.add_character(
            name=str(data.get("name", "Unknown")),
            personality_traits=[str(item) for item in data.get("personality_traits", [])],
            appearance_description=str(data.get("appearance_description", "")),
            reference_style=str(data.get("reference_style", "")),
            metadata=data.get("metadata") if isinstance(data.get("metadata"), dict) else None,
            record_id=str(data.get("record_id")) if data.get("record_id") else None,
        )
    else:
        record_id = str(data.get("record_id") or uuid4())
        memory.image_references.add(
            ids=[record_id],
            documents=[str(data.get("document", data.get("prompt", "")))],
            metadatas=[
                {
                    "created_at": MemoryLayer._utc_now_iso(),
                    **(data.get("metadata") if isinstance(data.get("metadata"), dict) else {}),
                }
            ],
        )

    return {
        "status": "success",
        "collection_name": collection_name,
        "record_id": record_id,
    }
