from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class ScriptScene(BaseModel):
    scene_id: str = Field(..., description="Unique scene identifier")
    title: str = Field(..., description="Scene title")
    summary: str = Field(..., description="Short scene summary")
    dialogue_beats: List[str] = Field(default_factory=list)
    visual_cues: List[str] = Field(default_factory=list)


class ScriptLLMResponse(BaseModel):
    scenes: List[ScriptScene] = Field(default_factory=list)
