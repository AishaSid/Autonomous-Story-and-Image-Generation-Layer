from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class CharacterProfile(BaseModel):
    name: str = Field(..., description="Exact character name")
    age: str = Field(..., description="Apparent age or age range")
    personality_traits: List[str] = Field(default_factory=list)
    appearance_description: str = Field(..., description="Physical build and face details")
    clothing: str = Field(..., description="Precise clothing with fabric and color details")
    hair_texture: str = Field(..., description="Hair texture and style")
    eye_color: str = Field(..., description="Eye color")
    signature_item: str = Field(..., description="Distinctive item or mark")
    base_visual_style: str = Field(..., description="Base visual style for image generation")
    reference_image_path: str = Field(default="", description="Optional image reference path")


class ScriptCharacter(BaseModel):
    name: str = Field(..., description="Exact character name")
    personality_traits: List[str] = Field(default_factory=list)
    appearance_description: str = Field(..., description="Visual identity and appearance details")
    reference_style: str = Field(..., description="Style guidance for image generation")


class ScriptScene(BaseModel):
    scene_id: str = Field(..., description="Unique scene identifier")
    title: str = Field(..., description="Scene title")
    summary: str = Field(..., description="Short scene summary")
    dialogue_beats: List[str] = Field(default_factory=list)
    visual_cues: List[str] = Field(default_factory=list)


class ScriptLLMResponse(BaseModel):
    characters: List[ScriptCharacter] = Field(default_factory=list)
    scenes: List[ScriptScene] = Field(default_factory=list)


class CharacterDBResponse(BaseModel):
    characters: List[CharacterProfile] = Field(default_factory=list)
