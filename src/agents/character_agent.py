from __future__ import annotations

import json
import re
from typing import Any, Dict, List
from pathlib import Path

from state import State
from pydantic import ValidationError

from .common import build_llm_client, ensure_outputs_dir, load_json_file, outputs_path, resolve_llm_client, PROJECT_ROOT
from .models import CharacterDBResponse


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.strip().lower()).strip()


def _has_required_character_fields(character: Dict[str, object]) -> bool:
    required = ["appearance_description", "clothing", "signature_item"]
    return all(str(character.get(field, "")).strip() for field in required)


def _extract_json_from_text(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting the largest JSON object or array from mixed text.
    first_object = text.find("{")
    first_array = text.find("[")
    starts = [idx for idx in (first_object, first_array) if idx != -1]
    if not starts:
        return {}
    start = min(starts)

    last_object = text.rfind("}")
    last_array = text.rfind("]")
    end = max(last_object, last_array)
    if end <= start:
        return {}

    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return {}


def _normalize_character_payload(payload: Any) -> CharacterDBResponse:
    if isinstance(payload, dict):
        if isinstance(payload.get("characters"), list):
            return CharacterDBResponse.model_validate(payload)
        if all(key in payload for key in ("name", "age", "appearance_description")):
            return CharacterDBResponse.model_validate({"characters": [payload]})

    if isinstance(payload, list):
        return CharacterDBResponse.model_validate({"characters": payload})

    return CharacterDBResponse(characters=[])


def _invoke_character_response(llm: Any, prompt: str) -> CharacterDBResponse:
    structured_llm = llm.with_structured_output(CharacterDBResponse)
    try:
        response = structured_llm.invoke(prompt)
        if isinstance(response, CharacterDBResponse):
            return response
        return _normalize_character_payload(response)
    except Exception as exc:
        raw = llm.invoke(prompt)
        content = getattr(raw, "content", raw)

        if isinstance(content, list):
            content = "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        elif not isinstance(content, str):
            content = str(content)

        parsed = _extract_json_from_text(content)

        if not parsed and hasattr(exc, "response"):
            details = getattr(exc.response, "json", None)
            if callable(details):
                try:
                    err_json = details() or {}
                    failed_generation = (
                        err_json.get("error", {}).get("failed_generation", "")
                        if isinstance(err_json, dict)
                        else ""
                    )
                    parsed = _extract_json_from_text(str(failed_generation))
                except Exception:
                    parsed = {}

        try:
            return _normalize_character_payload(parsed)
        except ValidationError:
            return CharacterDBResponse(characters=[])


def _build_character_prompt(prompt: str, character_names: List[str]) -> str:
    names_text = ", ".join(character_names) if character_names else ""
    return (
        "You are a character design specialist using Llama 3. Return strictly valid JSON only. "
        "Generate one structured character profile for each named character. "
        "The output must match the schema exactly and must not include extra keys.\n\n"
        f"Story prompt: {prompt}\n"
        f"Character names: {names_text}\n\n"
        "For each character, include: name, age, personality_traits, appearance_description, clothing, hair_texture, eye_color, signature_item, base_visual_style, reference_image_path.\n"
        "Requirements:\n"
        "1) Age must be a believable apparent age or age range.\n"
        "2) Clothing must be precise and include fabric and color.\n"
        "3) Hair texture must describe texture and style, not just length.\n"
        "4) Eye color must be explicit.\n"
        "5) Signature item must be a memorable object, scar, or accessory.\n"
        "6) Base visual style must be a concise cinematic style phrase such as 'Realistic 35mm film, grainy, high contrast'.\n"
        "7) The character profiles should stay consistent across all scenes.\n"
        "8) Return only JSON that conforms to the schema."
    )


def character_node(state: State) -> Dict[str, List[Dict[str, object]]]:
    script = state.get("script", {})
    script_characters = script.get("characters", []) if isinstance(script, dict) else []

    memory_db = load_json_file(outputs_path("character_db.json"))
    if not memory_db:
        memory_db = load_json_file(PROJECT_ROOT / "character_db.json")
    memory_lookup = {
        str(character.get("name", "")).strip().lower(): character
        for character in memory_db.get("characters", [])
        if isinstance(character, dict) and character.get("name")
    }

    characters: List[Dict[str, object]] = []
    source_characters = [character for character in script_characters if isinstance(character, dict)]
    character_names = [
        str(character.get("name", "")).strip()
        for character in source_characters
        if character.get("name") and str(character.get("name", "")).strip()
    ]
    character_names = list(dict.fromkeys(character_names))
    if not character_names:
        raise ValueError("character_node requires script['characters'] with concrete names from scriptwriter output")

    llm = resolve_llm_client(state, default_provider="groq")
    if llm is None:
        llm = build_llm_client(provider="groq", model=str(state.get("llm_model") or "llama-3.1-8b-instant"), temperature=float(state.get("llm_temperature", 0.3)))

    response = _invoke_character_response(llm, _build_character_prompt(state.get("user_prompt", ""), character_names))

    generated_by_name: Dict[str, Dict[str, object]] = {}

    for character in response.characters or []:
        if not character.name:
            continue

        normalized = {
            "name": character.name,
            "age": character.age,
            "personality_traits": [str(trait) for trait in character.personality_traits],
            "appearance_description": character.appearance_description,
            "clothing": character.clothing,
            "hair_texture": character.hair_texture,
            "eye_color": character.eye_color,
            "signature_item": character.signature_item,
            "base_visual_style": character.base_visual_style,
            "reference_style": character.base_visual_style,
            "reference_image_path": character.reference_image_path,
        }
        generated_by_name[_norm_name(str(character.name))] = normalized

    missing_names: List[str] = []
    for required_name in character_names:
        key = _norm_name(required_name)
        chosen = generated_by_name.get(key)

        if not chosen:
            fuzzy_key = next(
                (
                    generated_key
                    for generated_key in generated_by_name
                    if generated_key == key or generated_key in key or key in generated_key
                ),
                None,
            )
            if fuzzy_key:
                chosen = generated_by_name[fuzzy_key]

        if not chosen:
            cached = memory_lookup.get(required_name.strip().lower())
            if isinstance(cached, dict):
                cached_candidate = {
                    "name": required_name,
                    "age": str(cached.get("age", "")),
                    "personality_traits": [str(trait) for trait in cached.get("personality_traits", [])],
                    "appearance_description": str(cached.get("appearance_description", "")),
                    "clothing": str(cached.get("clothing", "")),
                    "hair_texture": str(cached.get("hair_texture", "")),
                    "eye_color": str(cached.get("eye_color", "")),
                    "signature_item": str(cached.get("signature_item", "")),
                    "base_visual_style": str(cached.get("base_visual_style", cached.get("reference_style", ""))),
                    "reference_style": str(cached.get("base_visual_style", cached.get("reference_style", ""))),
                    "reference_image_path": str(cached.get("reference_image_path", "")),
                }
                if _has_required_character_fields(cached_candidate):
                    chosen = cached_candidate

        if not chosen:
            missing_names.append(required_name)
            continue

        if not _has_required_character_fields(chosen):
            missing_names.append(required_name)
            continue

        chosen["name"] = required_name
        characters.append(chosen)

    if missing_names:
        raise ValueError(
            "character_node could not create profiles for script characters: " + ", ".join(missing_names)
        )

    outputs_dir = ensure_outputs_dir()
    character_db_payload = {"characters": characters}
    (outputs_dir / "character_db.json").write_text(json.dumps(character_db_payload, indent=2), encoding="utf-8")

    return {"characters": characters, "status": "characters_created"}
