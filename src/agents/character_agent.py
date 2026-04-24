from __future__ import annotations

import json
import re
from typing import Any, Dict
from state import State

from .common import build_llm_client, ensure_outputs_dir

def character_node(state: State) -> Dict[str, Any]:
    print("--- GENERATING CHARACTER PROFILES ---")
    
    script_characters = state.get("script", {}).get("characters", [])
    if not script_characters:
        script_characters = [{"name": "Lead", "appearance_description": "Protagonist"}]

    llm_provider = state.get("llm_provider", "groq") 
    llm_model = state.get("llm_model", "llama-3.3-70b-versatile")
    
    client = build_llm_client(provider=llm_provider, model=llm_model)
    final_characters = []

    for char in script_characters:
        name = char.get("name", "Unknown")
        print(f"Expanding profile for: {name}")
        
        prompt = f"""
        You are a character designer. Create a detailed visual profile for the character: '{name}'.
        Initial description: {char.get('appearance_description', 'A mysterious traveler')}
        
        Return ONLY a JSON object with these exact keys:
        {{
            "name": "{name}",
            "age": "string",
            "personality_traits": ["list"],
            "appearance_description": "physical build and face details",
            "clothing": "detailed clothing description",
            "hair_texture": "hair style",
            "eye_color": "eye color",
            "signature_item": "one unique item",
            "base_visual_style": "{char.get('reference_style', 'Cinematic 35mm film')}"
        }}
        """

        try:
            # CORRECT LangChain execution
            response = client.invoke(prompt)
            raw_content = response.content
            
            # Clean markdown code blocks if the LLM adds them
            clean_json = re.sub(r"```json\n?|\n?```", "", raw_content).strip()
            char_data = json.loads(clean_json)
            
            char_data["name"] = name  # Force exact match
            final_characters.append(char_data)
            
        except Exception as e:
            print(f"Error expanding character {name}: {e}")
            final_characters.append({
                "name": name,
                "age": "Unknown",
                "personality_traits": char.get("personality_traits", []),
                "appearance_description": char.get("appearance_description", "Detailed"),
                "clothing": "Standard gear",
                "hair_texture": "Natural",
                "eye_color": "Brown",
                "signature_item": "A mysterious locket",
                "base_visual_style": char.get("reference_style", "Cinematic 35mm film")
            })

    # Save to disk for resume.py
    outputs_dir = ensure_outputs_dir()
    (outputs_dir / "character_db.json").write_text(
        json.dumps({"characters": final_characters}, indent=2), encoding="utf-8"
    )

    return {"characters": final_characters, "status": "characters_generated"}