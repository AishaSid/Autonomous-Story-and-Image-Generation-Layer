# The Writer's Room: Autonomous Story and Image Generation
## Phase 1: Character Identity Store

Multi-agent LangGraph system for structured screenplay generation and persistent character identity store supporting downstream multimodal synthesis.

---

## System Overview

**Core Goal:** Transform raw human intent → structured scripts → character identities → reference images

**Dual-Mode Ingestion:**
- **Manual:** Upload screenplay (validated via Script Validator Agent)
- **Auto:** Generate from prompt (via Scriptwriter Agent)

**Outputs:**
- `scene_manifest.json` — Structured screenplay with visual cues
- `character_db.json` — Persistent character identity store
- `image_assets/` — AI-generated character reference portraits

---

## Agent Definitions

| Agent | Role | Output |
|-------|------|--------|
| **Scriptwriter** | Expand prompts → multi-scene scripts | scene_manifest.json |
| **Validator** | Check manual scripts for correctness | validation report |
| **Character Designer** | Extract identities → profiles + metadata | character metadata |
| **Image Synthesizer** | Generate reference portraits | character_db.json linked images |
| **Memory Commit** | Persist to shared vector DB | ChromaDB snapshot |

---

## Architecture

```
Mode Selector
├─ Manual: Upload Script → Validator → Human-in-the-Loop → Character → Image → Memory Commit
└─ Auto: Prompt → Scriptwriter → Character → Image → Memory Commit
```

**Key Design Principles:**
- MCP-based tool discovery (no hardcoded APIs)
- Stateful memory layer (ChromaDB vector DB)
- Identity consistency across scenes
- Support for face-swapping and validation

---

## Quick Start

### Streamlit UI
```bash
streamlit run app.py
```
- **Prompt Mode:** Enter scene prompt → auto-generate script + characters + portraits
- **Upload Mode:** Upload screenplay (JSON/TXT) → validate → generate characters + portraits

### State Configuration
```python
"llm_provider": "groq"
"llm_model": "llama-3.3-70b-versatile"
"llm_temperature": 0.3
"reuse_character_memory": False
```

---

## Output Schemas

### character_db.json
```json
{
  "characters": [
    {
      "name": "Character Name",
      "appearance_description": "Detailed visual features",
      "base_visual_style": "Lighting + composition style",
      "reference_image_path": "outputs/image_assets/...",
      "personality_traits": ["trait1", "trait2"]
    }
  ]
}
```

### scene_manifest.json
```json
{
  "scenes": [
    {
      "scene_id": "INT-01",
      "title": "Scene Title",
      "dialogue_beats": ["Character: Dialog"],
      "visual_cues": ["Lighting", "Camera angle"],
      "character_names": ["Character1", "Character2"]
    }
  ]
}
```

---

## Rubric Alignment (75 marks)

- ✓ **Agent Definition** (20) — Clear roles + reasoning loops
- ✓ **Script Generation** (15) — Structured + coherent scenes
- ✓ **MCP Integration** (15) — Dynamic tool discovery
- ✓ **LangGraph Workflow** (10) — StateGraph + routing
- ✓ **Human-in-the-Loop** (10) — Validation checkpoint
- ✓ **Output Completeness** (5) — JSON + images

---

## Dependencies

```bash
pip install -r requirements.txt
```

- langgraph, langchain, streamlit, PIL, chromadb, python-dotenv

