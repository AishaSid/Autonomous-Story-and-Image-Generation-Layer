from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import io
from pathlib import Path
from typing import Any

import numpy as np
import requests
import wave

try:
    from elevenlabs.client import ElevenLabs
except ImportError:
    ElevenLabs = None


def _dialogue_entry_to_line(entry: Any) -> str:
    if isinstance(entry, str):
        return entry.strip()

    if isinstance(entry, dict):
        for key in ("line", "text", "dialogue", "utterance", "content"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return ""


def _get_env_value(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip().strip('"').strip("'")
    return None


def _log_audio_http_response(provider: str, response: requests.Response) -> None:
    """Log status, sanitized headers, and content type for audio API responses."""
    interesting_headers = (
        "content-type",
        "content-length",
        "x-request-id",
        "request-id",
        "cf-ray",
    )
    headers_snapshot = {
        key: response.headers.get(key)
        for key in interesting_headers
        if response.headers.get(key) is not None
    }
    content_type = response.headers.get("Content-Type", "").strip()
    print(
        f"[{provider}] response status={response.status_code}, "
        f"content_type='{content_type}', headers={headers_snapshot}"
    )


def _extract_audio_from_response(response: requests.Response, provider: str) -> tuple[bytes, str]:
    """
    Extract raw audio bytes from an API response.
    Supports binary audio/* and JSON/base64 payloads.
    Returns (audio_bytes, input_format).
    """
    content_type = response.headers.get("Content-Type", "").lower()

    if content_type.startswith("audio/"):
        audio_bytes = response.content
        input_format = content_type.split(";", 1)[0].split("/")[-1].strip() or "mpeg"
        if not audio_bytes:
            raise RuntimeError(f"{provider} returned empty binary audio body.")
        return audio_bytes, input_format

    try:
        payload = response.json()
    except ValueError as ex:
        body_preview = response.text[:500]
        raise RuntimeError(
            f"{provider} returned unsupported content type '{content_type}' and non-JSON body: {body_preview}"
        ) from ex

    mime_type = str(payload.get("mime_type") or payload.get("audio_mime_type") or "").lower().strip()
    input_format = "mpeg"
    if mime_type.startswith("audio/"):
        input_format = mime_type.split("/", 1)[1].strip() or "mpeg"

    data_node = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    base64_candidate = (
        payload.get("audio_base64")
        or payload.get("audio")
        or payload.get("base64")
        or data_node.get("audio_base64")
        or data_node.get("audio")
        or data_node.get("base64")
    )
    if not isinstance(base64_candidate, str) or not base64_candidate.strip():
        raise RuntimeError(f"{provider} JSON response did not include audio payload/base64 fields.")

    try:
        audio_bytes = base64.b64decode(base64_candidate, validate=True)
    except Exception as ex:
        raise RuntimeError(f"{provider} returned invalid base64 audio payload.") from ex

    if not audio_bytes:
        raise RuntimeError(f"{provider} returned empty decoded base64 audio.")

    return audio_bytes, input_format


def _audio_bytes_to_wav_bytes(audio_bytes: bytes, input_format: str) -> bytes:
    """Convert input audio bytes (mp3/wav/other) into WAV bytes."""
    normalized_format = (input_format or "mpeg").lower().strip()
    if normalized_format in {"wav", "x-wav"}:
        return audio_bytes

    pydub_format = "mp3" if normalized_format in {"mpeg", "mpga", "mp3"} else normalized_format
    try:
        from pydub import AudioSegment

        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=pydub_format)
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        return buffer.getvalue()
    except ImportError:
        with tempfile.NamedTemporaryFile(suffix=f".{pydub_format}", delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in_path = tmp_in.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav_path = tmp_wav.name
        try:
            subprocess.run(
                ["ffmpeg", "-i", tmp_in_path, "-y", tmp_wav_path],
                check=True,
                capture_output=True,
            )
            return Path(tmp_wav_path).read_bytes()
        finally:
            if Path(tmp_in_path).exists():
                os.unlink(tmp_in_path)
            if Path(tmp_wav_path).exists():
                os.unlink(tmp_wav_path)


def _validate_generated_wav(audio_path: Path, provider: str) -> None:
    """Ensure generated output is a playable WAV file, not empty placeholder bytes."""
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise RuntimeError(f"{provider} produced missing/empty output file: {audio_path}")

    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            frame_rate = wav_file.getframerate()
            frames = wav_file.getnframes()
        if channels <= 0 or frame_rate <= 0 or frames <= 0:
            raise RuntimeError("WAV metadata invalid (channels/rate/frames).")
    except Exception as ex:
        raise RuntimeError(f"{provider} output is not a valid playable WAV: {ex}") from ex


def _request_elevenlabs_audio(text: str, voice_id: str, api_key: str, model_id: str = "eleven_flash_v2_5") -> tuple[bytes, str]:
    """Call ElevenLabs REST API and return normalized (audio_bytes, input_format)."""
    endpoint = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    payload = {
        "text": text,
        "model_id": model_id,
    }
    print(f"[ElevenLabs] endpoint called: {endpoint}")
    print(f"[ElevenLabs] payload sent: {json.dumps(payload, ensure_ascii=True)}")

    response = requests.post(
        endpoint,
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        json=payload,
        timeout=120,
    )
    _log_audio_http_response("ElevenLabs", response)

    if response.status_code != 200:
        body_preview = response.text[:700]
        raise RuntimeError(f"ElevenLabs failed ({response.status_code}): {body_preview}")

    return _extract_audio_from_response(response, provider="ElevenLabs")


def _generate_huggingface_tts(destination: Path, speech_text: str, model: str = "espnet/kan-bayashi_ljspeech_vits") -> str:
    hf_token = _get_env_value("HF_TOKEN", "HUGGING_FACE_TOKEN", "hf_token")
    if not hf_token:
        raise RuntimeError("Hugging Face token is not configured.")

    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Accept": "audio/wav",
    }
    response = requests.post(
        url,
        headers=headers,
        json={"inputs": speech_text},
        timeout=120,
    )
    _log_audio_http_response("HuggingFace", response)

    if response.status_code != 200:
        try:
            body = response.json()
        except ValueError:
            body = response.text
        raise RuntimeError(f"Hugging Face TTS failed ({response.status_code}): {body}")

    audio_bytes, input_format = _extract_audio_from_response(response, provider="HuggingFace")
    destination.write_bytes(_audio_bytes_to_wav_bytes(audio_bytes, input_format=input_format))
    _validate_generated_wav(destination, provider="HuggingFace")
    print(f"[OK] Generated audio using Hugging Face TTS ({destination.stat().st_size} bytes)")
    return str(destination)


def _generate_pyttsx3_wav(destination: Path, speech_text: str) -> str:
    try:
        import pyttsx3
    except ImportError as ex:
        raise RuntimeError("pyttsx3 is not installed for local TTS fallback.") from ex

    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 0.9)

    destination_str = str(destination)
    engine.save_to_file(speech_text, destination_str)
    engine.runAndWait()

    if not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError("pyttsx3 failed to generate audio.")

    print(f"[OK] Generated audio using pyttsx3 local TTS ({destination.stat().st_size} bytes)")
    return str(destination)


def _generate_pyttsx3_multi_speaker(destination: Path, segments: list[tuple[str, str]]) -> str:
    """
    Generate multi-speaker audio using pyttsx3 with different voice properties per speaker.
    """
    try:
        import pyttsx3
    except ImportError as ex:
        raise RuntimeError("pyttsx3 is not installed for local TTS fallback.") from ex

    # Different voice settings for different speakers
    speaker_voices: dict[str, dict] = {}
    base_voices = [
        {"rate": 150, "volume": 0.9, "voice": None},  # Default
        {"rate": 130, "volume": 0.9, "voice": None},  # Slower (male)
        {"rate": 170, "volume": 0.9, "voice": None},  # Faster (female)
        {"rate": 140, "volume": 0.85, "voice": None}, # Medium
    ]

    def _get_voice_props(speaker: str) -> dict:
        if speaker not in speaker_voices:
            speaker_voices[speaker] = base_voices[len(speaker_voices) % len(base_voices)]
        return speaker_voices[speaker]

    # Generate each segment separately
    temp_files: list[Path] = []
    
    try:
        for i, (speaker, text) in enumerate(segments):
            # Create temp file for this segment
            tmp_file = destination.parent / f"temp_segment_{i}_{destination.stem}.wav"
            temp_files.append(tmp_file)
            
            # Get voice properties for this speaker
            voice_props = _get_voice_props(speaker)
            
            # Initialize engine for this segment
            engine = pyttsx3.init()
            engine.setProperty("rate", voice_props["rate"])
            engine.setProperty("volume", voice_props["volume"])
            
            # Try to set a different voice if available
            voices = engine.getProperty("voices")
            if voices:
                # Select voice based on speaker index
                voice_idx = len(speaker_voices) % len(voices)
                engine.setProperty("voice", voices[voice_idx].id)
            
            engine.save_to_file(text, str(tmp_file))
            engine.runAndWait()
        
        # Combine all temp files
        if temp_files:
            wav_chunks: list[bytes] = []
            for tmp_file in temp_files:
                if tmp_file.exists() and tmp_file.stat().st_size > 0:
                    wav_chunks.append(tmp_file.read_bytes())
            
            if wav_chunks:
                _stitch_wav_bytes(wav_chunks, destination)
        
        # Clean up temp files
        for tmp_file in temp_files:
            if tmp_file.exists():
                os.unlink(tmp_file)
        
        if not destination.exists() or destination.stat().st_size == 0:
            raise RuntimeError("pyttsx3 multi-speaker failed to generate audio.")
        
        print(f"[OK] Generated multi-speaker audio using pyttsx3 ({destination.stat().st_size} bytes)")
        return str(destination)
        
    except Exception as e:
        # Clean up temp files on error
        for tmp_file in temp_files:
            if tmp_file.exists():
                os.unlink(tmp_file)
        raise RuntimeError(f"pyttsx3 multi-speaker failed: {str(e)}") from e


def _generate_piper_tts(destination: Path, speech_text: str) -> str:
    """
    Generate speech using Piper TTS (local neural TTS).
    Piper is a fast, local neural TTS system.
    """
    try:
        from piper import PiperVoice
        from piper.download_voices import download_voice
    except ImportError as ex:
        raise RuntimeError("piper-tts is not installed for local TTS fallback.") from ex

    voice_name = _get_env_value("PIPER_VOICE", "PIPER_VOICE_NAME") or "en_US-lessac-medium"
    model_dir = destination.parent / "piper_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{voice_name}.onnx"
    config_path = model_dir / f"{voice_name}.onnx.json"

    try:
        if not model_path.exists() or not config_path.exists():
            print(f"Downloading Piper voice model: {voice_name}")
            download_voice(voice_name, model_dir)

        voice = PiperVoice.load(model_path, config_path=config_path, download_dir=model_dir)
        with wave.open(str(destination), "wb") as wav_file:
            voice.synthesize_wav(speech_text, wav_file)

        if not destination.exists() or destination.stat().st_size == 0:
            raise RuntimeError("Piper TTS failed to generate audio.")

        print(f"[OK] Generated audio using Piper TTS ({destination.stat().st_size} bytes)")
        return str(destination)
    except Exception as e:
        raise RuntimeError(f"Piper TTS failed: {str(e)}") from e


def _generate_piper_tts_multi_speaker(destination: Path, segments: list[tuple[str, str]]) -> str:
    """
    Generate multi-speaker audio using Piper TTS with different voice models per speaker.
    Piper needs separate voice models to sound like distinct voices.
    """
    try:
        from piper import PiperVoice
        from piper.download_voices import download_voice
    except ImportError as ex:
        raise RuntimeError("piper-tts is not installed for local TTS fallback.") from ex

    female_voice = _get_env_value("PIPER_VOICE_FEMALE", "PIPER_FEMALE_VOICE") or "en_US-lessac-medium"
    male_voice = _get_env_value("PIPER_VOICE_MALE", "PIPER_MALE_VOICE") or "en_US-ryan-medium"
    neutral_voice = _get_env_value("PIPER_VOICE", "PIPER_VOICE_NAME") or female_voice

    model_dir = destination.parent / "piper_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    def _speaker_voice_name(speaker: str, text: str) -> str:
        name = speaker.lower().strip()
        if name.startswith("ethan"):
            return male_voice
        if name.startswith("lena"):
            return female_voice
        if name == "narrator":
            return neutral_voice

        # Small heuristic fallback if the speaker name is unknown.
        return male_voice if len(text) % 2 == 0 else female_voice

    def _load_voice_model(voice_name: str) -> PiperVoice:
        model_path = model_dir / f"{voice_name}.onnx"
        config_path = model_dir / f"{voice_name}.onnx.json"

        if not model_path.exists() or not config_path.exists():
            print(f"Downloading Piper voice model: {voice_name}")
            download_voice(voice_name, model_dir)

        return PiperVoice.load(model_path, config_path=config_path, download_dir=model_dir)

    voice_cache: dict[str, PiperVoice] = {}

    def _get_voice(voice_name: str) -> PiperVoice:
        if voice_name not in voice_cache:
            voice_cache[voice_name] = _load_voice_model(voice_name)
        return voice_cache[voice_name]

    try:
        with wave.open(str(destination), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            first_chunk = True

            for speaker, text in segments:
                if not text.strip():
                    continue

                speaker_voice_name = _speaker_voice_name(speaker, text)
                try:
                    voice = _get_voice(speaker_voice_name)
                except Exception as load_error:
                    fallback_name = female_voice if speaker_voice_name != female_voice else male_voice
                    print(f"[WARN] Piper voice '{speaker_voice_name}' failed for {speaker}; trying '{fallback_name}'")
                    voice = _get_voice(fallback_name)

                for chunk in voice.synthesize(text):
                    if first_chunk:
                        wav_file.setframerate(chunk.sample_rate)
                        first_chunk = False
                    wav_file.writeframes(chunk.audio_int16_bytes)

                # 120ms silence between turns.
                current_rate = wav_file.getframerate() if not first_chunk else 22050
                silence = np.zeros(int(current_rate * 0.12), dtype=np.int16)
                wav_file.writeframes(silence.tobytes())

        if not destination.exists() or destination.stat().st_size == 0:
            raise RuntimeError("Piper TTS failed to generate audio.")

        print(f"[OK] Generated multi-speaker audio using Piper TTS ({destination.stat().st_size} bytes)")
        return str(destination)
    except Exception as e:
        raise RuntimeError(f"Piper TTS multi-speaker failed: {str(e)}") from e


def _dialogue_to_text(dialogue_beats: list[Any], scene_id: str) -> str:
    lines = [_dialogue_entry_to_line(entry) for entry in dialogue_beats]
    normalized: list[str] = []
    for line in lines:
        if not line:
            continue
        if ":" in line:
            _, text = line.split(":", 1)
            line = text.strip()
        if line:
            normalized.append(line)
    if not normalized:
        return f"Scene {scene_id} dialogue."
    return " ".join(normalized)


def _parse_dialogue_segments(dialogue_beats: list[Any]) -> list[tuple[str, str]]:
    segments: list[tuple[str, str]] = []
    for entry in dialogue_beats:
        line = _dialogue_entry_to_line(entry)
        if not line:
            continue
        if ":" in line:
            speaker, text = line.split(":", 1)
            speaker = speaker.strip() or "Narrator"
            text = text.strip()
        else:
            speaker = "Narrator"
            text = line
        if text:
            segments.append((speaker, text))
    return segments


# Predefined distinct voices for different characters
# These are stable ElevenLabs voice IDs that don't require special permissions
CHARACTER_VOICES = {
    # Explicit per-character mappings.
    "ethan": "29vDmlR7DoOmaJ0C9W6R",      # Drew
    "ethan thompson": "29vDmlR7DoOmaJ0C9W6R",
    "lena": "21m00Tcm4TlvDq8ikWAM",       # Rachel
    "lena lee": "21m00Tcm4TlvDq8ikWAM",

    # Gender hints.
    "male": "29vDmlR7DoOmaJ0C9W6R",
    "female": "21m00Tcm4TlvDq8ikWAM",

    # Default fallback.
    "narrator": "EXAVITQu4vr4xnSDxMaL",    # Bella - neutral
    "default": "EXAVITQu4vr4xnSDxMaL",
}

# Valid public ElevenLabs voices for deterministic distinct-assignment.
EXTRA_VOICES = [
    "21m00Tcm4TlvDq8ikWAM",  # Rachel
    "AZnzlk1XvdvUeBnXmlld",  # Domi
    "EXAVITQu4vr4xnSDxMaL",  # Bella
    "ErXwobaYiN019PkySvjV",  # Antoni
    "MF3mGyEYCl7XYWbV9V6O",  # Elli
    "TxGEqnHWrfWFTfGW9XjX",  # Josh
    "VR6AewLTigWG4xSOukaG",  # Arnold
    "pNInz6obpgDQGcFmaJgB",  # Adam
]


def _get_voice_id_for_character(
    character_name: str,
    character_profile: dict[str, Any] | None = None,
    gender_hint: str = "neutral",
) -> str:
    """
    Map character to ElevenLabs voice based on character name.
    
    Args:
        character_name: Name of the character (used for voice mapping)
        character_profile: Character profile dict with age, appearance, etc.
        gender_hint: "male", "female", or "neutral" (for fallback)
    
    Returns:
        ElevenLabs voice_id
    """
    if not character_name:
        return CHARACTER_VOICES["default"]
    
    # Normalize character name for lookup
    name_lower = character_name.lower().strip()
    
    # First try exact match
    if name_lower in CHARACTER_VOICES:
        return CHARACTER_VOICES[name_lower]
    
    # Try to find by first name (if full name provided)
    first_name = name_lower.split()[0] if name_lower else ""
    if first_name in CHARACTER_VOICES:
        return CHARACTER_VOICES[first_name]
    
    # Check character profile for gender if available
    if character_profile:
        appearance = str(character_profile.get("appearance_description", "")).lower()
        traits = [str(t).lower() for t in character_profile.get("personality_traits", [])]
        
        # Simple heuristic: check appearance/traits for gender indicators
        male_indicators = ["male", "man", "he", "his", "masculine", "beard", "adam's"]
        female_indicators = ["female", "woman", "she", "her", "feminine", "girl", "lady"]
        
        male_score = sum(1 for indicator in male_indicators if indicator in appearance or any(indicator in t for t in traits))
        female_score = sum(1 for indicator in female_indicators if indicator in appearance or any(indicator in t for t in traits))
        
        if male_score > female_score:
            return CHARACTER_VOICES["male"]
        elif female_score > male_score:
            return CHARACTER_VOICES["female"]
    
    # Use gender_hint as fallback
    if gender_hint.lower() in CHARACTER_VOICES:
        return CHARACTER_VOICES[gender_hint.lower()]
    
    # Hash the character name to get a consistent voice from extra voices
    name_hash = sum(ord(c) for c in name_lower)
    return EXTRA_VOICES[name_hash % len(EXTRA_VOICES)]


def _generate_fallback_wav(destination: Path, speech_text: str) -> str:
    """Generate deterministic local fallback audio when ElevenLabs is unavailable."""
    sample_rate = 22050
    # Keep fallback speech duration readable and stable.
    duration_seconds = min(max(len(speech_text) / 18.0, 1.0), 8.0)
    total_samples = int(sample_rate * duration_seconds)
    time_axis = np.linspace(0, duration_seconds, total_samples, endpoint=False)
    waveform = (
        0.20 * np.sin(2.0 * np.pi * 220.0 * time_axis)
        + 0.10 * np.sin(2.0 * np.pi * 330.0 * time_axis)
    )
    pcm = np.clip(waveform * 32767.0, -32768, 32767).astype(np.int16)

    with wave.open(str(destination), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return str(destination)


def _generate_multi_speaker_fallback_wav(destination: Path, segments: list[tuple[str, str]]) -> str:
    """Generate fallback WAV where each speaker uses a distinct tone."""
    sample_rate = 22050
    speaker_freqs: dict[str, tuple[float, float]] = {}
    base_pairs = [
        (170.0, 260.0),
        (220.0, 330.0),
        (280.0, 420.0),
        (140.0, 210.0),
    ]

    def _freqs_for_speaker(speaker: str) -> tuple[float, float]:
        if speaker not in speaker_freqs:
            speaker_freqs[speaker] = base_pairs[len(speaker_freqs) % len(base_pairs)]
        return speaker_freqs[speaker]

    pcm_parts: list[np.ndarray] = []
    for speaker, text in segments:
        duration_seconds = min(max(len(text) / 16.0, 0.6), 6.0)
        total_samples = int(sample_rate * duration_seconds)
        t = np.linspace(0, duration_seconds, total_samples, endpoint=False)
        freq1, freq2 = _freqs_for_speaker(speaker)
        waveform = 0.18 * np.sin(2.0 * np.pi * freq1 * t) + 0.09 * np.sin(2.0 * np.pi * freq2 * t)
        pcm_parts.append(np.clip(waveform * 32767.0, -32768, 32767).astype(np.int16))
        # 120ms silence between dialogue turns.
        pcm_parts.append(np.zeros(int(sample_rate * 0.12), dtype=np.int16))

    combined = np.concatenate(pcm_parts) if pcm_parts else np.zeros(sample_rate, dtype=np.int16)
    with wave.open(str(destination), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(combined.tobytes())
    return str(destination)


def _stitch_wav_bytes(chunks: list[bytes], destination: Path) -> str:
    """Append mono/stereo WAV chunks into one output file."""
    if not chunks:
        raise RuntimeError("No audio chunks to stitch.")

    with wave.open(io.BytesIO(chunks[0]), "rb") as first_wav:
        channels = first_wav.getnchannels()
        sample_width = first_wav.getsampwidth()
        frame_rate = first_wav.getframerate()
        frames = [first_wav.readframes(first_wav.getnframes())]

    for chunk in chunks[1:]:
        with wave.open(io.BytesIO(chunk), "rb") as wav_file:
            if wav_file.getnchannels() != channels or wav_file.getsampwidth() != sample_width:
                raise RuntimeError("WAV chunks have incompatible formats.")
            frames.append(wav_file.readframes(wav_file.getnframes()))

    with wave.open(str(destination), "wb") as output_wav:
        output_wav.setnchannels(channels)
        output_wav.setsampwidth(sample_width)
        output_wav.setframerate(frame_rate)
        for pcm in frames:
            output_wav.writeframes(pcm)
    return str(destination)


def voice_cloning_synthesizer(
    scene_id: str,
    dialogue_beats: list[Any],
    output_path: str,
    character_profile: dict[str, Any] | None = None,
    scene_character_profiles: dict[str, dict[str, Any]] | None = None,
) -> str:
    """
    Generate speech synthesis using ElevenLabs API.
    
    Args:
        scene_id: Unique scene identifier
        dialogue_beats: List of dialogue lines/beats
        output_path: Output WAV file path
        character_profile: Optional character profile for gender-based voice selection
    
    Returns:
        Path to generated audio file
    
    Raises:
        RuntimeError: If API call fails or file generation fails
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure output is WAV format
    if not str(destination).lower().endswith('.wav'):
        destination = destination.with_suffix('.wav')
    
    speech_text = _dialogue_to_text(dialogue_beats=dialogue_beats, scene_id=scene_id)
    segments = _parse_dialogue_segments(dialogue_beats)
    scene_character_profiles = scene_character_profiles or {}
    
    # Check for API key
    api_key = _get_env_value("ELEVEN_LABS_KEY", "ELEVEN_API_KEY", "ELEVEN_TTS_KEY")
    if not api_key:
        print("[WARN] ElevenLabs API key not set. Using local fallback audio.")
        if segments:
            print(f"Using multi-speaker fallback synthesis for {len(segments)} dialogue turns.")
            return _generate_multi_speaker_fallback_wav(destination, segments)
        return _generate_fallback_wav(destination, speech_text)
    
    try:
        # Multi-speaker synthesis: one voice per dialogue turn.
        if segments:
            wav_chunks: list[bytes] = []
            for speaker, text in segments:
                profile = scene_character_profiles.get(speaker) or character_profile or {}
                voice_id = _get_voice_id_for_character(
                    character_name=speaker,
                    character_profile=profile,
                )
                print(f"Generating speaker turn with voice '{voice_id}': {speaker}")
                audio_bytes, input_format = _request_elevenlabs_audio(
                    text=text,
                    voice_id=voice_id,
                    api_key=api_key,
                    model_id="eleven_flash_v2_5",
                )
                wav_chunks.append(_audio_bytes_to_wav_bytes(audio_bytes, input_format=input_format))
            _stitch_wav_bytes(wav_chunks, destination)
        else:
            # Single-speaker fallback path.
            voice_id = _get_voice_id_for_character(
                character_name=scene_id,
                character_profile=character_profile,
            )
            audio_bytes, input_format = _request_elevenlabs_audio(
                text=speech_text,
                voice_id=voice_id,
                api_key=api_key,
                model_id="eleven_flash_v2_5",
            )
            destination.write_bytes(_audio_bytes_to_wav_bytes(audio_bytes, input_format=input_format))
        
        _validate_generated_wav(destination, provider="ElevenLabs")
        
        print(f"[OK] Generated audio for {scene_id} ({destination.stat().st_size} bytes)")
        return str(destination)
        
    except Exception as e:
        error_msg = f"ElevenLabs TTS failed for scene {scene_id}: {str(e)}"
        print(f"[ERROR] {error_msg}")

        # Try Piper TTS first (local neural TTS - fast and high quality)
        try:
            print("[WARN] Falling back to Piper TTS (local neural TTS)...")
            if segments:
                return _generate_piper_tts_multi_speaker(destination, segments)
            return _generate_piper_tts(destination, speech_text)
        except Exception as piper_error:
            print(f"[ERROR] Piper TTS fallback failed: {piper_error}")

        # Hugging Face fallback if available
        hf_token = _get_env_value("HF_TOKEN", "HUGGING_FACE_TOKEN", "hf_token")
        if hf_token:
            try:
                print("[WARN] Falling back to Hugging Face TTS...")
                return _generate_huggingface_tts(destination, speech_text)
            except Exception as hf_error:
                print(f"[ERROR] Hugging Face TTS fallback failed: {hf_error}")

        try:
            print("[WARN] Falling back to local pyttsx3 TTS...")
            if segments:
                # For multi-speaker, use pyttsx3 with different voices
                return _generate_pyttsx3_multi_speaker(destination, segments)
            return _generate_pyttsx3_wav(destination, speech_text)
        except Exception as py_error:
            print(f"[ERROR] pyttsx3 fallback failed: {py_error}")

        if segments:
            print(f"Using multi-speaker fallback synthesis for {len(segments)} dialogue turns.")
            return _generate_multi_speaker_fallback_wav(destination, segments)

        return _generate_fallback_wav(destination, speech_text)
