import io
import json
import os
import subprocess
import wave
from pathlib import Path
from typing import Optional
import soundfile as sf

def get_audio_duration_ms(audio_data: bytes, format: str) -> int:
    """
    Query the duration of audio data in milliseconds.
    Uses wave and soundfile for WAV/FLAC, and ffprobe for MP3/OGG.
    """
    fmt = format.lower().strip()
    
    if fmt == "wav":
        try:
            with wave.open(io.BytesIO(audio_data), "rb") as w:
                return int((w.getnframes() / w.getframerate()) * 1000)
        except Exception:
            pass

    if fmt in ("wav", "flac"):
        try:
            data, samplerate = sf.read(io.BytesIO(audio_data))
            return int((len(data) / samplerate) * 1000)
        except Exception:
            pass

    # Use ffprobe for compressed/streamed audio formats
    import tempfile
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_path = Path(temp_file.name)
            
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(temp_path)
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        metadata = json.loads(result.stdout)
        duration_seconds = float(metadata.get("format", {}).get("duration", 0.0))
        if duration_seconds > 0.0:
            return int(duration_seconds * 1000)
            
    except Exception:
        pass
    finally:
        if temp_path and temp_path.exists():
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    # Fallback rough estimate for PCM or when probing fails
    return int(len(audio_data) / 48)
