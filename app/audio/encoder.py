import subprocess
import tempfile
import os
from pathlib import Path
from typing import Set

ALLOWED_FORMATS: Set[str] = {"ogg", "mp3", "wav", "pcm"}

def encode_audio(input_wav_path: Path, output_format: str) -> bytes:
    """
    Encodes an input WAV file to OGG, MP3, or PCM format using ffmpeg.
    Args:
        input_wav_path: Path to local source WAV file.
        output_format: Destination format string (ogg, mp3, wav, pcm).
    """
    fmt = output_format.lower().strip()
    if fmt not in ALLOWED_FORMATS:
        raise ValueError(f"Format '{fmt}' is not supported. Allowed: {ALLOWED_FORMATS}")
        
    if not input_wav_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_wav_path}")
        
    if fmt == "wav":
        return input_wav_path.read_bytes()
        
    with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as temp_out:
        temp_out_path = Path(temp_out.name)
        
    try:
        # Base command configuration
        cmd = ["ffmpeg", "-y", "-i", str(input_wav_path)]
        
        if fmt == "ogg":
            cmd.extend(["-c:a", "libvorbis", "-q:a", "4"])
        elif fmt == "mp3":
            cmd.extend(["-c:a", "libmp3lame", "-q:a", "2"])
        elif fmt == "pcm":
            # Raw signed 16-bit Little Endian PCM
            cmd.extend(["-f", "s16le", "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1"])
            
        cmd.append(str(temp_out_path))
        
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        return temp_out_path.read_bytes()
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg encoding failed: {e.stderr}")
    finally:
        if temp_out_path.exists():
            try:
                os.unlink(temp_out_path)
            except OSError:
                pass
