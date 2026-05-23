import numpy as np
import io
import soundfile as sf
import logging
from typing import Optional

logger = logging.getLogger("audio-pipeline")

class AudioPipeline:
    def __init__(self, target_sample_rate: int = 24000, target_loudness: float = -23.0, max_peak: float = 0.95, default_profile: str = "voice_agent_fast"):
        self.target_sample_rate = target_sample_rate
        self.target_loudness = target_loudness
        self.max_peak = max_peak
        self.default_profile = default_profile

    def trim_silence(self, audio: np.ndarray, threshold: float = 0.001, margin_ms: int = 100) -> np.ndarray:
        """
        Trims leading and trailing silence based on amplitude threshold,
        preserving a safety margin at both ends to protect soft phonemes.
        Supports 1D and 2D arrays (samples, channels).
        """
        if len(audio) == 0:
            return audio

        if len(audio.shape) > 1:
            amplitude = np.max(np.abs(audio), axis=1)
        else:
            amplitude = np.abs(audio)

        non_silent = np.where(amplitude > threshold)[0]
        if len(non_silent) > 0:
            margin_samples = int(margin_ms * self.target_sample_rate / 1000)
            start_idx = max(0, non_silent[0] - margin_samples)
            end_idx = min(len(audio), non_silent[-1] + 1 + margin_samples)
            return audio[start_idx:end_idx]
        return audio

    def normalize_loudness(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Applies EBU R128 integrated loudness normalization targeting -23 LUFS.
        Uses pyloudnorm.
        """
        try:
            import pyloudnorm as pyln
            
            meter = pyln.Meter(sample_rate)
            if len(audio.shape) == 1:
                audio_2d = audio[:, np.newaxis]
            else:
                audio_2d = audio

            loudness = meter.integrated_loudness(audio_2d)
            if loudness > -70.0 and not np.isnan(loudness) and not np.isinf(loudness):
                normalized = pyln.normalize.loudness(audio_2d, loudness, self.target_loudness)
                if len(audio.shape) == 1:
                    return normalized.squeeze(axis=-1)
                return normalized
        except Exception as e:
            logger.warning(f"Loudness normalization failed: {e}")
            
        return audio

    def prevent_clipping(self, audio: np.ndarray) -> np.ndarray:
        """
        Prevents digital clipping using a hybrid method of soft limiting 
        and peak scaling to keep peaks below self.max_peak while preserving loudness.
        """
        if len(audio) == 0:
            return audio
            
        peak = np.max(np.abs(audio))
        if peak <= self.max_peak:
            return audio
            
        # Use a soft knee limiter for samples exceeding 0.7
        threshold = 0.7
        mask = np.abs(audio) > threshold
        if np.any(mask):
            # Soft clip values above threshold
            sign = np.sign(audio)
            abs_val = np.abs(audio)
            # Smoothly compress values between threshold and peak to threshold -> max_peak
            scale = self.max_peak - threshold
            compressed = threshold + scale * np.tanh((abs_val - threshold) / scale)
            audio = np.where(mask, sign * compressed, audio)
            
        # If the peak still exceeds max_peak, scale slightly
        new_peak = np.max(np.abs(audio))
        if new_peak > self.max_peak:
            audio = audio * (self.max_peak / new_peak)
            
        return audio

    def match_sample_rate(self, audio: np.ndarray, current_sr: int, target_sr: int) -> tuple[np.ndarray, int]:
        """
        Resamples audio to target sample rate using librosa.
        """
        if current_sr == target_sr:
            return audio, current_sr

        try:
            import librosa
            resampled_audio = librosa.resample(audio, orig_sr=float(current_sr), target_sr=float(target_sr), axis=0)
            return resampled_audio, target_sr
        except Exception as e:
            logger.warning(f"Resampling failed: {e}")
            return audio, current_sr

    def process(self, audio: np.ndarray, current_sr: int, profile: Optional[str] = None) -> tuple[np.ndarray, int]:
        """
        Runs the optimization pipeline based on the configured profile:
        - voice_agent_fast: full suite (24kHz, trim, normalize, limit)
        - game_npc_ogg: keep pauses (no trim), normalize, limit, resample to native/target
        - high_quality_narration: no trim, no normalization, limit only, native/target sample rate
        - raw_model_output: no processing at all
        """
        active_profile = profile or self.default_profile
        
        if active_profile == "raw_model_output":
            return audio, current_sr

        # Remove DC offset first
        if len(audio) > 0:
            audio = audio - np.mean(audio, axis=0)

        # Resampling sample rate target
        target_sr = self.target_sample_rate
        if active_profile == "game_npc_ogg":
            # Prefer higher sample rate for games (44.1kHz or native)
            target_sr = max(44100, current_sr)
        elif active_profile == "high_quality_narration":
            target_sr = current_sr # Keep native

        audio, sr = self.match_sample_rate(audio, current_sr, target_sr)

        # Silence trimming
        if active_profile == "voice_agent_fast":
            audio = self.trim_silence(audio)

        # Loudness normalization
        if active_profile in ("voice_agent_fast", "game_npc_ogg"):
            audio = self.normalize_loudness(audio, sr)

        # Clipping prevention
        if active_profile in ("voice_agent_fast", "game_npc_ogg", "high_quality_narration"):
            audio = self.prevent_clipping(audio)

        return audio, sr

    def process_wav_bytes(self, wav_bytes: bytes, current_sr: int, profile: Optional[str] = None) -> tuple[bytes, int]:
        """
        Helper method to deserialize WAV bytes, process them through the pipeline, and serialize back.
        """
        try:
            audio, sr = sf.read(io.BytesIO(wav_bytes))
            processed_audio, new_sr = self.process(audio, sr, profile)
            
            buffer = io.BytesIO()
            sf.write(buffer, processed_audio, new_sr, format="WAV", subtype="PCM_16")
            return buffer.getvalue(), new_sr
        except Exception as e:
            logger.error(f"Error processing WAV bytes: {e}")
            return wav_bytes, current_sr
