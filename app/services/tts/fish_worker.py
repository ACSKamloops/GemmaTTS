import os
import io
import soundfile as sf
import torch

class FishWorker:
    def __init__(self, llama_path: str = "models/fish_audio/gpt", decoder_path: str = "models/fish_audio/decoder"):
        self.llama_path = llama_path
        self.decoder_path = decoder_path
        self.text2semantic = None
        self.semantic2audio = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        if self.text2semantic is None:
            if not os.path.exists(self.llama_path) or not os.path.exists(self.decoder_path):
                raise FileNotFoundError("Fish Audio gpt or decoder checkpoints not found.")
                
            from fish_speech.models.text2semantic.inference import TextToSemanticInference
            from fish_speech.models.vqgan.inference import SemanticToAudioInference
            
            self.text2semantic = TextToSemanticInference(self.llama_path, device=self.device)
            self.semantic2audio = SemanticToAudioInference(self.decoder_path, device=self.device)

    def synthesize(self, text: str, voice_id: str = "default") -> tuple[bytes, int]:
        self.load()
        
        # Dual-stage synthesis: text -> semantic codes -> waveform
        semantic_tokens = self.text2semantic.generate(text)
        waveform = self.semantic2audio.decode(semantic_tokens)
        waveform = waveform.cpu().numpy().squeeze()
        
        sample_rate = 44100  # Fish Speech native sample rate
        
        buffer = io.BytesIO()
        sf.write(buffer, waveform, sample_rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue(), sample_rate
