from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
import os
import secrets

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # API Server Config
    host: str = "127.0.0.1"
    port: int = 8000
    
    # LLM Settings
    llama_cpp_url: str = "http://localhost:8080"
    gemma_model_name: str = "gemma-2b"
    
    # TTS Settings
    kokoro_url: str = "http://localhost:8888"
    piper_url: str = "http://localhost:10200"
    
    # Security & Signing
    secret_key: str = Field(default_factory=lambda: os.getenv("SECRET_KEY", secrets.token_hex(32)))
    signed_url_expiry_seconds: int = 300 # 5 minutes default
    
    # Audio Cache settings
    audio_cache_dir: Path = Path("public/data/audio_cache").resolve()
    max_cache_size_bytes: int = 50 * 1024 * 1024 # 50 MB default cache limit
    max_file_size_bytes: int = 5 * 1024 * 1024 # 5 MB max per file limit
    
    # Safety Limits
    max_text_chars: int = 1000
    max_text_words: int = 150

settings = Settings()
