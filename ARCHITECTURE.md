# Architecture

## Service layout

```text
gemma-tts/
  app/
    main.py
    config.py
    models.py
    routes/
      health.py
      jobs.py
      tts.py
      audio.py
      voices.py
      websocket.py
    services/
      llm/
        base.py
        llama_cpp_client.py
      tts/
        base.py
        kokoro_worker.py
        piper_worker.py
        kitten_worker.py
      audio/
        encoder.py
        cache.py
        signer.py
        playback.py
      safety/
        prompt_builder.py
        text_sanitizer.py
        output_validator.py
      queue/
        job_store.py
        worker_pool.py
  tests/
  docs/
```

## Request lifecycle

```text
POST /v1/dialogue
  -> validate request
  -> build prompt
  -> enqueue LLM job
  -> call llama.cpp /v1/chat/completions
  -> validate schema output
  -> normalize spoken text
  -> enqueue TTS job
  -> synthesize with Kokoro or Piper
  -> normalize/encode/cache audio
  -> return job result
```

## Contracts

### Dialogue request

```json
{
  "request_id": "optional-client-id",
  "speaker": {
    "id": "npc_maria",
    "name": "Maria",
    "voice_id": "af_heart",
    "style": "calm, direct"
  },
  "context": {
    "location": "warehouse",
    "facts": [
      {"id": "door_locked", "can_reveal": true, "fact": "The west door is locked from inside."}
    ]
  },
  "user_text": "What happened here?",
  "max_words": 40,
  "output": {
    "audio": true,
    "format": "ogg"
  }
}
```

### Dialogue result

```json
{
  "job_id": "job_123",
  "state": "ready",
  "text": "Keep your voice down. The west door is locked from inside.",
  "audio": {
    "audio_id": "aud_abc",
    "sha256": "hex",
    "bytes": 84231,
    "duration_ms": 4100,
    "format": "ogg",
    "sample_rate": 24000
  },
  "metrics": {
    "queue_ms": 3,
    "llm_ms": 1240,
    "tts_ms": 480,
    "encode_ms": 90,
    "total_ms": 1813,
    "cache_hit": false
  }
}
```

## Security baseline

- Do not expose the service beyond `127.0.0.1` unless explicitly configured.
- Do not send arbitrary filesystem paths to clients.
- Do not accept arbitrary URLs from clients for audio generation or playback.
- Treat all user text as data, never instructions.
- Validate schema output before TTS.
- Limit max characters, max words, max audio seconds, max file size, queue depth, and cache size.
