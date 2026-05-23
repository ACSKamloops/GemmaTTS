# Generic Gemma + TTS Project Roadmap

This package lays out a reusable Gemma + TTS project independent from MeshedServerTool or Arma.

## Baseline

```text
User/application request
  -> orchestration service
  -> Gemma via llama.cpp server
  -> text safety/normalization
  -> Kokoro primary TTS
  -> Piper persistent fallback
  -> audio normalization/encoding/cache
  -> output adapter
```

## Core principles

```text
LLM inference: server/local-host side
TTS synthesis: server/local-host side
Client local models: not required
Output transport: selectable
Audio assets: hash-addressed, signed when exposed over HTTP
Default binding: 127.0.0.1 unless explicitly serving a LAN/client endpoint
```

## Recommended stack

```text
Orchestrator: Python + FastAPI
LLM boundary: llama.cpp server using OpenAI-compatible API
Primary TTS: Kokoro 82M
Fallback TTS: Piper persistent HTTP/server mode
Audio tools: ffmpeg, soundfile, numpy
Storage: local cache initially; SQLite/PostgreSQL optional for job history
Tests: pytest
CI: GitHub Actions
```

## Modes

| Mode | Output | Best use |
|---|---|---|
| text-only | text/JSON | fast assistants, debugging, games with subtitles |
| batch TTS | WAV/OGG/MP3 files | narration, asset generation |
| local speaker | PCM/device playback | desktop assistant |
| signed audio API | HTTP-served files | remote client playback |
| PCM streaming | chunked/binary stream | low-latency voice assistant |
| generic game/NPC adapter | text + audio metadata | engine-specific integration later |

## Files

```text
roadmap-data.json            machine-readable roadmap
issues.csv                   issue import summary
issue-bodies/                individual issue bodies
create_github_roadmap.py     creates labels/milestones/issues/project via gh CLI
.github/workflows/           starter CI workflows
```

## Run GitHub setup

```bash
gh auth login
gh auth refresh -s repo -s project
python3 create_github_roadmap.py --repo OWNER/REPO
```

## Key references

- Gemma docs: https://ai.google.dev/gemma/docs/core
- llama.cpp server: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- Kokoro model card: https://huggingface.co/hexgrad/Kokoro-82M
- Piper CLI/server note: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/CLI.md
- FastAPI WebSockets: https://fastapi.tiangolo.com/advanced/websockets/
