# Choose default architecture: Python FastAPI orchestrator + llama.cpp + Kokoro

## Summary
Choose the initial implementation stack for the generic project.

## Acceptance criteria
- [ ] FastAPI orchestrator is selected for Python-native TTS libraries.
- [ ] llama.cpp server is selected as the LLM boundary.
- [ ] Kokoro is selected as primary TTS; Piper as persistent fallback.
- [ ] The design allows replacing LLM and TTS providers through interfaces.

## References
- FastAPI WebSockets docs: https://fastapi.tiangolo.com/advanced/websockets/
- llama.cpp server README: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
