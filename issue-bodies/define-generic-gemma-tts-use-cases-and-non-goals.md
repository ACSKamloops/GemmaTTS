# Define generic Gemma+TTS use cases and non-goals

## Summary
Define the generic product target independent from Arma or MeshedServerTool: local/hosted Gemma text generation plus server-side TTS with reusable output transports.

## Acceptance criteria
- [ ] Use cases include desktop assistant, game/NPC dialogue, batch narration, API service, and local accessibility readout.
- [ ] Non-goals include client-side LLM/TTS requirement, arbitrary URL/file playback, and public internet exposure by default.
- [ ] Each use case has latency, quality, and deployment constraints documented.

## References
- Gemma core docs: https://ai.google.dev/gemma/docs/core
- Kokoro model card: https://huggingface.co/hexgrad/Kokoro-82M
