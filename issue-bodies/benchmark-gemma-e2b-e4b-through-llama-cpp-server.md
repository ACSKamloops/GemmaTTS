# Benchmark Gemma E2B/E4B through llama.cpp server

## Summary
Measure real TTFT, tokens/sec, memory, CPU/GPU load, context-size behavior, and queue behavior on the target hardware.

## Implementation notes
Use llama.cpp server with an OpenAI-compatible client. Run short dialogue, paragraph, and structured JSON workloads.

## Acceptance criteria
- [ ] Benchmark Gemma E2B Q4_0 and E4B Q4_0 if available.
- [ ] Record base model load memory, KV-cache growth, TTFT, tokens/sec, and max stable context.
- [ ] Document fallback policy when E4B exceeds memory or latency budget.

## References
- Gemma memory table: https://ai.google.dev/gemma/docs/core
- llama.cpp server README: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
