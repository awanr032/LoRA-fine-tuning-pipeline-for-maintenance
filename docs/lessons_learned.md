# Lessons learned (debugging log)

Each entry: symptom → diagnosis → fix → lesson.

## 1. "Invalid JSON" on 40 of 50 pilot texts (DeepSeek)
- **Symptom:** empty answers.
- **Diagnosis:** `finish_reason: length`, `reasoning_tokens: 1000`: the model's hidden reasoning used the whole 1,000-token limit before writing any JSON.
- **Fix:** raised the limit to 4,000 and retried only truncated texts with 16,000; recorded reasoning tokens; robust JSON parsing.
- **Lesson:** with reasoning models, check `finish_reason` and reasoning-token counts before blaming the prompt.

## 2. Cost higher than estimated
- **Diagnosis:** reasoning tokens are billed as output (~200x the price of cached input); truncated attempts are paid for and then retried.
- **Fix:** measured real cost per text ($0.0015), used prompt caching and off-peak pricing, then moved to self-hosting.
- **Lesson:** estimate cost from a measured pilot, not from assumptions about output length.

## 3. 429 errors: concurrency limit tied to account balance
- **Fix:** reduced parallel requests; later self-hosted.

## 4. Same model name, different model
- `deepseek-flash` pointed to a new model version from 10 Sept 2026.
- **Fix:** every output line records model version and date.
- **Lesson:** pin versions; log them with every result.

## 5. vLLM crashed at startup
- **Diagnosis:** PyTorch built for CUDA 13 vs Colab's pre-installed TorchAudio built for CUDA 12.8.
- **Fix:** uninstall TorchAudio (not needed).

## 6. Pilot took over an hour
- **Diagnosis:** requests sent one at a time; vLLM's batching unused.
- **Fix:** send requests in parallel (thread pool); minutes instead of hours.

## 7. "Connection error" on every request
- **Diagnosis:** (a) stopping a notebook cell also stopped a server started from the notebook; (b) `pkill -f "vllm serve"` did not kill the engine process (`EngineCore`), which kept ~51 GB of GPU memory, so the new server failed to start; (c) the cleanup command also killed its own shell, skipping the wait.
- **Fix:** start the server with `setsid`; kill both processes using `[v]llm` / `[E]ngineCore` patterns; wait in Python until GPU memory is actually free; check server health before labelling.

## 8. Request timeouts with 48 parallel requests and thinking on
- **Diagnosis:** long reasoning exceeded the client's default timeout; each timeout discarded the work done and restarted.
- **Fix:** longer timeout, no hidden client retries, and thinking mode off after measuring its effect on the pilot (-2 points, much faster).

## 9. Colab disconnected during evaluation
- **Fix:** generation now saves after every batch and resumes; trained adapters live on Drive, so no retraining was needed.
