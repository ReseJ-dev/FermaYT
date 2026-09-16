# Visual Planning provider request safety

Reviewed against the official provider documentation on 2026-09-15.

| Provider endpoint | Response/request identifier | Idempotency guarantee | Status lookup | Server-side cancel |
| --- | --- | --- | --- | --- |
| Moonshot/Kimi `POST /v1/chat/completions` | The synchronous response contains a completion `id`; HTTP request IDs may also be returned | No idempotency-key contract is documented for this endpoint | No lookup operation is documented for a synchronous chat completion | No cancellation operation is documented for a synchronous chat completion |
| DashScope OpenAI-compatible `POST /chat/completions` | Responses expose completion/request identifiers depending on API mode | No idempotency-key contract is documented for the compatible synchronous endpoint | No lookup operation is documented for a synchronous chat completion | No cancellation operation is documented for a synchronous chat completion |

Moonshot documents create/retrieve/cancel operations for its separate Batch API. Those
semantics do not apply to a synchronous Chat Completions POST. DashScope also has async
task APIs for some model families; they do not establish cancellation or idempotency for
the OpenAI-compatible synchronous Chat Completions endpoint used by FermaYT.

Consequently FermaYT does not send an invented idempotency header and does not claim that
closing the local HTTPS connection cancels provider execution. Once dispatch begins, a
timeout, network interruption, process restart, or local cancellation is conservatively
classified as billing/execution unknown and cannot trigger an automatic resend.

The Kimi readiness check uses `GET /v1/models`; it does not create a Chat Completion and
does not consume the two-attempt paid planning budget.

Official references:

- Moonshot/Kimi Chat Completions: <https://platform.moonshot.ai/docs/api/chat>
- Moonshot/Kimi Batch cancellation (separate API): <https://platform.moonshot.ai/docs/api/batch-cancel>
- Alibaba Cloud Model Studio OpenAI Chat Completions compatibility: <https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions>
