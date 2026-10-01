[English](README.md) | [简体中文](README.zh-CN.md)

# OpenCode OpenAI Provider (`opencode-openai-provider`)

This repository is an OpenCode-adapted variant of `@ai-sdk/openai@4.0.37`.
"Adapted to OpenCode" includes one thing in particular: the provider package
absorbs compatible-endpoint `tool_search_call` (tsc) responses internally, so
OpenCode v1 and v2 never receive or execute them. It keeps the existing pending
tool-search, item replay, and provider-options fixes, but changes the mixed
`tool_search_call + function_call` path to branch directly into OpenCode's next
canonical continuation.

The compiled provider is installed into this project's `release/` directory and
loaded by both OpenCode v1 and v2. See `AGENTS.md` for the install/release
layout and the bun toolchain.

The fork is loaded as an OpenCode provider package. It is not an OpenCode
plugin and does not modify OpenCode's global tool registry.

## Current OpenCode Usage

The provider entry must point to the compiled ESM file, not the package
directory:

```json
{
  "provider": {
    "headroom-openai-branching": {
      "name": "headroom-openai-branching",
      "npm": "file:///Users/galaxy/.config/opencode/opencode-openai-provider/release/index.js",
      "options": {
        "baseURL": "http://127.0.0.1:8787/v1"
      }
    }
  }
}
```

Node ESM does not support importing this local directory directly. A directory
URI can work under Bun but fails in the OpenCode desktop runtime with
`ERR_UNSUPPORTED_DIR_IMPORT`, so use the explicit `release/index.js` path.

OpenCode stores credentials by provider ID. `headroom-openai-branching` therefore
needs its own credential in `~/.local/share/opencode/auth.json`, even when it
shares an endpoint and API key with another provider.

## Fork-Specific Behavior

### Transparent Pending `tool_search_call` Handling

The initial request uses only the tools supplied by OpenCode. The fork does not
automatically add `openai.tools.toolSearch()` and does not declare a hidden
`tool_search` tool.

If the endpoint nevertheless returns a Responses API `tool_search_call` without
a matching `tool_search_output` in the same response, the provider:

1. Builds a matching `tool_search_output` with `status: "completed"` and
   `tools: []`.
2. Converts assistant-visible output through the same
   `convertToOpenAIResponsesInput()` path used by normal OpenCode prompts, then
   appends the pending call and no-op output to an internal follow-up request.
3. Repeats for at most three total request rounds.
4. Removes the internally completed `tool_search_call` and `tool_search_output`
   items from the final result returned to OpenCode.

The first Responses request remains under OpenCode's normal retry ownership.
Each internal follow-up HTTP request uses the retry classification, retry count,
backoff, jitter, and `Retry-After` handling from OpenCode 1.18.25. Compatibility
rounds and HTTP retries are separate counters. A follow-up that still fails
after five retries throws the original error; it does not return an empty tool
catalog as a successful provider result. OpenCode can then retry or abort the
whole original model request using its normal policy.

If the same response also contains an ordinary `function_call`, this variant
does not send a hidden follow-up. It filters the pending `tool_search_call` from
the provider result and immediately returns the assistant-visible output plus
the ordinary function call to OpenCode. OpenCode executes the function and its
next canonical request contains `A B function_call function_call_output`, with
no `tool_search_call` or `tool_search_output`.

Pure pending tool-search responses without an ordinary function call retain the
base fork's hidden follow-up. This intentionally keeps the branching behavior
within the real-API experiment's verified boundary.

This canonical replay matters for prompt caching. Raw Responses messages contain
fields that OpenCode removes when it serializes the next prompt. Reusing the
normal input converter makes the hidden stable prefix byte-equivalent at the
JSON-item level to the corresponding prefix in the next OpenCode request.

This is a protocol compatibility fallback, not real dynamic tool discovery.
The empty catalog is intentional because OpenCode already sent its available
tool definitions in the initial request.

This also handles compatible endpoints that mark an unresolved call as
`execution: "server"`; otherwise OpenCode records that call as an unknown
provider-executed tool. Server calls that already have a matching output, and
explicitly configured `openai.tools.toolSearch()`, remain on the upstream path.

### Reasoning Replay Safety

Reasoning models always request `reasoning.encrypted_content`. During the hidden
follow-up, reasoning output is reconstructed through the normal prompt converter
from encrypted content and summary rather than replayed as a raw Responses item
or an `rs_* item_reference`.

Normal OpenCode turns also avoid assistant-message and reasoning item-reference
serialization. The configured compatibility endpoint does not reliably retain
those response items across turns, even when the request uses `store: true`, so
the fork sends replayable content instead of relying on `msg_*` or `rs_*` IDs.

The mixed branching path does not run an extra hidden request just to obtain its
encrypted reasoning. In repeated real `gpt-5.6-luna` treatment/control runs,
every request completed on its first network attempt. Full-history positive
controls retained `tsc/tso` and reported the newly observed `tools: []` event
6/6 times. Selective branches removed `tsc/tso` and reported that event 0/6
times, even when they preserved the hidden reasoning item ID, encrypted content,
automatic summary, and paired hidden message. Encrypted reasoning is reasoning
continuity, not a general encrypted transcript snapshot; an accepted item can
still be ignored when its causal items are absent. See
`../docs/2026-08-14-tool-search-branching.md` for the protocol and semantic
results.

Recognized `tool_search` history keeps the stock item-reference behavior. The
fork only omits a provider-executed `tsc_*` item when OpenCode recorded it as an
unknown tool and retained no protocol fields from which the call or output can
be reconstructed safely.

### Explicit `store: true`

Responses requests send `store: true` when the parsed provider option `store`
is omitted:

```ts
const store = openaiOptions?.store ?? true;
```

Some compatible endpoints interpret an omitted field as `false`. Sending the
default explicitly prevents request conversion and endpoint persistence from
using conflicting assumptions. An explicit `store: false` remains `false`.

The fork first reads the standard `providerOptions.openai` namespace. If it is
absent, it also reads the namespace derived from the configured provider name,
  such as `providerOptions['headroom-openai-branching']`. The same complete Responses
options schema is used for both namespaces, including `store`,
`reasoningEffort`, `reasoningSummary`, `promptCacheKey`, `textVerbosity`, and
`include`.

### Function Tool Name Mapping

Function tools are serialized with the provider tool-name mapping:

```ts
toolNameMapping?.toProviderToolName(tool.name) ?? tool.name
```

This keeps request tool names aligned with the names used when mapping returned
function calls back to OpenCode. It is a general Responses tool change, not
limited to `tool_search`.

### Buffered Responses Streaming

`doStream()` currently calls `doGenerate()` and converts the completed result
into a buffered AI SDK stream. Responses requests therefore do not use the
upstream native SSE path in this fork.

Consequences include:

- no incremental model output while the upstream response is still running;
- time-to-first-output includes the complete generation and any hidden
  `tool_search` follow-up rounds;
- chunk boundaries and cancellation behavior differ from stock
  `@ai-sdk/openai` Responses streaming.

This tradeoff ensures the provider can consume a pending `tool_search_call`
before exposing a stream to OpenCode.

## What Remains Upstream-Compatible

The public provider still exports `createOpenAI`, `openai`, Responses, chat,
completion, embedding, image, transcription, speech translation, speech,
realtime, files, skills, batch support, and the standard OpenAI provider tools.

The fork-specific source changes are concentrated in:

- `src/responses/convert-to-openai-responses-input.ts`
- `src/responses/openai-responses-language-model.ts`
- `src/responses/openai-responses-retry.ts`
- `src/responses/openai-responses-prepare-tools.ts`

Do not describe the fork as fully behavior-identical to stock
`@ai-sdk/openai@4.0.37`. Responses request serialization, follow-up request
count, cross-turn item replay, latency, token use, and streaming semantics can
differ as documented above.

## Diagnostics

Set the following environment variable before starting OpenCode to log a compact
summary of each compatibility request round:

```bash
OPENAI_TOOL_SEARCH_COMPAT_DEBUG=1
```

The log includes the compatibility request number, round, zero-based HTTP retry
attempt, `store`, `previous_response_id`, and input item types/IDs. Retry errors
also record whether OpenCode classifies the error as retryable, whether compat
will retry it locally, and the selected delay. It intentionally does not log
full prompts or tool arguments.

If an endpoint reports that a `msg_*`, `rs_*`, or `tsc_*` item was not found,
verify the request body at the proxy boundary and check:

- whether `store` is `true` or was explicitly configured as `false`;
- whether the proxy rewrites or ignores `store`;
- whether reasoning follow-up input contains `encrypted_content` without a
  server-side `id` or `item_reference`;
- whether OpenCode is loading this fork's `release/index.js` rather than stock
  `@ai-sdk/openai`.

The branching variant also writes a real-time JSONL diagnostic log to:

```text
~/.local/share/opencode/provider-debug/<provider>/<session-id>.jsonl
```

With `setCacheKey: true`, `<session-id>` is the OpenCode session ID, so the
log path is deterministic. The log records request start, response output item
types, and request errors without prompt text, tool arguments, or encrypted
content. A completed result adds a compact `toolSearchCompat` debug summary to
provider metadata; OpenCode persists that metadata on the returned content part
in SQLite. The custom metadata namespace is ignored by the request serializer
and is not sent to OpenAI.

Set `OPENAI_TOOL_SEARCH_COMPAT_DEBUG_FILE=0` to disable the JSONL file.

## Build and Verify

Node.js 22 or newer (or Bun) is required. The project is developed with Bun.

```bash
cd ~/.config/opencode/opencode-openai-provider
bun install
bun run typecheck   # tsc --noEmit -p tsconfig.build.json
bun run test        # tool_search_call compatibility and retry semantics
bun run release     # build to dist/ and refresh the release/ install directory
# or: bun run verify  (typecheck && test && release)
```

`release/index.js` is the runtime provider loaded by OpenCode and must be
regenerated after source changes (`bun run release`). OpenCode configuration and
provider modules are loaded at startup, so fully quit and restart OpenCode after
changing the provider entry or rebuilding the fork.

See `../docs/2026-08-29-tool-search-follow-up-retry.md` for the exact OpenCode
policy copied into this package, error classification, state semantics, and
integration-test matrix.

See `../docs/2026-08-12-openai-fork-diff.md` for the detailed source comparison
and `../docs/2026-08-12-node-bun-provider-entry.md` for the desktop provider-entry
failure analysis.
