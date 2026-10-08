[English](README.md) | [简体中文](README.zh-CN.md)

# OpenCode OpenAI Provider（`opencode-openai-provider`）

本仓库是 `@ai-sdk/openai@4.0.37` 的 **OpenCode 适配版**。“适配 OpenCode” 的核心之一，就是由 provider 包**在内部吸收**兼容端点返回的 `tool_search_call`（简称 `tsc`）响应，使 OpenCode v1 与 v2 **永远不会收到、也不会执行**它。本 fork 保留了既有的 pending tool-search、item replay 与 provider-options 修正，但把 `tool_search_call + function_call` 混合路径改为直接分支到 OpenCode 下一轮的 canonical continuation。

编译产物安装在本项目的 `release/` 目录，由 OpenCode v1 与 v2 共同加载。安装/release 布局与 bun 工具链见 `AGENTS.md`。

本 fork 以 **OpenCode provider 包**的方式加载，**不是** OpenCode 插件，也不修改 OpenCode 的全局工具注册表。

## 当前 OpenCode 用法

`npm` 必须指向编译后的 ESM 文件，而不是包目录：

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

Node ESM 不支持直接 import 本地目录：目录形式的 URI 在 Bun 下或许可行，但在 OpenCode 桌面运行时会以 `ERR_UNSUPPORTED_DIR_IMPORT` 失败，因此请使用显式的 `release/index.js` 路径。

OpenCode 按 provider ID 存储凭据。即使 `headroom-openai-branching` 与其它 provider 共用端点和 API key，它仍需要在 `~/.local/share/opencode/auth.json` 中有自己的凭据。

## Fork 专属行为

### 透明处理 pending `tool_search_call`

初始请求只使用 OpenCode 提供的工具。本 fork **不会**自动添加 `openai.tools.toolSearch()`，也**不会**声明隐藏的 `tool_search` 工具。

如果端点仍在 Responses API 返回中给出一个 `tool_search_call`，且同一响应中没有配对的 `tool_search_output`，provider 会：

1. 构造配对的 `tool_search_output`，其 `status: "completed"`、`tools: []`。
2. 经普通 OpenCode 提示同一路径的 `convertToOpenAIResponsesInput()` 转换助手可见输出，再把 pending call 与空输出追加到内部 follow-up 请求。
3. 最多重复三轮请求。
4. 从返回给 OpenCode 的最终结果中移除内部已完成的 `tool_search_call` 与 `tool_search_output`。

首个 Responses 请求仍归 OpenCode 的常规重试机制管理。每个内部 follow-up HTTP 请求都使用 OpenCode 1.18.25 的重试分类、重试次数、退避、抖动与 `Retry-After` 处理。兼容轮次与 HTTP 重试是**独立**计数器。五次重试后仍失败的 follow-up 会抛出原始错误，而不会把空工具目录当作成功的 provider 结果返回；OpenCode 之后可按其常规策略重试或中止整个原始模型请求。

如果同一响应还包含普通 `function_call`，本变体不发隐藏 follow-up，而是从 provider 结果中过滤掉 pending `tool_search_call`，立即把助手可见输出加上该普通函数调用返回给 OpenCode。OpenCode 执行该函数，其下一轮 canonical 请求包含 `A B function_call function_call_output`，不含 `tool_search_call` 或 `tool_search_output`。

没有普通函数调用的纯 pending tool-search 响应保留 base fork 的隐藏 follow-up。这样刻意把分支行为限制在真实 API 实验已验证的边界内。

这种 canonical replay 对 prompt caching 很关键。原始 Responses 消息含有的字段，会在 OpenCode 序列化下一轮提示时被移除。复用普通输入转换器，可让隐藏的稳定前缀在 JSON item 级别与下一轮 OpenCode 请求中的对应前缀**字节等价**。

这是协议兼容兜底，**不是**真正的动态工具发现。空目录是刻意的：OpenCode 已在初始请求中发送了其可用工具定义。

它同时处理把未解决调用标为 `execution: "server"` 的兼容端点；否则 OpenCode 会把该调用记为未知的 provider 执行工具。已有配对输出的 server call，以及显式配置的 `openai.tools.toolSearch()`，仍走上游路径。

### Reasoning 重放安全

Reasoning 模型总是请求 `reasoning.encrypted_content`。隐藏 follow-up 期间，reasoning 输出经普通提示转换器从加密内容与 summary 重建，而**不是**以原始 Responses item 或 `rs_* item_reference` 重放。

普通 OpenCode 轮次也避免 assistant-message 与 reasoning item-reference 的序列化。所配置的兼容端点即便在请求带 `store: true` 时也无法可靠地在跨轮之间保留这些响应 item，因此 fork 发送可重放内容，而不依赖 `msg_*` 或 `rs_*` ID。

混合分支路径不会为了获取加密 reasoning 而额外发一次隐藏请求。在反复进行的真实 `gpt-5.6-luna` treatment/control 运行中，每个请求都在首次网络尝试完成。全历史 positive control 保留了 `tsc/tso`，并 6/6 次报告了新观察到的 `tools: []` 事件。选择性分支移除了 `tsc/tso`，且 0/6 次报告该事件——即便它们保留了隐藏 reasoning item ID、加密内容、自动 summary 与成对的隐藏 message。加密 reasoning 是 reasoning 的连续性，而不是通用的加密对话快照；当其因果 item 缺失时，一个被接受的 item 仍可能被忽略。协议与语义结果见 `../docs/2026-08-14-tool-search-branching.md`。

已识别的 `tool_search` 历史保留 stock 的 item-reference 行为。只有当 OpenCode 把 provider 执行的 `tsc_*` item 记为未知工具、且未保留可安全重建该调用/输出的协议字段时，fork 才省略该 item。

### 显式 `store: true`

当解析出的 provider 选项 `store` 缺省时，Responses 请求发送 `store: true`：

```ts
const store = openaiOptions?.store ?? true;
```

部分兼容端点会把缺省字段解释为 `false`。显式发送默认值，可避免请求转换与端点持久化使用相互冲突的假设。显式的 `store: false` 保持为 `false`。

fork 先读取标准的 `providerOptions.openai` 命名空间。若不存在，则再读取由所配置 provider 名派生的命名空间，例如 `providerOptions['headroom-openai-branching']`。两个命名空间使用同一套完整 Responses 选项 schema，包括 `store`、`reasoningEffort`、`reasoningSummary`、`promptCacheKey`、`textVerbosity` 与 `include`。

### Function 工具名映射

Function 工具以 provider 工具名映射序列化：

```ts
toolNameMapping?.toProviderToolName(tool.name) ?? tool.name
```

这保证请求中的工具名与把返回的 function call 映射回 OpenCode 时所使用的名字一致。这是通用的 Responses 工具改动，不限于 `tool_search`。

### 缓冲式 Responses 流

`doStream()` 调用 `doGenerate()`，再把完成的结果转换为缓冲的 AI SDK 流。因此 OpenCode 永远看不到本 fork 的增量模型输出。

在网络上，`doGenerate()` 以 `stream: true` 发起请求并消费上游 SSE 流，在运行 `tool_search_call` 逻辑之前，从终止事件 `response.completed` 组装出最终响应。这既保证能在向 OpenCode 暴露任何内容之前先消费掉 pending 的 `tool_search_call`，也避免了某些中转站对长时间非流式请求返回非终态 keepalive 占位响应体的问题（见下文）。

后果包括：

- 上游响应仍在运行时没有增量模型输出；
- time-to-first-output 包含完整生成以及所有隐藏的 `tool_search` follow-up 轮次；
- chunk 边界与取消行为与 stock `@ai-sdk/openai` 的 Responses 流不同。

### 非流式 SSE 回退

由于 `doStream()` 是缓冲式的，OpenCode 端的流并非上游流。兼容中转站仍可能返回 SSE 响应体（`event: response.in_progress\ndata: {...}`），或者用 `response.failed` / `error` 帧表示失败而非 HTTP 错误状态；另有一些中转站会对长时间请求返回纯 JSON 的非终态占位响应体（`{"id":"…_keepalive","status":"in_progress","output":[]}`）。stock JSON 解析器会把这种响应体当成单个对象读取，报出笼统的 `Invalid JSON response`，掩盖真正的上游信息。

`src/responses/openai-responses-tolerant-response.ts` 会探测 `text/event-stream`（或形如 `event:` / `data:` 的响应体），从终止事件 `response.completed` 重建响应，并把非终态的 `in_progress` / `keepalive` 响应体视为可重试的 pending 状态。遇到 `response.failed` / `error` 帧时，它抛出携带真实上游消息与映射状态码的 `APICallError`，使 `server_is_overloaded` / `service_unavailable_error` 这类过载以可重试错误的形式暴露，而不是 `Invalid JSON response`。

## 保持上游兼容的部分

公开 provider 仍导出 `createOpenAI`、`openai`、Responses、chat、completion、embedding、image、transcription、speech translation、speech、realtime、files、skills、batch 支持，以及标准 OpenAI provider 工具。

fork 专属源码改动集中在：

- `src/responses/convert-to-openai-responses-input.ts`
- `src/responses/openai-responses-language-model.ts`
- `src/responses/openai-responses-retry.ts`
- `src/responses/openai-responses-prepare-tools.ts`
- `src/responses/openai-responses-tolerant-response.ts`

不要宣称本 fork 与 stock `@ai-sdk/openai@4.0.37` 行为完全一致。如前所述，Responses 请求序列化、follow-up 请求次数、跨轮 item 重放、延迟、token 用量与流式语义都可能不同。

## 诊断

启动 OpenCode 前设置以下环境变量，可打印每一轮兼容请求的紧凑摘要：

```bash
OPENAI_TOOL_SEARCH_COMPAT_DEBUG=1
```

日志包含兼容请求序号、轮次、从 0 开始的 HTTP 重试次数、`store`、`previous_response_id` 以及输入 item 的类型/ID。重试错误还会记录 OpenCode 是否将该错误分类为可重试、compat 是否会在本地重试，以及所选延迟。它刻意不记录完整提示或工具参数。

如果端点报告 `msg_*`、`rs_*` 或 `tsc_*` item 找不到，请在代理边界核对请求体，并检查：

- `store` 是 `true` 还是被显式配置为 `false`；
- 代理解析或忽略 `store` 的方式；
- reasoning follow-up 输入是否包含 `encrypted_content` 却没有服务端 `id` 或 `item_reference`；
- OpenCode 是否加载的是本 fork 的 `release/index.js`，而非 stock `@ai-sdk/openai`。

分支变体还会写实时 JSONL 诊断日志到：

```text
~/.local/share/opencode/provider-debug/<provider>/<session-id>.jsonl
```

配合 `setCacheKey: true` 时，`<session-id>` 即 OpenCode 会话 ID，因此日志路径是确定的。日志记录请求开始、响应输出 item 类型与请求错误，不含提示文本、工具参数或加密内容。完成的结果会把紧凑的 `toolSearchCompat` 调试摘要加入 provider metadata；OpenCode 会把它持久化到 SQLite 中对应返回 content part 上。该自定义 metadata 命名空间会被请求序列化器忽略，不会发送给 OpenAI。

设置 `OPENAI_TOOL_SEARCH_COMPAT_DEBUG_FILE=0` 可关闭 JSONL 文件。

## 构建与验证

需要 Node.js 22 或更新版本（或 Bun）。本项目使用 Bun 开发。

```bash
cd ~/.config/opencode/opencode-openai-provider
bun install
bun run typecheck   # tsc --noEmit -p tsconfig.build.json
bun run test        # tool_search_call 兼容与重试语义
bun run release     # 构建到 dist/ 并刷新安装目录 release/
# 或一条命令：bun run verify  (typecheck && test && release)
```

`release/index.js` 是 OpenCode 加载的运行时 provider，源码变更后必须重新生成（`bun run release`）。OpenCode 的配置与 provider 模块在启动时加载，因此在改动 provider 条目或重建 fork 后，需**完全退出并重启 OpenCode**。

把 OpenCode 策略原样复制进本包的细节、错误分类、状态语义与集成测试矩阵，见 `../docs/2026-08-29-tool-search-follow-up-retry.md`。

详细源码对比见 `../docs/2026-08-12-openai-fork-diff.md`，桌面端 provider-entry 失败分析见 `../docs/2026-08-12-node-bun-provider-entry.md`。
