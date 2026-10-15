# AGENTS.md — opencode-openai-provider

本仓库是 `@ai-sdk/openai` 的本地 fork（原始基线 `4.0.37`，已把上游 `4.0.93` 的不冲突变更移植进来，见下文「与上游的差异清单」），作为 **OpenCode provider 包** 使用（不是 OpenCode 插件，不修改 OpenCode 全局工具注册表）。**“适配 OpenCode”本身就包含处理 `tool_search_call`（tsc）**：兼容端点（Codex 风格）会注入客户端执行的 `tool_search_call`，而 OpenCode 并未注册或请求 `tool_search`，因此由 provider 包内部吸收。

## 目的

兼容端点（Codex 风格）会在 Responses 响应里注入客户端执行的 `tool_search_call`（简称 `tsc`，配套 `tool_search_output`，简称 `tso`），即使 OpenCode 并未注册或请求 `tool_search`。本 fork 让 **provider 包内部**吸收并完成这条 `tool_search_call + tool_search_output` 协议项：

- OpenCode（v1 与 v2）**不会收到也不会执行** `tool_search_call`；
- 隐藏的 follow-up 请求复用 OpenCode 的 `convertToOpenAIResponsesInput()` 规范序列化，使稳定前缀在 item 级别与下一轮 canonical 请求字节等价，从而**保留 prompt cache、节省 token**；
- 混有普通 `function_call` 时不再发隐藏 follow-up，而是过滤掉 pending `tsc` 直接返回 `function_call` 给 OpenCode。

行为细节、协议边界、诊断开关见 `README.md`。

## 安装目录（release）

**安装目录是本项目的 `release/`**，OpenCode 加载其中的编译产物：

```text
<project>/release/index.js
```

`release/` 由 `bun run release` 生成（先 build 到 `dist/`，再复制到 `release/`），**不提交进 Git**（见 `.gitignore` 的 `dist` 与 `release/`）。

`release/index.js` 保持 `@ai-sdk/provider`、`@ai-sdk/provider-utils`、`zod` 为 external，运行时由项目 `node_modules/` 或宿主 OpenCode 解析——因此不要删除本项目的 `node_modules/`，也不要让 `release/` 脱离项目目录树单独放置。

## OpenCode v1 / v2 用法

两个版本的 provider 配置都用 `npm` 字段指向同一份编译产物，因此一个 `release/index.js` 同时服务 v1 与 v2：

```jsonc
{
  "provider": {
    "headroom-openai-branching": {
      "name": "headroom-openai-branching",
      "npm": "file:///<project>/release/index.js",
      "options": { "baseURL": "http://127.0.0.1:8787/v1" }
    }
  }
}
```

注意：

- 必须指向显式文件 `release/index.js`，**不能指向目录**：Node ESM 直接 import 目录会以 `ERR_UNSUPPORTED_DIR_IMPORT` 失败（桌面运行时尤其如此）。
- OpenCode 按 provider ID 存凭证，新 provider ID 需要自己的凭据（即使与别的 provider 共用端点/密钥）。
- 重建 `release/` 或改动 provider 配置后，需**完全退出并重启 OpenCode**（provider 模块在启动时加载）。

## 构建与测试（bun）

```bash
cd ~/.config/opencode/opencode-openai-provider
bun install
bun run typecheck   # tsc --noEmit -p tsconfig.build.json
bun run test        # tool_search_call 兼容与重试语义 + SSE 组装/非终态 keepalive/失败帧 的用例
bun run release     # typecheck 通过后构建并更新安装目录 release/
# 或一条命令：bun run verify
```

**测试通过后才更新安装目录**：流程固定为 `typecheck → test → release`，`release/` 只反映已通过的构建。

## 与上游的差异清单（4.0.93 移植，2026-10-15）

本 fork 原基线为 `@ai-sdk/openai@4.0.37`。2026-10-15 用三方合并（base 4.0.37 / target 4.0.93）把上游 `4.0.93` 中**不冲突**的部分移植进本仓库（34 个修改文件、11 个新增文件、1 个删除文件），依赖随之升到 `@ai-sdk/provider 4.0.27`、`@ai-sdk/provider-utils 5.0.59`；`package.json` 版本号保持 `4.0.37` 不动。3 个文件存在冲突，整体保留 fork 侧实现（见下文「未移植的上游行为」）。

### fork 独有行为（上游没有）

兼容端点/代理的健壮性（均与真实故障对应）：

1. **非流式 `doGenerate` + 线上 `stream:true`**：`doGenerate` 以 `stream:true` 发请求并经 tolerant handler 聚合 SSE；`doStream` 仅把 `doGenerate` 结果包成缓冲流（`createBufferedLanguageModelStream`，无增量输出，TTFB = 整段生成时间）。原因：吸收隐藏 `tool_search_call` 需要完整响应后再决策。上游保留原生流式装配。
2. **tolerant SSE**（`src/responses/openai-responses-tolerant-response.ts`，fork 独有文件）：解析 `event:`/`data:` 帧；非终态 keepalive（`resp_*_keepalive` 空 placeholder）→ 可重试错误而非 `Invalid JSON response`；`response.failed`/错误帧 → 映射为 `APICallError`。
3. **空终态 output 重建**：终态 `response.completed.response.output == []` 时，从流式 `response.output_item.added/done` 帧按 `output_index` 重建 output（终止帧有内容时优先采用）。
4. **`output_text.annotations` 默认 `[]`**：Tencent DeepSeek 省略该字段时不再校验失败（zod `.default([])`）。
5. **DeepSeek `reasoning_text`**：`response.reasoning_text.delta/done` 作为 reasoning 流部件暴露。
6. **completed + 空 output → 可重试 503**：`doGenerate` 对空 `output` 抛 `APICallError`（statusCode 503、isRetryable），不再静默 `finish:"stop"`。
7. **`APICallError.responseBody` 序列化**：`rawResponse` 可能是对象（`safeParseJSON` 的 `rawValue`），统一 `JSON.stringify`，避免 OpenCode core `responseBody?.includes` 抛 TypeError 杀死会话。
8. **`avoidAssistantMessageItemReferences` / `avoidReasoningItemReferences`**（默认 false；LM 调用点置 true）：用可重放内容替代 `item_reference` 回放。
9. **始终请求 encrypted reasoning**：`isReasoningModel` 时自动 `include: reasoning.encrypted_content`；follow-up 丢弃未知 `tsc_` 前缀的 `item_reference`。
10. **`normalizeFileData`**：兼容旧 raw file data 形状（`convert-to-openai-responses-input.ts`）。
11. **fork 专属重试层**（`src/responses/openai-responses-retry.ts`）：镜像 OpenCode 重试语义（初始 2000ms、×2 退避、抖动 0.25、无头部上限 30s、`OPEN_CODE_RETRY_MAX_RETRIES=5`、Retry-After、可重试消息模式、上下文溢出模式）；round 0 不重试，后续轮才走本层。

### 上游 4.0.93 已移植（不冲突部分）

- 34 个修改文件（chat / completion / embedding / files / image / skills / speech / tool / transcription / realtime / responses 及 `index.ts`、`openai-provider.ts`、`openai-config.ts`、`openai-stream-error.ts` 等）。
- 11 个新增文件：`src/live/`（4 个）、`src/normalize-openai-json-schema.ts`、`src/openai-batch.ts`、`src/openai-decision-model.ts`、`src/openai-decision-model-options.ts`、`src/realtime/openai-realtime-factory.ts`、`src/responses/expand-parallel-tool-call.ts`、`src/responses/map-openai-responses-annotation.ts`；删除 `src/openai-responses-batch.ts`（被 `openai-batch.ts` 取代）。
- 集成补丁（为衔接 3 个保留 fork 侧的冲突文件）：LM 导出 `openaiResponsesSupportedUrls`、`mapWebSearchOutput`，新增 `static prepareRequest()` 供 `openai-batch.ts` 调用；batch 对 `part.summary` 判空；`convert-to-openai-responses-input.ts` 的 assistant 消息改为 `content: part.text`（上游类型删除了 `id` 与数组 content）；`usage ?? undefined`；`image_generation.action` 透传。

### 未移植的上游行为（3 个冲突文件，保留 fork 侧）

| 文件 | 未采用的上游新行为 |
| --- | --- |
| `src/responses/openai-responses-language-model.ts` | `isUndeclaredParallelToolCall` / `suppressInputStreaming` 流装配保护；`expandParallelToolCall` 并行工具调用展开；`mapOpenAIResponsesAnnotationSource` 注解 → sources 映射；`getOpenAIToolName` 工具存在性检查；`reasoningEffortUpdate` 会话中途更新；`getConfigurationUpdateUnsupportedReason` |
| `src/responses/openai-responses-prepare-tools.ts` | function tool 的 `normalizeOpenAIJsonSchema` 归一化；`strict: tool.strict ?? false` 默认下发；`resolveAsyncToolOption` 的 async 工具选项 |
| `src/responses/convert-to-openai-responses-input.ts` | assistant 消息 `(!isReasoningModel \|\| hasPrecedingReasoning)` 的 item_reference 条件；`emittedTextItemIds` 去重 |

已知后果：

- `expand-parallel-tool-call.ts`、`map-openai-responses-annotation.ts` 已随移植进入源码树，但**未被引用**（上游只有冲突文件引用它们）；
- `reasoningEffortUpdate` 的选项 schema 已随自动合并进入 `openai-responses-language-model-options.ts`，但 LM 侧不处理；
- 上游 `strict:false` 默认未采用：本 fork 仅在显式指定时上传 `strict`，保持 4.0.37 请求体语义，避免影响 prompt cache。

### 验证记录（2026-10-15）

`bun run verify` 全绿（typecheck + 34 个测试 + release 重建）；v1 CLI 冒烟通过：`headroom-openai-fork/gpt-5.6-luna`（简单问答 + read 工具调用）、`headroom-tencent-fork/deepseek-v4.1-flash`（简单问答）。

## 约束

- 不修改 OpenCode core；不注册 OpenCode 插件；不改全局工具注册表。
- 保持 `@ai-sdk/provider` 等为 external，不要打进 bundle（否则 `instanceof APIConversionError` 等跨包判断会因双份实现而失效）。
- `dist/`、`release/`、`node_modules/` 均不提交。
- 诊断：`OPENAI_TOOL_SEARCH_COMPAT_DEBUG=1`（紧凑摘要）、`OPENAI_TOOL_SEARCH_COMPAT_DEBUG_FILE=0`（关闭 JSONL 文件）；JSONL 落在 `~/.local/share/opencode/provider-debug/<provider>/<session-id>.jsonl`。
