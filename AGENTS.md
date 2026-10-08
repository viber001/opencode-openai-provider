# AGENTS.md — opencode-openai-provider

本仓库是 `@ai-sdk/openai@4.0.37` 的本地 fork，作为 **OpenCode provider 包** 使用（不是 OpenCode 插件，不修改 OpenCode 全局工具注册表）。**“适配 OpenCode”本身就包含处理 `tool_search_call`（tsc）**：兼容端点（Codex 风格）会注入客户端执行的 `tool_search_call`，而 OpenCode 并未注册或请求 `tool_search`，因此由 provider 包内部吸收。

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

## 约束

- 不修改 OpenCode core；不注册 OpenCode 插件；不改全局工具注册表。
- 保持 `@ai-sdk/provider` 等为 external，不要打进 bundle（否则 `instanceof APIConversionError` 等跨包判断会因双份实现而失效）。
- `dist/`、`release/`、`node_modules/` 均不提交。
- 诊断：`OPENAI_TOOL_SEARCH_COMPAT_DEBUG=1`（紧凑摘要）、`OPENAI_TOOL_SEARCH_COMPAT_DEBUG_FILE=0`（关闭 JSONL 文件）；JSONL 落在 `~/.local/share/opencode/provider-debug/<provider>/<session-id>.jsonl`。
