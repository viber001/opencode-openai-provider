import {
  APICallError,
  type JSONValue,
  type LanguageModelV4,
  type LanguageModelV4Prompt,
  type LanguageModelV4CallOptions,
  type LanguageModelV4Content,
  type LanguageModelV4FinishReason,
  type LanguageModelV4GenerateResult,
  type LanguageModelV4ProviderTool,
  type LanguageModelV4StreamPart,
  type LanguageModelV4StreamResult,
  type LanguageModelV4ToolApprovalRequest,
  type SharedV4ProviderMetadata,
  type SharedV4Warning,
} from '@ai-sdk/provider';
import {
  combineHeaders,
  createEventSourceResponseHandler,
  createJsonResponseHandler,
  createToolNameMapping,
  generateId,
  isCustomReasoning,
  parseProviderOptions,
  postJsonToApi,
  serializeModelOptions,
  WORKFLOW_DESERIALIZE,
  WORKFLOW_SERIALIZE,
  type InferSchema,
  type ParseResult,
} from '@ai-sdk/provider-utils';
import type { OpenAIConfig } from '../openai-config';
import { openaiFailedResponseHandler } from '../openai-error';
import { getOpenAILanguageModelCapabilities } from '../openai-language-model-capabilities';
import { throwIfOpenAIStreamErrorBeforeOutput } from '../openai-stream-error';
import type { applyPatchInputSchema } from '../tool/apply-patch';
import type {
  codeInterpreterInputSchema,
  codeInterpreterOutputSchema,
} from '../tool/code-interpreter';
import type { computerInputSchema } from '../tool/computer';
import type { fileSearchOutputSchema } from '../tool/file-search';
import type { imageGenerationOutputSchema } from '../tool/image-generation';
import type { localShellInputSchema } from '../tool/local-shell';
import type { mcpOutputSchema } from '../tool/mcp';
import type { shellInputSchema, shellOutputSchema } from '../tool/shell';
import type {
  toolSearchInputSchema,
  toolSearchOutputSchema,
} from '../tool/tool-search';
import type {
  programmaticToolCallingInputSchema,
  programmaticToolCallingOutputSchema,
} from '../tool/programmatic-tool-calling';
import type { webSearchOutputSchema } from '../tool/web-search';
import {
  convertOpenAIResponsesUsage,
  type OpenAIResponsesUsage,
} from './convert-openai-responses-usage';
import { convertToOpenAIResponsesInput } from './convert-to-openai-responses-input';
import { mapOpenAIResponseFinishReason } from './map-openai-responses-finish-reason';
import {
  getOpenCodeResponseErrorStatusCode,
  OPEN_CODE_RETRY_MAX_RETRIES,
  retryWithOpenCodePolicy,
} from './openai-responses-retry';
import {
  openaiResponsesChunkSchema,
  openaiResponsesResponseSchema,
  type OpenAIResponsesChunk,
  type OpenAIResponsesIncludeOptions,
  type OpenAIResponsesIncludeValue,
  type OpenAIResponsesLogprobs,
  type OpenAIResponsesWebSearchAction,
  type OpenAIResponsesApplyPatchOperationDiffDeltaChunk,
  type OpenAIResponsesApplyPatchOperationDiffDoneChunk,
  type OpenAIResponsesComputerAction,
  type OpenAIResponsesInput,
  type OpenAIResponsesToolSearchCall,
} from './openai-responses-api';
import {
  openaiLanguageModelResponsesOptionsSchema,
  TOP_LOGPROBS_MAX,
  type OpenAIResponsesModelId,
} from './openai-responses-language-model-options';
import { prepareResponsesTools } from './openai-responses-prepare-tools';

const MAX_CODEX_TOOL_SEARCH_ROUNDS = 3;
let codexToolSearchRequestSequence = 0;
const TOOL_SEARCH_COMPAT_PROVIDER_METADATA = 'toolSearchCompat';

type DiagnosticLogModules = {
  appendFile: typeof import('node:fs/promises')['appendFile'];
  mkdir: typeof import('node:fs/promises')['mkdir'];
  join: typeof import('node:path')['join'];
};

type DiagnosticLog = {
  filePath: string;
  directory: string;
  provider: string;
  modelId: string;
  sessionKey: string;
  requestCount: number;
  hiddenRoundCount: number;
  writeQueue: Promise<void>;
  modules: DiagnosticLogModules;
};

let diagnosticLogModulesPromise: Promise<DiagnosticLogModules | undefined> | undefined;
const diagnosticLogs = new Map<string, DiagnosticLog>();
const diagnosticFallbackSessionKeys = new Map<string, string>();

type OpenAIResponsesOutput = NonNullable<
  InferSchema<typeof openaiResponsesResponseSchema>['output']
>;
type OpenAIResponsesResponseWithOutput = InferSchema<
  typeof openaiResponsesResponseSchema
> & { output: OpenAIResponsesOutput };
type OpenAIResponsesOutputItem = OpenAIResponsesOutput[number];
type AssistantPromptContent = Extract<
  LanguageModelV4Prompt[number],
  { role: 'assistant' }
>['content'];

function logCodexToolSearchRequest({
  request,
  round,
  attempt,
  requestBody,
}: {
  request: number;
  round: number;
  attempt: number;
  requestBody: Record<string, unknown>;
}) {
  if (process.env.OPENAI_TOOL_SEARCH_COMPAT_DEBUG !== '1') {
    return;
  }

  const input = Array.isArray(requestBody.input) ? requestBody.input : [];
  const inputItems = summarizeDiagnosticItems(input);

  console.error(
    'OPENAI TOOL SEARCH COMPAT REQUEST',
    JSON.stringify({
      request,
      round,
      attempt,
      store: requestBody.store,
      previousResponseId: requestBody.previous_response_id,
      inputItems,
    }),
  );
}

function summarizeDiagnosticItems(items: unknown[]) {
  return items.map((item, index) => {
    if (item == null || typeof item !== 'object') {
      return { index, type: typeof item };
    }

    const record = item as Record<string, unknown>;
    return {
      index,
      type: record.type ?? record.role ?? 'unknown',
      ...(typeof record.id === 'string' ? { id: record.id } : {}),
      ...(typeof record.call_id === 'string'
        ? { callId: record.call_id }
        : {}),
      ...(typeof record.execution === 'string'
        ? { execution: record.execution }
        : {}),
      ...(typeof record.status === 'string'
        ? { status: record.status }
        : {}),
      ...(record.type === 'reasoning'
        ? { hasEncryptedContent: typeof record.encrypted_content === 'string' }
        : {}),
    };
  });
}

function summarizeDiagnosticOutput(output: OpenAIResponsesOutput) {
  return summarizeDiagnosticItems(output);
}

function sanitizeDiagnosticFileComponent(value: string): string {
  return value.replace(/[^a-zA-Z0-9._-]/g, '_').slice(0, 160) || 'unknown';
}

async function loadDiagnosticLogModules(): Promise<DiagnosticLogModules | undefined> {
  if (
    typeof process === 'undefined' ||
    process.versions?.node == null
  ) {
    return undefined;
  }

  diagnosticLogModulesPromise ??= Promise.all([
    import('node:fs/promises'),
    import('node:path'),
  ])
    .then(([fs, path]) => ({
      appendFile: fs.appendFile,
      mkdir: fs.mkdir,
      join: path.join,
    }))
    .catch(() => undefined);

  return diagnosticLogModulesPromise;
}

async function getDiagnosticLog({
  provider,
  modelId,
  sessionKey,
}: {
  provider: string;
  modelId: string;
  sessionKey?: string;
}): Promise<DiagnosticLog | undefined> {
  if (process.env.OPENAI_TOOL_SEARCH_COMPAT_DEBUG_FILE === '0') {
    return undefined;
  }

  const modules = await loadDiagnosticLogModules();
  const home = process.env.HOME ?? process.env.USERPROFILE;
  if (modules == null || home == null) {
    return undefined;
  }

  const providerModelKey = `${provider}\0${modelId}`;
  let normalizedSessionKey = sessionKey?.trim();
  if (normalizedSessionKey == null || normalizedSessionKey.length === 0) {
    normalizedSessionKey = diagnosticFallbackSessionKeys.get(providerModelKey);
    if (normalizedSessionKey == null) {
      normalizedSessionKey = `run_${generateId()}`;
      diagnosticFallbackSessionKeys.set(providerModelKey, normalizedSessionKey);
    }
  }

  const key = `${provider}\0${normalizedSessionKey}`;
  const existing = diagnosticLogs.get(key);
  if (existing != null) {
    return existing;
  }

  const fileName = `${sanitizeDiagnosticFileComponent(normalizedSessionKey)}.jsonl`;
  const providerDirectory = sanitizeDiagnosticFileComponent(
    provider.endsWith('.responses')
      ? provider.slice(0, -'.responses'.length)
      : provider,
  );
  const directory = modules.join(
    home,
    '.local',
    'share',
    'opencode',
    'provider-debug',
    providerDirectory,
  );
  const log: DiagnosticLog = {
    filePath: modules.join(directory, fileName),
    directory,
    provider,
    modelId,
    sessionKey: normalizedSessionKey,
    requestCount: 0,
    hiddenRoundCount: 0,
    writeQueue: Promise.resolve(),
    modules,
  };
  diagnosticLogs.set(key, log);
  return log;
}

function appendDiagnosticLog(
  log: DiagnosticLog | undefined,
  event: Record<string, unknown>,
): Promise<void> {
  if (log == null) {
    return Promise.resolve();
  }

  const line = `${JSON.stringify({
    timestamp: new Date().toISOString(),
    provider: log.provider,
    modelId: log.modelId,
    sessionKey: log.sessionKey,
    ...event,
  })}\n`;
  const write = log.writeQueue.then(async () => {
    await log.modules.mkdir(log.directory, { recursive: true });
    await log.modules.appendFile(log.filePath, line, 'utf8');
  });
  log.writeQueue = write.catch(() => undefined);
  return write.catch(() => undefined);
}

function toolSearchOutput(
  call: Extract<
    OpenAIResponsesInput[number],
    { type: 'tool_search_call' }
  >,
): Extract<OpenAIResponsesInput[number], { type: 'tool_search_output' }> {
  return {
    type: 'tool_search_output',
    execution: call.execution,
    call_id: call.call_id ?? call.id,
    status: 'completed',
    // OpenCode's tool descriptions are already present in the initial request.
    // Do not duplicate them as dynamically loaded tools.
    tools: [],
  };
}

function getReasoningSummaryParts(
  part: Extract<OpenAIResponsesOutputItem, { type: 'reasoning' }>,
) {
  return part.summary.length === 0
    ? [{ type: 'summary_text' as const, text: '' }]
    : part.summary;
}

function getReasoningProviderOptions(
  part: Extract<OpenAIResponsesOutputItem, { type: 'reasoning' }>,
): ResponsesReasoningProviderMetadata {
  return {
    itemId: part.id,
    reasoningEncryptedContent: part.encrypted_content ?? null,
  };
}

function getTextProviderOptions(
  part: Extract<OpenAIResponsesOutputItem, { type: 'message' }>,
  contentPart: Extract<
    Extract<OpenAIResponsesOutputItem, { type: 'message' }>['content'][number],
    { type: 'output_text' }
  >,
): ResponsesTextProviderMetadata {
  return {
    itemId: part.id,
    ...(part.phase != null && { phase: part.phase }),
    ...(contentPart.annotations.length > 0 && {
      annotations: contentPart.annotations,
    }),
  };
}

function prepareCodexToolSearchCanonicalPrompt({
  output,
  providerOptionsName,
}: {
  output: OpenAIResponsesOutput;
  providerOptionsName: string;
}): LanguageModelV4Prompt {
  const content: AssistantPromptContent = [];

  for (const part of output) {
    switch (part.type) {
      case 'reasoning': {
        for (const summary of getReasoningSummaryParts(part)) {
          content.push({
            type: 'reasoning',
            text: summary.text,
            providerOptions: {
              [providerOptionsName]: getReasoningProviderOptions(part),
            },
          });
        }
        break;
      }

      case 'message': {
        for (const contentPart of part.content) {
          content.push({
            type: 'text',
            text: contentPart.text,
            providerOptions: {
              [providerOptionsName]: getTextProviderOptions(part, contentPart),
            },
          });
        }
        break;
      }

      case 'compaction': {
        content.push({
          type: 'custom',
          kind: 'openai.compaction',
          providerOptions: {
            [providerOptionsName]: {
              type: 'compaction',
              itemId: part.id,
              encryptedContent: part.encrypted_content,
            } satisfies ResponsesCompactionProviderMetadata,
          },
        });
        break;
      }
    }
  }

  return content.length === 0 ? [] : [{ role: 'assistant', content }];
}

function prepareCodexToolSearchPassthroughInput({
  output,
  pendingToolSearchCallIds,
  excludeFunctionCalls,
}: {
  output: OpenAIResponsesOutput;
  pendingToolSearchCallIds: Set<string>;
  excludeFunctionCalls: boolean;
}): OpenAIResponsesInput {
  return output.flatMap(part => {
    if (
      part.type === 'message' ||
      part.type === 'reasoning' ||
      part.type === 'compaction' ||
      (part.type === 'function_call' && excludeFunctionCalls) ||
      (part.type === 'tool_search_call' &&
        pendingToolSearchCallIds.has(part.call_id ?? part.id))
    ) {
      return [];
    }

    return [part as unknown as OpenAIResponsesInput[number]];
  });
}

function getConfiguredProviderOptionsName(provider: string): string {
  return provider.endsWith('.responses')
    ? provider.slice(0, -'.responses'.length)
    : provider;
}

import type {
  ResponsesCompactionProviderMetadata,
  ResponsesProviderMetadata,
  ResponsesReasoningProviderMetadata,
  ResponsesSourceDocumentProviderMetadata,
  ResponsesTextProviderMetadata,
} from './openai-responses-provider-metadata';

/**
 * Extracts a mapping from MCP approval request IDs to their corresponding tool call IDs
 * from the prompt. When an MCP tool requires approval, we generate a tool call ID to track
 * the pending approval in our system. When the user responds to the approval (and we
 * continue the conversation), we need to map the approval request ID back to our tool call ID
 * so that tool results reference the correct tool call.
 */
function extractApprovalRequestIdToToolCallIdMapping(
  prompt: LanguageModelV4Prompt,
): Record<string, string> {
  const mapping: Record<string, string> = {};
  for (const message of prompt) {
    if (message.role !== 'assistant') continue;
    for (const part of message.content) {
      if (part.type !== 'tool-call') continue;
      const approvalRequestId = part.providerOptions?.openai
        ?.approvalRequestId as string | undefined;
      if (approvalRequestId != null) {
        mapping[approvalRequestId] = part.toolCallId;
      }
    }
  }
  return mapping;
}

function mapComputerAction(
  action: OpenAIResponsesComputerAction,
): InferSchema<typeof computerInputSchema>['actions'][number] {
  switch (action.type) {
    case 'click':
      return {
        type: 'click',
        button: action.button,
        x: action.x,
        y: action.y,
        ...(action.keys != null && { keys: action.keys }),
      };
    case 'double_click':
      return {
        type: 'double_click',
        x: action.x,
        y: action.y,
        ...(action.keys != null && { keys: action.keys }),
      };
    case 'drag':
      return {
        type: 'drag',
        path: action.path,
        ...(action.keys != null && { keys: action.keys }),
      };
    case 'keypress':
      return action;
    case 'move':
      return {
        type: 'move',
        x: action.x,
        y: action.y,
        ...(action.keys != null && { keys: action.keys }),
      };
    case 'screenshot':
      return action;
    case 'scroll':
      return {
        type: 'scroll',
        x: action.x,
        y: action.y,
        scrollX: action.scroll_x,
        scrollY: action.scroll_y,
        ...(action.keys != null && { keys: action.keys }),
      };
    case 'type':
      return action;
    case 'wait':
      return action;
  }
}

function mapComputerCallInput({
  action,
  actions,
  pending_safety_checks,
  status,
}: {
  action?: OpenAIResponsesComputerAction | null;
  actions?: OpenAIResponsesComputerAction[] | null;
  pending_safety_checks?: Array<{
    id: string;
    code?: string | null;
    message?: string | null;
  }> | null;
  status: 'in_progress' | 'completed' | 'incomplete';
}): InferSchema<typeof computerInputSchema> {
  return {
    actions: (actions ?? (action != null ? [action] : [])).map(
      mapComputerAction,
    ),
    pendingSafetyChecks:
      pending_safety_checks?.map(safetyCheck => ({
        id: safetyCheck.id,
        ...(safetyCheck.code != null && { code: safetyCheck.code }),
        ...(safetyCheck.message != null && { message: safetyCheck.message }),
      })) ?? [],
    status,
  };
}

export class OpenAIResponsesLanguageModel implements LanguageModelV4 {
  readonly specificationVersion = 'v4';

  readonly modelId: OpenAIResponsesModelId;

  private readonly config: OpenAIConfig;

  static [WORKFLOW_SERIALIZE](model: OpenAIResponsesLanguageModel) {
    return serializeModelOptions({
      modelId: model.modelId,
      config: model.config,
    });
  }

  static [WORKFLOW_DESERIALIZE](options: {
    modelId: OpenAIResponsesModelId;
    config: OpenAIConfig;
  }) {
    return new OpenAIResponsesLanguageModel(options.modelId, options.config);
  }

  constructor(modelId: OpenAIResponsesModelId, config: OpenAIConfig) {
    this.modelId = modelId;
    this.config = config;
  }

  readonly supportedUrls: Record<string, RegExp[]> = {
    'image/*': [/^https?:\/\/.*$/],
    'application/pdf': [/^https?:\/\/.*$/],
  };

  get provider(): string {
    return this.config.provider;
  }

  protected async getArgs({
    maxOutputTokens,
    temperature,
    stopSequences,
    topP,
    topK,
    presencePenalty,
    frequencyPenalty,
    seed,
    prompt,
    reasoning,
    providerOptions,
    tools,
    toolChoice,
    responseFormat,
  }: LanguageModelV4CallOptions) {
    const warnings: SharedV4Warning[] = [];
    const modelCapabilities = getOpenAILanguageModelCapabilities(this.modelId);

    if (topK != null) {
      warnings.push({ type: 'unsupported', feature: 'topK' });
    }

    if (seed != null) {
      warnings.push({ type: 'unsupported', feature: 'seed' });
    }

    if (presencePenalty != null) {
      warnings.push({ type: 'unsupported', feature: 'presencePenalty' });
    }

    if (frequencyPenalty != null) {
      warnings.push({ type: 'unsupported', feature: 'frequencyPenalty' });
    }

    if (stopSequences != null) {
      warnings.push({ type: 'unsupported', feature: 'stopSequences' });
    }

    const providerOptionsName = this.config.provider.includes('azure')
      ? 'azure'
      : 'openai';
    let openaiOptions = await parseProviderOptions({
      provider: providerOptionsName,
      providerOptions,
      schema: openaiLanguageModelResponsesOptionsSchema,
    });

    if (openaiOptions == null && providerOptionsName !== 'openai') {
      openaiOptions = await parseProviderOptions({
        provider: 'openai',
        providerOptions,
        schema: openaiLanguageModelResponsesOptionsSchema,
      });
    }

    const configuredProviderOptionsName = getConfiguredProviderOptionsName(
      this.config.provider,
    );
    if (
      openaiOptions == null &&
      configuredProviderOptionsName !== providerOptionsName &&
      configuredProviderOptionsName !== 'openai'
    ) {
      openaiOptions = await parseProviderOptions({
        provider: configuredProviderOptionsName,
        providerOptions,
        schema: openaiLanguageModelResponsesOptionsSchema,
      });
    }

    const diagnosticLog = await getDiagnosticLog({
      provider: this.config.provider,
      modelId: this.modelId,
      sessionKey:
        typeof openaiOptions?.promptCacheKey === 'string'
          ? openaiOptions.promptCacheKey
          : undefined,
    });

    const resolvedReasoningEffort =
      openaiOptions?.reasoningEffort ??
      (isCustomReasoning(reasoning) ? reasoning : undefined);
    const resolvedReasoningSummary =
      openaiOptions?.reasoningSummary !== undefined
        ? openaiOptions.reasoningSummary
        : resolvedReasoningEffort != null && resolvedReasoningEffort !== 'none'
          ? 'detailed'
          : undefined;

    const isReasoningModel =
      openaiOptions?.forceReasoning ?? modelCapabilities.isReasoningModel;

    if (openaiOptions?.conversation && openaiOptions?.previousResponseId) {
      warnings.push({
        type: 'unsupported',
        feature: 'conversation',
        details: 'conversation and previousResponseId cannot be used together',
      });
    }

    // `tool_search_call` is an OpenAI Responses protocol item. It is handled
    // below and is never registered as an OpenCode tool.
    const requestTools = tools ?? [];

    const toolNameMapping = createToolNameMapping({
      tools: requestTools,
      providerToolNames: {
        'openai.code_interpreter': 'code_interpreter',
        'openai.computer': 'computer',
        'openai.file_search': 'file_search',
        'openai.image_generation': 'image_generation',
        'openai.local_shell': 'local_shell',
        'openai.shell': 'shell',
        'openai.web_search': 'web_search',
        'openai.web_search_preview': 'web_search_preview',
        'openai.mcp': 'mcp',
        'openai.apply_patch': 'apply_patch',
        'openai.tool_search': 'tool_search',
        'openai.programmatic_tool_calling': 'programmatic_tool_calling',
      },
    });

    const customProviderToolNames = new Set<string>();
    // OpenAI requires function_call_output.output to contain valid JSON when the
    // function declares output_schema. Preserve the affected SDK tool names so
    // prompt conversion can apply that encoding only to their results.
    const outputSchemaToolNames = new Set<string>();
    const {
      tools: openaiTools,
      toolChoice: openaiToolChoice,
      toolWarnings,
    } = await prepareResponsesTools({
      tools: requestTools,
      toolChoice,
      allowedTools: openaiOptions?.allowedTools ?? undefined,
      toolNameMapping,
      customProviderToolNames,
      outputSchemaToolNames,
    });

    // Keep the Responses API default explicit because some compatible
    // endpoints treat an omitted `store` value as false.
    const store = openaiOptions?.store ?? true;

    const convertPromptToInput = (prompt: LanguageModelV4Prompt) =>
      convertToOpenAIResponsesInput({
        prompt,
        toolNameMapping,
        systemMessageMode:
          openaiOptions?.systemMessageMode ??
          (isReasoningModel
            ? 'developer'
            : modelCapabilities.systemMessageMode),
        providerOptionsName,
        fileIdPrefixes: this.config.fileIdPrefixes,
        passThroughUnsupportedFiles:
          openaiOptions?.passThroughUnsupportedFiles ?? false,
        store,
        hasConversation: openaiOptions?.conversation != null,
        hasPreviousResponseId: openaiOptions?.previousResponseId != null,
        // The compatibility endpoint does not reliably persist response items
        // across OpenCode turns, even when the request uses store: true.
        avoidAssistantMessageItemReferences: true,
        avoidReasoningItemReferences: true,
        // Legacy unknown provider-tool parts only retain a tsc_* item ID, so
        // they cannot be reconstructed or referenced safely on a later turn.
        dropUnknownToolSearchItemReferences: true,
        hasLocalShellTool: hasOpenAITool('openai.local_shell'),
        hasShellTool: hasOpenAITool('openai.shell'),
        hasApplyPatchTool: hasOpenAITool('openai.apply_patch'),
        hasComputerTool: hasOpenAITool('openai.computer'),
        customProviderToolNames:
          customProviderToolNames.size > 0
            ? customProviderToolNames
            : undefined,
        outputSchemaToolNames:
          outputSchemaToolNames.size > 0 ? outputSchemaToolNames : undefined,
      });

    const { input, warnings: inputWarnings } =
      await convertPromptToInput(prompt);

    warnings.push(...inputWarnings);

    const strictJsonSchema = openaiOptions?.strictJsonSchema ?? true;

    let include: OpenAIResponsesIncludeOptions = openaiOptions?.include;

    function addInclude(key: OpenAIResponsesIncludeValue) {
      if (include == null) {
        include = [key];
      } else if (!include.includes(key)) {
        include = [...include, key];
      }
    }

    function hasOpenAITool(id: string) {
      return (
        requestTools.find(tool => tool.type === 'provider' && tool.id === id) !=
        null
      );
    }

    // when logprobs are requested, automatically include them:
    const topLogprobs =
      typeof openaiOptions?.logprobs === 'number'
        ? openaiOptions?.logprobs
        : openaiOptions?.logprobs === true
          ? TOP_LOGPROBS_MAX
          : undefined;

    if (topLogprobs) {
      addInclude('message.output_text.logprobs');
    }

    // when a web search tool is present, automatically include the sources:
    const webSearchToolName = (
        requestTools.find(
        tool =>
          tool.type === 'provider' &&
          (tool.id === 'openai.web_search' ||
            tool.id === 'openai.web_search_preview'),
      ) as LanguageModelV4ProviderTool | undefined
    )?.name;

    if (webSearchToolName) {
      addInclude('web_search_call.action.sources');
    }

    // when a code interpreter tool is present, automatically include the outputs:
    if (hasOpenAITool('openai.code_interpreter')) {
      addInclude('code_interpreter_call.outputs');
    }

    // The proxy may emit a hidden client-side tool_search_call even when the
    // caller did not declare openai.tool_search. Always request encrypted
    // reasoning so that follow-ups can avoid replaying server-side IDs.
    if (isReasoningModel) {
      addInclude('reasoning.encrypted_content');
    }

    const baseArgs = {
      model: this.modelId,
      input,
      temperature,
      top_p: topP,
      max_output_tokens: maxOutputTokens,

      ...((responseFormat?.type === 'json' || openaiOptions?.textVerbosity) && {
        text: {
          ...(responseFormat?.type === 'json' && {
            format:
              responseFormat.schema != null
                ? {
                    type: 'json_schema',
                    strict: strictJsonSchema,
                    name: responseFormat.name ?? 'response',
                    description: responseFormat.description,
                    schema: responseFormat.schema,
                  }
                : { type: 'json_object' },
          }),
          ...(openaiOptions?.textVerbosity && {
            verbosity: openaiOptions.textVerbosity,
          }),
        },
      }),

      // provider options:
      conversation: openaiOptions?.conversation,
      max_tool_calls: openaiOptions?.maxToolCalls,
      metadata: openaiOptions?.metadata,
      parallel_tool_calls: openaiOptions?.parallelToolCalls,
      previous_response_id: openaiOptions?.previousResponseId,
      store,
      user: openaiOptions?.user,
      instructions: openaiOptions?.instructions,
      service_tier: openaiOptions?.serviceTier,
      include,
      prompt_cache_key: openaiOptions?.promptCacheKey,
      prompt_cache_options: openaiOptions?.promptCacheOptions,
      prompt_cache_retention: openaiOptions?.promptCacheRetention,
      safety_identifier: openaiOptions?.safetyIdentifier,
      top_logprobs: topLogprobs,
      truncation: openaiOptions?.truncation,

      // context management (server-side compaction):
      ...(openaiOptions?.contextManagement && {
        context_management: openaiOptions.contextManagement.map(cm => ({
          type: cm.type,
          compact_threshold: cm.compactThreshold,
        })),
      }),

      // model-specific settings:
      ...(isReasoningModel &&
        (resolvedReasoningEffort != null ||
          resolvedReasoningSummary != null ||
          openaiOptions?.reasoningMode != null ||
          openaiOptions?.reasoningContext != null) && {
          reasoning: {
            ...(resolvedReasoningEffort != null && {
              effort: resolvedReasoningEffort,
            }),
            ...(resolvedReasoningSummary != null && {
              summary: resolvedReasoningSummary,
            }),
            ...(openaiOptions?.reasoningMode != null && {
              mode: openaiOptions.reasoningMode,
            }),
            ...(openaiOptions?.reasoningContext != null && {
              context: openaiOptions.reasoningContext,
            }),
          },
        }),
    };

    // remove unsupported settings for reasoning models
    // see https://platform.openai.com/docs/guides/reasoning#limitations
    if (isReasoningModel) {
      // when reasoning effort is none, gpt-5.1 models allow temperature, topP, logprobs
      //  https://platform.openai.com/docs/guides/latest-model#gpt-5-1-parameter-compatibility
      if (
        !(
          resolvedReasoningEffort === 'none' &&
          modelCapabilities.supportsNonReasoningParameters
        )
      ) {
        if (baseArgs.temperature != null) {
          baseArgs.temperature = undefined;
          warnings.push({
            type: 'unsupported',
            feature: 'temperature',
            details: 'temperature is not supported for reasoning models',
          });
        }

        if (baseArgs.top_p != null) {
          baseArgs.top_p = undefined;
          warnings.push({
            type: 'unsupported',
            feature: 'topP',
            details: 'topP is not supported for reasoning models',
          });
        }
      }
    } else {
      if (openaiOptions?.reasoningEffort != null) {
        warnings.push({
          type: 'unsupported',
          feature: 'reasoningEffort',
          details: 'reasoningEffort is not supported for non-reasoning models',
        });
      }

      if (openaiOptions?.reasoningSummary != null) {
        warnings.push({
          type: 'unsupported',
          feature: 'reasoningSummary',
          details: 'reasoningSummary is not supported for non-reasoning models',
        });
      }

      if (openaiOptions?.reasoningMode != null) {
        warnings.push({
          type: 'unsupported',
          feature: 'reasoningMode',
          details: 'reasoningMode is not supported for non-reasoning models',
        });
      }

      if (openaiOptions?.reasoningContext != null) {
        warnings.push({
          type: 'unsupported',
          feature: 'reasoningContext',
          details: 'reasoningContext is not supported for non-reasoning models',
        });
      }
    }

    // Validate flex processing support
    if (
      openaiOptions?.serviceTier === 'flex' &&
      !modelCapabilities.supportsFlexProcessing
    ) {
      warnings.push({
        type: 'unsupported',
        feature: 'serviceTier',
        details:
          'flex processing is only available for o3, o4-mini, and gpt-5 models',
      });
      // Remove from args if not supported
      delete (baseArgs as any).service_tier;
    }

    // Validate priority processing support
    if (
      (openaiOptions?.serviceTier === 'priority' ||
        openaiOptions?.serviceTier === 'fast') &&
      !modelCapabilities.supportsPriorityProcessing
    ) {
      warnings.push({
        type: 'unsupported',
        feature: 'serviceTier',
        details:
          'priority processing is only available for supported models (gpt-4, gpt-5, gpt-5-mini, o3, o4-mini) and requires Enterprise access. gpt-5-nano is not supported',
      });
      // Remove from args if not supported
      delete (baseArgs as any).service_tier;
    }

    const shellToolEnvType = (
      requestTools.find(
        tool => tool.type === 'provider' && tool.id === 'openai.shell',
      ) as { args?: { environment?: { type?: string } } } | undefined
    )?.args?.environment?.type;

    const isShellProviderExecuted =
      shellToolEnvType === 'containerAuto' ||
      shellToolEnvType === 'containerReference';

    return {
      webSearchToolName,
      args: {
        ...baseArgs,
        tools: openaiTools,
        tool_choice: openaiToolChoice,
      },
      warnings: [...warnings, ...toolWarnings],
      store,
      toolNameMapping,
      providerOptionsName,
      isShellProviderExecuted,
      convertPromptToInput,
      diagnosticLog,
    };
  }

  async doGenerate(
    options: LanguageModelV4CallOptions,
  ): Promise<LanguageModelV4GenerateResult> {
    const {
      args: body,
      warnings,
      webSearchToolName,
      toolNameMapping,
      providerOptionsName,
      isShellProviderExecuted,
      convertPromptToInput,
      diagnosticLog,
    } = await this.getArgs(options);
    let requestBody = body;
    const url = this.config.url({
      path: '/responses',
      modelId: this.modelId,
    });

    const approvalRequestIdToDummyToolCallIdFromPrompt =
      extractApprovalRequestIdToToolCallIdMapping(options.prompt);

    let responseHeaders: Record<string, string> | undefined;
    let response!: OpenAIResponsesResponseWithOutput;
    let rawResponse: unknown;
    const accumulatedVisibleOutput: OpenAIResponsesOutput = [];
    const request = ++codexToolSearchRequestSequence;

    for (let round = 0; round < MAX_CODEX_TOOL_SEARCH_ROUNDS; round++) {
      if (round > 0 && diagnosticLog != null) {
        diagnosticLog.hiddenRoundCount += 1;
      }
      const result = await retryWithOpenCodePolicy({
        maxRetries: round === 0 ? 0 : OPEN_CODE_RETRY_MAX_RETRIES,
        abortSignal: options.abortSignal,
        execute: async attempt => {
          if (diagnosticLog != null) {
            diagnosticLog.requestCount += 1;
          }
          await appendDiagnosticLog(diagnosticLog, {
            event: 'request_start',
            request,
            round,
            attempt,
            store: requestBody.store,
            promptCacheKey: requestBody.prompt_cache_key,
            inputItems: summarizeDiagnosticItems(
              Array.isArray(requestBody.input) ? requestBody.input : [],
            ),
          });
          logCodexToolSearchRequest({
            request,
            round,
            attempt,
            requestBody,
          });
          const result = await postJsonToApi({
            url,
            headers: combineHeaders(this.config.headers?.(), options.headers),
            body: requestBody,
            failedResponseHandler: openaiFailedResponseHandler,
            successfulResponseHandler: createJsonResponseHandler(
              openaiResponsesResponseSchema,
            ),
            abortSignal: options.abortSignal,
            fetch: this.config.fetch,
          });

          responseHeaders = result.responseHeaders;
          rawResponse = result.rawValue;
          const response = result.value;

          if (response.error) {
            throw new APICallError({
              message: response.error.message,
              url,
              requestBodyValues: requestBody,
              statusCode: getOpenCodeResponseErrorStatusCode(response.error),
              responseHeaders,
              responseBody: rawResponse as string,
              data: { error: response.error },
              isRetryable: false,
            });
          }

          if (response.output == null) {
            const detail = response.incomplete_details?.reason;
            throw new APICallError({
              message: detail
                ? `Responses API returned no output (${detail})`
                : 'Responses API returned no output',
              url,
              requestBodyValues: requestBody,
              statusCode: 500,
              responseHeaders,
              responseBody: rawResponse as string,
              data: { response },
              isRetryable: false,
            });
          }

          const responseWithOutput = response as typeof response & {
            output: NonNullable<typeof response.output>;
          };
          return { ...result, value: responseWithOutput };
        },
        onError: async ({ error, attempt, retryable, willRetry, delayMs }) => {
          await appendDiagnosticLog(diagnosticLog, {
            event: 'request_error',
            request,
            round,
            attempt,
            retryableByOpenCode: retryable,
            willRetryInCompat: willRetry,
            ...(delayMs != null ? { retryDelayMs: delayMs } : {}),
            error:
              error instanceof Error
                ? { name: error.name, message: error.message }
                : String(error),
          });
        },
      });

      responseHeaders = result.responseHeaders;
      response = result.value as OpenAIResponsesResponseWithOutput;
      rawResponse = result.rawValue;

      await appendDiagnosticLog(diagnosticLog, {
        event: 'response',
        request,
        round,
        responseId: response.id,
        outputItems: summarizeDiagnosticOutput(response.output ?? []),
        error: response.error?.message,
      });

      const toolSearchOutputCallIds = new Set(
        response.output.flatMap(part =>
          part.type === 'tool_search_output' && part.call_id != null
            ? [part.call_id]
            : [],
        ),
      );
      const pendingToolSearchCalls = response.output.filter(
        (part): part is OpenAIResponsesToolSearchCall =>
          part.type === 'tool_search_call' &&
          !toolSearchOutputCallIds.has(part.call_id ?? part.id),
      );

      if (pendingToolSearchCalls.length === 0) {
        if (accumulatedVisibleOutput.length > 0) {
          response = {
            ...response,
            output: [...accumulatedVisibleOutput, ...response.output],
          };
        }
        break;
      }

      const pendingToolSearchCallIds = new Set(
        pendingToolSearchCalls.map(call => call.call_id ?? call.id),
      );
      const visibleOutput = response.output.filter(
        part =>
          part.type !== 'tool_search_call' ||
          !pendingToolSearchCallIds.has(part.call_id ?? part.id),
      );
      const hasFunctionCall = response.output.some(
        part => part.type === 'function_call',
      );

      if (hasFunctionCall) {
        // The next OpenCode turn can continue from A/B + function_call/output
        // without replaying this internally consumed tool-search call.
        response = {
          ...response,
          output: [...accumulatedVisibleOutput, ...visibleOutput],
        };
        break;
      }

      accumulatedVisibleOutput.push(...visibleOutput);

      if (round === MAX_CODEX_TOOL_SEARCH_ROUNDS - 1) {
        throw new APICallError({
          message:
            'Responses API returned too many client-side tool_search_call items',
          url,
          requestBodyValues: requestBody,
          statusCode: 500,
          responseHeaders,
          responseBody: rawResponse as string,
          isRetryable: false,
        });
      }

      const canonicalPrompt = prepareCodexToolSearchCanonicalPrompt({
        output: visibleOutput,
        providerOptionsName,
      });
      // Use the same prompt converter as the next OpenCode turn so stable
      // assistant output has an identical wire representation in both inputs.
      const { input: canonicalInput, warnings: canonicalInputWarnings } =
        await convertPromptToInput(canonicalPrompt);
      warnings.push(...canonicalInputWarnings);

      requestBody = {
        ...requestBody,
        input: [
          ...(Array.isArray(requestBody.input) ? requestBody.input : []),
          ...canonicalInput,
          ...prepareCodexToolSearchPassthroughInput({
            output: response.output,
            pendingToolSearchCallIds,
            excludeFunctionCalls: hasFunctionCall,
          }),
          ...pendingToolSearchCalls,
          ...pendingToolSearchCalls.map(call =>
            toolSearchOutput(call),
          ),
        ],
      };
    }

    if (response.output == null) {
      throw new APICallError({
        message: 'Responses API returned no output',
        url,
        requestBodyValues: requestBody,
        statusCode: 500,
        responseHeaders,
        responseBody: rawResponse as string,
        isRetryable: false,
      });
    }

    const output = response.output;

    const content: Array<LanguageModelV4Content> = [];
    const logprobs: Array<OpenAIResponsesLogprobs> = [];

    // flag that checks if there have been client-side tool calls (not executed by openai)
    let hasFunctionCall = false;
    const hostedToolSearchCallIds: string[] = [];

    // map response content to content array
    for (const part of output) {
      switch (part.type) {
        case 'reasoning': {
          for (const summary of getReasoningSummaryParts(part)) {
            content.push({
              type: 'reasoning' as const,
              text: summary.text,
              providerMetadata: {
                [providerOptionsName]: getReasoningProviderOptions(part),
              },
            });
          }
          break;
        }

        case 'image_generation_call': {
          content.push({
            type: 'tool-call',
            toolCallId: part.id,
            toolName: toolNameMapping.toCustomToolName('image_generation'),
            input: '{}',
            providerExecuted: true,
          });

          content.push({
            type: 'tool-result',
            toolCallId: part.id,
            toolName: toolNameMapping.toCustomToolName('image_generation'),
            result: {
              result: part.result,
            } satisfies InferSchema<typeof imageGenerationOutputSchema>,
          });

          break;
        }

        case 'tool_search_call': {
          if (part.execution === 'client') {
            break;
          }

          const toolCallId = part.call_id ?? part.id;
          const isHosted = part.execution === 'server';

          if (isHosted) {
            hostedToolSearchCallIds.push(toolCallId);
          }

          content.push({
            type: 'tool-call',
            toolCallId,
            toolName: toolNameMapping.toCustomToolName('tool_search'),
            input: JSON.stringify({
              arguments: part.arguments,
              call_id: part.call_id,
            } satisfies InferSchema<typeof toolSearchInputSchema>),
            ...(isHosted ? { providerExecuted: true } : {}),
            providerMetadata: {
              [providerOptionsName]: {
                itemId: part.id,
              },
            },
          });

          break;
        }

        case 'tool_search_output': {
          if (part.execution === 'client') {
            break;
          }

          const toolCallId =
            part.call_id ?? hostedToolSearchCallIds.shift() ?? part.id;

          content.push({
            type: 'tool-result',
            toolCallId,
            toolName: toolNameMapping.toCustomToolName('tool_search'),
            result: {
              tools: part.tools,
            } satisfies InferSchema<typeof toolSearchOutputSchema>,
            providerMetadata: {
              [providerOptionsName]: {
                itemId: part.id,
              },
            },
          });

          break;
        }

        case 'local_shell_call': {
          content.push({
            type: 'tool-call',
            toolCallId: part.call_id,
            toolName: toolNameMapping.toCustomToolName('local_shell'),
            input: JSON.stringify({
              action: part.action,
            } satisfies InferSchema<typeof localShellInputSchema>),
            providerMetadata: {
              [providerOptionsName]: {
                itemId: part.id,
              },
            },
          });

          break;
        }

        case 'shell_call': {
          content.push({
            type: 'tool-call',
            toolCallId: part.call_id,
            toolName: toolNameMapping.toCustomToolName('shell'),
            input: JSON.stringify({
              action: {
                commands: part.action.commands,
              },
            } satisfies InferSchema<typeof shellInputSchema>),
            ...(isShellProviderExecuted && { providerExecuted: true }),
            providerMetadata: {
              [providerOptionsName]: {
                itemId: part.id,
              },
            },
          });

          break;
        }

        case 'shell_call_output': {
          content.push({
            type: 'tool-result',
            toolCallId: part.call_id,
            toolName: toolNameMapping.toCustomToolName('shell'),
            result: {
              output: part.output.map(item => ({
                stdout: item.stdout,
                stderr: item.stderr,
                outcome:
                  item.outcome.type === 'exit'
                    ? {
                        type: 'exit' as const,
                        exitCode: item.outcome.exit_code,
                      }
                    : { type: 'timeout' as const },
              })),
            } satisfies InferSchema<typeof shellOutputSchema>,
          });
          break;
        }

        case 'message': {
          for (const contentPart of part.content) {
            if (
              options.providerOptions?.[providerOptionsName]?.logprobs &&
              contentPart.logprobs
            ) {
              logprobs.push(contentPart.logprobs);
            }

            const providerMetadata: SharedV4ProviderMetadata[string] =
              getTextProviderOptions(part, contentPart);

            content.push({
              type: 'text',
              text: contentPart.text,
              providerMetadata: {
                [providerOptionsName]: providerMetadata,
              },
            });

            for (const annotation of contentPart.annotations) {
              if (annotation.type === 'url_citation') {
                content.push({
                  type: 'source',
                  sourceType: 'url',
                  id: this.config.generateId?.() ?? generateId(),
                  url: annotation.url,
                  title: annotation.title,
                });
              } else if (annotation.type === 'file_citation') {
                content.push({
                  type: 'source',
                  sourceType: 'document',
                  id: this.config.generateId?.() ?? generateId(),
                  mediaType: 'text/plain',
                  title: annotation.filename,
                  filename: annotation.filename,
                  providerMetadata: {
                    [providerOptionsName]: {
                      type: annotation.type,
                      fileId: annotation.file_id,
                      index: annotation.index,
                    } satisfies Extract<
                      ResponsesSourceDocumentProviderMetadata,
                      { type: 'file_citation' }
                    >,
                  },
                });
              } else if (annotation.type === 'container_file_citation') {
                content.push({
                  type: 'source',
                  sourceType: 'document',
                  id: this.config.generateId?.() ?? generateId(),
                  mediaType: 'text/plain',
                  title: annotation.filename,
                  filename: annotation.filename,
                  providerMetadata: {
                    [providerOptionsName]: {
                      type: annotation.type,
                      fileId: annotation.file_id,
                      containerId: annotation.container_id,
                    } satisfies Extract<
                      ResponsesSourceDocumentProviderMetadata,
                      { type: 'container_file_citation' }
                    >,
                  },
                });
              } else if (annotation.type === 'file_path') {
                content.push({
                  type: 'source',
                  sourceType: 'document',
                  id: this.config.generateId?.() ?? generateId(),
                  mediaType: 'application/octet-stream',
                  title: annotation.file_id,
                  filename: annotation.file_id,
                  providerMetadata: {
                    [providerOptionsName]: {
                      type: annotation.type,
                      fileId: annotation.file_id,
                      index: annotation.index,
                    } satisfies Extract<
                      ResponsesSourceDocumentProviderMetadata,
                      { type: 'file_path' }
                    >,
                  },
                });
              }
            }
          }

          break;
        }

        case 'function_call': {
          hasFunctionCall = true;

          content.push({
            type: 'tool-call',
            toolCallId: part.call_id,
            toolName: toolNameMapping.toCustomToolName(part.name),
            input: part.arguments,
            providerMetadata: {
              [providerOptionsName]: {
                itemId: part.id,
                ...(part.namespace != null && { namespace: part.namespace }),
                ...(part.caller != null && {
                  caller:
                    part.caller.type === 'program'
                      ? {
                          type: 'program',
                          callerId: part.caller.caller_id,
                        }
                      : part.caller,
                }),
              },
            },
          });
          break;
        }

        case 'program': {
          content.push({
            type: 'tool-call',
            toolCallId: part.call_id,
            toolName: toolNameMapping.toCustomToolName(
              'programmatic_tool_calling',
            ),
            input: JSON.stringify({
              code: part.code,
              fingerprint: part.fingerprint,
            } satisfies InferSchema<typeof programmaticToolCallingInputSchema>),
            providerExecuted: true,
            providerMetadata: {
              [providerOptionsName]: {
                itemId: part.id,
              },
            },
          });
          break;
        }

        case 'program_output': {
          content.push({
            type: 'tool-result',
            toolCallId: part.call_id,
            toolName: toolNameMapping.toCustomToolName(
              'programmatic_tool_calling',
            ),
            result: {
              result: part.result,
              status: part.status,
            } satisfies InferSchema<typeof programmaticToolCallingOutputSchema>,
            providerMetadata: {
              [providerOptionsName]: {
                itemId: part.id,
              },
            },
          });
          break;
        }

        case 'custom_tool_call': {
          hasFunctionCall = true;
          const toolName = toolNameMapping.toCustomToolName(part.name);

          content.push({
            type: 'tool-call',
            toolCallId: part.call_id,
            toolName,
            input: JSON.stringify(part.input),
            providerMetadata: {
              [providerOptionsName]: {
                itemId: part.id,
              },
            },
          });
          break;
        }

        case 'web_search_call': {
          content.push({
            type: 'tool-call',
            toolCallId: part.id,
            toolName: toolNameMapping.toCustomToolName(
              webSearchToolName ?? 'web_search',
            ),
            input: JSON.stringify({}),
            providerExecuted: true,
          });

          content.push({
            type: 'tool-result',
            toolCallId: part.id,
            toolName: toolNameMapping.toCustomToolName(
              webSearchToolName ?? 'web_search',
            ),
            result: mapWebSearchOutput(part.action),
          });

          break;
        }

        case 'mcp_call': {
          const toolCallId =
            part.approval_request_id != null
              ? (approvalRequestIdToDummyToolCallIdFromPrompt[
                  part.approval_request_id
                ] ?? part.id)
              : part.id;

          const toolName = `mcp.${part.name}`;

          content.push({
            type: 'tool-call',
            toolCallId,
            toolName,
            input: part.arguments,
            providerExecuted: true,
            dynamic: true,
          });

          content.push({
            type: 'tool-result',
            toolCallId,
            toolName,
            result: {
              type: 'call',
              serverLabel: part.server_label,
              name: part.name,
              arguments: part.arguments,
              ...(part.output != null ? { output: part.output } : {}),
              ...(part.error != null
                ? { error: part.error as unknown as JSONValue }
                : {}),
            } satisfies InferSchema<typeof mcpOutputSchema>,
            providerMetadata: {
              [providerOptionsName]: {
                itemId: part.id,
              },
            },
          });
          break;
        }

        case 'mcp_list_tools': {
          // skip
          break;
        }

        case 'mcp_approval_request': {
          const approvalRequestId = part.approval_request_id ?? part.id;
          const dummyToolCallId = this.config.generateId?.() ?? generateId();
          const toolName = `mcp.${part.name}`;

          content.push({
            type: 'tool-call',
            toolCallId: dummyToolCallId,
            toolName,
            input: part.arguments,
            providerExecuted: true,
            dynamic: true,
          });

          content.push({
            type: 'tool-approval-request',
            approvalId: approvalRequestId,
            toolCallId: dummyToolCallId,
          } satisfies LanguageModelV4ToolApprovalRequest);
          break;
        }

        case 'computer_call': {
          if (part.call_id == null) {
            content.push({
              type: 'tool-call',
              toolCallId: part.id,
              toolName: toolNameMapping.toCustomToolName('computer_use'),
              input: '',
              providerExecuted: true,
            });

            content.push({
              type: 'tool-result',
              toolCallId: part.id,
              toolName: toolNameMapping.toCustomToolName('computer_use'),
              result: {
                type: 'computer_use_tool_result',
                status: part.status,
              },
            });
            break;
          }

          hasFunctionCall = true;
          const toolName = toolNameMapping.toCustomToolName('computer');

          content.push({
            type: 'tool-call',
            toolCallId: part.call_id,
            toolName,
            input: JSON.stringify(mapComputerCallInput(part)),
            providerMetadata: {
              [providerOptionsName]: {
                itemId: part.id,
              },
            },
          });
          break;
        }

        case 'file_search_call': {
          content.push({
            type: 'tool-call',
            toolCallId: part.id,
            toolName: toolNameMapping.toCustomToolName('file_search'),
            input: '{}',
            providerExecuted: true,
          });

          content.push({
            type: 'tool-result',
            toolCallId: part.id,
            toolName: toolNameMapping.toCustomToolName('file_search'),
            result: {
              queries: part.queries,
              results:
                part.results?.map(result => ({
                  attributes: result.attributes,
                  fileId: result.file_id,
                  filename: result.filename,
                  score: result.score,
                  text: result.text,
                })) ?? null,
            } satisfies InferSchema<typeof fileSearchOutputSchema>,
          });
          break;
        }

        case 'code_interpreter_call': {
          content.push({
            type: 'tool-call',
            toolCallId: part.id,
            toolName: toolNameMapping.toCustomToolName('code_interpreter'),
            input: JSON.stringify({
              code: part.code,
              containerId: part.container_id,
            } satisfies InferSchema<typeof codeInterpreterInputSchema>),
            providerExecuted: true,
          });

          content.push({
            type: 'tool-result',
            toolCallId: part.id,
            toolName: toolNameMapping.toCustomToolName('code_interpreter'),
            result: {
              outputs: part.outputs,
            } satisfies InferSchema<typeof codeInterpreterOutputSchema>,
          });
          break;
        }

        case 'apply_patch_call': {
          content.push({
            type: 'tool-call',
            toolCallId: part.call_id,
            toolName: toolNameMapping.toCustomToolName('apply_patch'),
            input: JSON.stringify({
              callId: part.call_id,
              operation: part.operation,
            } satisfies InferSchema<typeof applyPatchInputSchema>),
            providerMetadata: {
              [providerOptionsName]: {
                itemId: part.id,
              },
            },
          });

          break;
        }

        case 'compaction': {
          content.push({
            type: 'custom',
            kind: 'openai.compaction',
            providerMetadata: {
              [providerOptionsName]: {
                type: 'compaction',
                itemId: part.id,
                encryptedContent: part.encrypted_content,
              } satisfies ResponsesCompactionProviderMetadata,
            },
          });
          break;
        }
      }
    }

    const providerMetadata: SharedV4ProviderMetadata = {
      [providerOptionsName]: {
        responseId: response.id,
        ...(logprobs.length > 0 ? { logprobs } : {}),
        ...(typeof response.service_tier === 'string'
          ? { serviceTier: response.service_tier }
          : {}),
        ...(response.reasoning?.context != null
          ? { reasoningContext: response.reasoning.context }
          : {}),
      } satisfies ResponsesProviderMetadata,
    };

    if (diagnosticLog != null) {
      const debugMetadata: Record<string, JSONValue> = {
        requestCount: diagnosticLog.requestCount,
        hiddenRoundCount: diagnosticLog.hiddenRoundCount,
      };
      const debugProviderMetadata = {
        debug: debugMetadata,
      };
      providerMetadata[TOOL_SEARCH_COMPAT_PROVIDER_METADATA] =
        debugProviderMetadata;

      if (content.length > 0) {
        const firstContent = content[0] as LanguageModelV4Content & {
          providerMetadata?: SharedV4ProviderMetadata;
        };
        content[0] = {
          ...firstContent,
          providerMetadata: {
            ...firstContent.providerMetadata,
            [TOOL_SEARCH_COMPAT_PROVIDER_METADATA]: debugProviderMetadata,
          },
        } as LanguageModelV4Content;
      }
    }

    const usage = response.usage!; // defined when there is no error

    return {
      content,
      finishReason: {
        unified: mapOpenAIResponseFinishReason({
          finishReason: response.incomplete_details?.reason,
          hasFunctionCall,
        }),
        raw: response.incomplete_details?.reason ?? undefined,
      },
      usage: convertOpenAIResponsesUsage(usage),
      request: { body: requestBody },
      response: {
        id: response.id,
        timestamp: new Date(response.created_at! * 1000),
        modelId: response.model,
        headers: responseHeaders,
        body: rawResponse,
      },
      providerMetadata,
      warnings,
    };
  }

  async doStream(
    options: LanguageModelV4CallOptions,
  ): Promise<LanguageModelV4StreamResult> {
    const generated = await this.doGenerate(options);

    return {
      stream: createBufferedLanguageModelStream(generated),
      request: generated.request,
      response: generated.response,
    };

    /*
     * The native Responses stream path is retained below for reference while
     * the fork uses the generate path to consume client-side tool_search calls.
     */
    const {
      args: body,
      warnings,
      webSearchToolName,
      toolNameMapping,
      store,
      providerOptionsName,
      isShellProviderExecuted,
    } = await this.getArgs(options);

    const url = this.config.url({
      path: '/responses',
      modelId: this.modelId,
    });

    const { responseHeaders, value: response } = await postJsonToApi({
      url,
      headers: combineHeaders(this.config.headers?.(), options.headers),
      body: {
        ...body,
        stream: true,
      },
      failedResponseHandler: openaiFailedResponseHandler,
      successfulResponseHandler: createEventSourceResponseHandler(
        openaiResponsesChunkSchema,
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    const checkedResponse = await throwIfOpenAIStreamErrorBeforeOutput({
      stream: response,
      getError: chunk =>
        isErrorChunk(chunk) ||
        (isResponseFailedChunk(chunk) && chunk.response.error != null)
          ? chunk
          : undefined,
      isOutputChunk: isResponseOutputChunk,
      isAcceptedChunk: isResponseInProgressChunk,
      url,
      requestBodyValues: body,
      responseHeaders,
    });

    const self = this;

    const approvalRequestIdToDummyToolCallIdFromPrompt =
      extractApprovalRequestIdToToolCallIdMapping(options.prompt);

    const approvalRequestIdToDummyToolCallIdFromStream = new Map<
      string,
      string
    >();

    let finishReason: LanguageModelV4FinishReason = {
      unified: 'other',
      raw: undefined,
    };
    let usage: OpenAIResponsesUsage | undefined = undefined;
    const logprobs: Array<OpenAIResponsesLogprobs> = [];
    let responseId: string | null = null;

    const ongoingToolCalls: Record<
      number,
      | {
          toolName: string;
          toolCallId: string;
          codeInterpreter?: {
            containerId: string;
          };
          applyPatch?: {
            hasDiff: boolean;
            endEmitted: boolean;
          };
          toolSearchExecution?: 'server' | 'client';
        }
      | undefined
    > = {};

    // set annotations in 'text-end' part providerMetadata.
    const ongoingAnnotations: Array<
      Extract<
        OpenAIResponsesChunk,
        { type: 'response.output_text.annotation.added' }
      >['annotation']
    > = [];

    // track the phase of the current message being streamed
    let activeMessagePhase: 'commentary' | 'final_answer' | undefined;

    // flag that checks if there have been client-side tool calls (not executed by openai)
    let hasFunctionCall = false;

    const activeReasoning: Record<
      string,
      {
        encryptedContent?: string | null;
        // summary index as string to reasoning part state:
        summaryParts: Record<string, 'active' | 'can-conclude' | 'concluded'>;
      }
    > = {};

    // OpenAI-compatible providers can rotate opaque item ids between events.
    // output_index remains stable for the lifetime of an output item.
    const activeOutputItemIds: Record<number, string | undefined> = {};
    const resolveOutputItemId = ({
      itemId,
      outputIndex,
    }: {
      itemId: string;
      outputIndex?: number | null;
    }) =>
      outputIndex == null
        ? itemId
        : (activeOutputItemIds[outputIndex] ?? itemId);

    let serviceTier: string | undefined;
    let reasoningContext: ResponsesProviderMetadata['reasoningContext'];
    const hostedToolSearchCallIds: string[] = [];
    let encounteredStreamError = false;

    const result = {
      stream: checkedResponse.pipeThrough(
        new TransformStream<
          ParseResult<OpenAIResponsesChunk>,
          LanguageModelV4StreamPart
        >({
          start(controller) {
            controller.enqueue({ type: 'stream-start', warnings });
          },

          transform(chunk, controller) {
            if (options.includeRawChunks) {
              controller.enqueue({ type: 'raw', rawValue: chunk.rawValue });
            }

            // handle failed chunk parsing / validation:
            if (!chunk.success) {
              const error = isOpenAIChatCompletionChunk(chunk.rawValue)
                ? createOpenAIResponsesChatCompletionsMismatchError({
                    value: chunk.rawValue,
                    cause: chunk.error,
                    url,
                    requestBodyValues: body,
                    responseHeaders,
                  })
                : chunk.error;

              finishReason = { unified: 'error', raw: undefined };
              controller.enqueue({ type: 'error', error });
              return;
            }

            const value = chunk.value;

            if (isResponseOutputItemAddedChunk(value)) {
              if (value.item.type === 'function_call') {
                const toolName = toolNameMapping.toCustomToolName(
                  value.item.name,
                );
                ongoingToolCalls[value.output_index] = {
                  toolName,
                  toolCallId: value.item.call_id,
                };

                controller.enqueue({
                  type: 'tool-input-start',
                  id: value.item.call_id,
                  toolName,
                });
              } else if (value.item.type === 'custom_tool_call') {
                const toolName = toolNameMapping.toCustomToolName(
                  value.item.name,
                );
                ongoingToolCalls[value.output_index] = {
                  toolName,
                  toolCallId: value.item.call_id,
                };

                controller.enqueue({
                  type: 'tool-input-start',
                  id: value.item.call_id,
                  toolName,
                });
              } else if (value.item.type === 'web_search_call') {
                ongoingToolCalls[value.output_index] = {
                  toolName: toolNameMapping.toCustomToolName(
                    webSearchToolName ?? 'web_search',
                  ),
                  toolCallId: value.item.id,
                };

                controller.enqueue({
                  type: 'tool-input-start',
                  id: value.item.id,
                  toolName: toolNameMapping.toCustomToolName(
                    webSearchToolName ?? 'web_search',
                  ),
                  providerExecuted: true,
                });

                controller.enqueue({
                  type: 'tool-input-end',
                  id: value.item.id,
                });

                controller.enqueue({
                  type: 'tool-call',
                  toolCallId: value.item.id,
                  toolName: toolNameMapping.toCustomToolName(
                    webSearchToolName ?? 'web_search',
                  ),
                  input: JSON.stringify({}),
                  providerExecuted: true,
                });
              } else if (value.item.type === 'computer_call') {
                const toolCallId = value.item.call_id ?? value.item.id;
                ongoingToolCalls[value.output_index] = {
                  toolName: toolNameMapping.toCustomToolName('computer'),
                  toolCallId,
                };

                controller.enqueue({
                  type: 'tool-input-start',
                  id: toolCallId,
                  toolName: toolNameMapping.toCustomToolName('computer'),
                });
              } else if (value.item.type === 'code_interpreter_call') {
                ongoingToolCalls[value.output_index] = {
                  toolName:
                    toolNameMapping.toCustomToolName('code_interpreter'),
                  toolCallId: value.item.id,
                  codeInterpreter: {
                    containerId: value.item.container_id,
                  },
                };

                controller.enqueue({
                  type: 'tool-input-start',
                  id: value.item.id,
                  toolName:
                    toolNameMapping.toCustomToolName('code_interpreter'),
                  providerExecuted: true,
                });

                controller.enqueue({
                  type: 'tool-input-delta',
                  id: value.item.id,
                  delta: `{"containerId":"${value.item.container_id}","code":"`,
                });
              } else if (value.item.type === 'file_search_call') {
                controller.enqueue({
                  type: 'tool-call',
                  toolCallId: value.item.id,
                  toolName: toolNameMapping.toCustomToolName('file_search'),
                  input: '{}',
                  providerExecuted: true,
                });
              } else if (value.item.type === 'image_generation_call') {
                controller.enqueue({
                  type: 'tool-call',
                  toolCallId: value.item.id,
                  toolName:
                    toolNameMapping.toCustomToolName('image_generation'),
                  input: '{}',
                  providerExecuted: true,
                });
              } else if (value.item.type === 'tool_search_call') {
                const toolCallId = value.item.id;
                const toolName =
                  toolNameMapping.toCustomToolName('tool_search');
                const isHosted = value.item.execution === 'server';

                ongoingToolCalls[value.output_index] = {
                  toolName,
                  toolCallId,
                  toolSearchExecution: value.item.execution ?? 'server',
                };

                if (isHosted) {
                  controller.enqueue({
                    type: 'tool-input-start',
                    id: toolCallId,
                    toolName,
                    providerExecuted: true,
                  });
                }
              } else if (value.item.type === 'tool_search_output') {
                // handled on output_item.done so we can pair it with the call
              } else if (
                value.item.type === 'mcp_call' ||
                value.item.type === 'mcp_list_tools' ||
                value.item.type === 'mcp_approval_request'
              ) {
                // Emit MCP tool-call/approval parts on output_item.done instead, so we can:
                // - alias mcp_call IDs when an approval_request_id is present
                // - emit a proper tool-approval-request part for MCP approvals
              } else if (value.item.type === 'apply_patch_call') {
                const { call_id: callId, operation } = value.item;

                ongoingToolCalls[value.output_index] = {
                  toolName: toolNameMapping.toCustomToolName('apply_patch'),
                  toolCallId: callId,
                  applyPatch: {
                    // delete_file doesn't have diff
                    hasDiff: operation.type === 'delete_file',
                    endEmitted: operation.type === 'delete_file',
                  },
                };

                controller.enqueue({
                  type: 'tool-input-start',
                  id: callId,
                  toolName: toolNameMapping.toCustomToolName('apply_patch'),
                });

                if (operation.type === 'delete_file') {
                  const inputString = JSON.stringify({
                    callId,
                    operation,
                  } satisfies InferSchema<typeof applyPatchInputSchema>);

                  controller.enqueue({
                    type: 'tool-input-delta',
                    id: callId,
                    delta: inputString,
                  });

                  controller.enqueue({
                    type: 'tool-input-end',
                    id: callId,
                  });
                } else {
                  controller.enqueue({
                    type: 'tool-input-delta',
                    id: callId,
                    delta: `{"callId":"${escapeJSONDelta(callId)}","operation":{"type":"${escapeJSONDelta(operation.type)}","path":"${escapeJSONDelta(operation.path)}","diff":"`,
                  });
                }
              } else if (value.item.type === 'shell_call') {
                ongoingToolCalls[value.output_index] = {
                  toolName: toolNameMapping.toCustomToolName('shell'),
                  toolCallId: value.item.call_id,
                };
              } else if (value.item.type === 'shell_call_output') {
                // shell_call_output is handled in output_item.done
              } else if (value.item.type === 'message') {
                activeOutputItemIds[value.output_index] = value.item.id;
                ongoingAnnotations.splice(0, ongoingAnnotations.length);
                activeMessagePhase = value.item.phase ?? undefined;
                controller.enqueue({
                  type: 'text-start',
                  id: value.item.id,
                  providerMetadata: {
                    [providerOptionsName]: {
                      itemId: value.item.id,
                      ...(value.item.phase != null && {
                        phase: value.item.phase,
                      }),
                    },
                  },
                });
              } else if (
                isResponseOutputItemAddedChunk(value) &&
                value.item.type === 'reasoning'
              ) {
                activeOutputItemIds[value.output_index] = value.item.id;
                activeReasoning[value.item.id] = {
                  encryptedContent: value.item.encrypted_content,
                  summaryParts: { 0: 'active' },
                };

                controller.enqueue({
                  type: 'reasoning-start',
                  id: `${value.item.id}:0`,
                  providerMetadata: {
                    [providerOptionsName]: {
                      itemId: value.item.id,
                      reasoningEncryptedContent:
                        value.item.encrypted_content ?? null,
                    } satisfies ResponsesReasoningProviderMetadata,
                  },
                });
              }
            } else if (isResponseOutputItemDoneChunk(value)) {
              if (value.item.type === 'message') {
                const itemId = resolveOutputItemId({
                  itemId: value.item.id,
                  outputIndex: value.output_index,
                });
                const phase = value.item.phase ?? activeMessagePhase;
                activeMessagePhase = undefined;
                controller.enqueue({
                  type: 'text-end',
                  id: itemId,
                  providerMetadata: {
                    [providerOptionsName]: {
                      itemId,
                      ...(phase != null && { phase }),
                      ...(ongoingAnnotations.length > 0 && {
                        annotations: ongoingAnnotations,
                      }),
                    } satisfies ResponsesTextProviderMetadata,
                  },
                });
                activeOutputItemIds[value.output_index] = undefined;
              } else if (value.item.type === 'function_call') {
                ongoingToolCalls[value.output_index] = undefined;
                hasFunctionCall = true;
                const toolName = toolNameMapping.toCustomToolName(
                  value.item.name,
                );

                controller.enqueue({
                  type: 'tool-input-end',
                  id: value.item.call_id,
                  ...(value.item.namespace != null && {
                    providerMetadata: {
                      [providerOptionsName]: {
                        namespace: value.item.namespace,
                      },
                    },
                  }),
                });

                controller.enqueue({
                  type: 'tool-call',
                  toolCallId: value.item.call_id,
                  toolName,
                  input: value.item.arguments,
                  providerMetadata: {
                    [providerOptionsName]: {
                      itemId: value.item.id,
                      ...(value.item.namespace != null && {
                        namespace: value.item.namespace,
                      }),
                      ...(value.item.caller != null && {
                        caller:
                          value.item.caller.type === 'program'
                            ? {
                                type: 'program',
                                callerId: value.item.caller.caller_id,
                              }
                            : value.item.caller,
                      }),
                    },
                  },
                });
              } else if (value.item.type === 'program') {
                controller.enqueue({
                  type: 'tool-call',
                  toolCallId: value.item.call_id,
                  toolName: toolNameMapping.toCustomToolName(
                    'programmatic_tool_calling',
                  ),
                  input: JSON.stringify({
                    code: value.item.code,
                    fingerprint: value.item.fingerprint,
                  } satisfies InferSchema<
                    typeof programmaticToolCallingInputSchema
                  >),
                  providerExecuted: true,
                  providerMetadata: {
                    [providerOptionsName]: {
                      itemId: value.item.id,
                    },
                  },
                });
              } else if (value.item.type === 'program_output') {
                controller.enqueue({
                  type: 'tool-result',
                  toolCallId: value.item.call_id,
                  toolName: toolNameMapping.toCustomToolName(
                    'programmatic_tool_calling',
                  ),
                  result: {
                    result: value.item.result,
                    status: value.item.status,
                  } satisfies InferSchema<
                    typeof programmaticToolCallingOutputSchema
                  >,
                  providerMetadata: {
                    [providerOptionsName]: {
                      itemId: value.item.id,
                    },
                  },
                });
              } else if (value.item.type === 'custom_tool_call') {
                ongoingToolCalls[value.output_index] = undefined;
                hasFunctionCall = true;
                const toolName = toolNameMapping.toCustomToolName(
                  value.item.name,
                );

                controller.enqueue({
                  type: 'tool-input-end',
                  id: value.item.call_id,
                });

                controller.enqueue({
                  type: 'tool-call',
                  toolCallId: value.item.call_id,
                  toolName,
                  input: JSON.stringify(value.item.input),
                  providerMetadata: {
                    [providerOptionsName]: {
                      itemId: value.item.id,
                    },
                  },
                });
              } else if (value.item.type === 'web_search_call') {
                ongoingToolCalls[value.output_index] = undefined;

                controller.enqueue({
                  type: 'tool-result',
                  toolCallId: value.item.id,
                  toolName: toolNameMapping.toCustomToolName(
                    webSearchToolName ?? 'web_search',
                  ),
                  result: mapWebSearchOutput(value.item.action),
                });
              } else if (value.item.type === 'computer_call') {
                ongoingToolCalls[value.output_index] = undefined;

                if (value.item.call_id == null) {
                  controller.enqueue({
                    type: 'tool-input-end',
                    id: value.item.id,
                  });
                  controller.enqueue({
                    type: 'tool-call',
                    toolCallId: value.item.id,
                    toolName: toolNameMapping.toCustomToolName('computer_use'),
                    input: '',
                    providerExecuted: true,
                  });
                  controller.enqueue({
                    type: 'tool-result',
                    toolCallId: value.item.id,
                    toolName: toolNameMapping.toCustomToolName('computer_use'),
                    result: {
                      type: 'computer_use_tool_result',
                      status: value.item.status,
                    },
                  });
                  return;
                }

                hasFunctionCall = true;
                const toolName = toolNameMapping.toCustomToolName('computer');
                const input = JSON.stringify(mapComputerCallInput(value.item));

                controller.enqueue({
                  type: 'tool-input-delta',
                  id: value.item.call_id,
                  delta: input,
                });

                controller.enqueue({
                  type: 'tool-input-end',
                  id: value.item.call_id,
                });

                controller.enqueue({
                  type: 'tool-call',
                  toolCallId: value.item.call_id,
                  toolName,
                  input,
                  providerMetadata: {
                    [providerOptionsName]: {
                      itemId: value.item.id,
                    },
                  },
                });
              } else if (value.item.type === 'file_search_call') {
                ongoingToolCalls[value.output_index] = undefined;

                controller.enqueue({
                  type: 'tool-result',
                  toolCallId: value.item.id,
                  toolName: toolNameMapping.toCustomToolName('file_search'),
                  result: {
                    queries: value.item.queries,
                    results:
                      value.item.results?.map(result => ({
                        attributes: result.attributes,
                        fileId: result.file_id,
                        filename: result.filename,
                        score: result.score,
                        text: result.text,
                      })) ?? null,
                  } satisfies InferSchema<typeof fileSearchOutputSchema>,
                });
              } else if (value.item.type === 'code_interpreter_call') {
                ongoingToolCalls[value.output_index] = undefined;

                controller.enqueue({
                  type: 'tool-result',
                  toolCallId: value.item.id,
                  toolName:
                    toolNameMapping.toCustomToolName('code_interpreter'),
                  result: {
                    outputs: value.item.outputs,
                  } satisfies InferSchema<typeof codeInterpreterOutputSchema>,
                });
              } else if (value.item.type === 'image_generation_call') {
                controller.enqueue({
                  type: 'tool-result',
                  toolCallId: value.item.id,
                  toolName:
                    toolNameMapping.toCustomToolName('image_generation'),
                  result: {
                    result: value.item.result,
                  } satisfies InferSchema<typeof imageGenerationOutputSchema>,
                });
              } else if (value.item.type === 'tool_search_call') {
                const toolCall = ongoingToolCalls[value.output_index];
                const isHosted = value.item.execution === 'server';

                // Client-side tool_search calls are consumed by doGenerate's
                // follow-up loop. The native SSE path is currently retained
                // only as reference and must not expose this protocol item.
                if (!isHosted) {
                  ongoingToolCalls[value.output_index] = undefined;
                  return;
                }

                if (toolCall != null) {
                  const toolCallId = isHosted
                    ? toolCall.toolCallId
                    : (value.item.call_id ?? value.item.id);

                  if (isHosted) {
                    hostedToolSearchCallIds.push(toolCallId);
                  } else {
                    controller.enqueue({
                      type: 'tool-input-start',
                      id: toolCallId,
                      toolName: toolCall.toolName,
                    });
                  }

                  controller.enqueue({
                    type: 'tool-input-end',
                    id: toolCallId,
                  });

                  controller.enqueue({
                    type: 'tool-call',
                    toolCallId,
                    toolName: toolCall.toolName,
                    input: JSON.stringify({
                      arguments: value.item.arguments,
                      call_id: isHosted ? null : toolCallId,
                    } satisfies InferSchema<typeof toolSearchInputSchema>),
                    ...(isHosted ? { providerExecuted: true } : {}),
                    providerMetadata: {
                      [providerOptionsName]: {
                        itemId: value.item.id,
                      },
                    },
                  });
                }

                ongoingToolCalls[value.output_index] = undefined;
              } else if (value.item.type === 'tool_search_output') {
                // Client-side output is generated by doGenerate's follow-up
                // loop and is never exposed through the provider stream.
                if (value.item.execution === 'client') {
                  return;
                }

                const toolCallId =
                  value.item.call_id ??
                  hostedToolSearchCallIds.shift() ??
                  value.item.id;

                controller.enqueue({
                  type: 'tool-result',
                  toolCallId,
                  toolName: toolNameMapping.toCustomToolName('tool_search'),
                  result: {
                    tools: value.item.tools,
                  } satisfies InferSchema<typeof toolSearchOutputSchema>,
                  providerMetadata: {
                    [providerOptionsName]: {
                      itemId: value.item.id,
                    },
                  },
                });
              } else if (value.item.type === 'mcp_call') {
                ongoingToolCalls[value.output_index] = undefined;

                const approvalRequestId =
                  value.item.approval_request_id ?? undefined;

                // when MCP tools require approval, we track them with our own
                // tool call IDs and then map OpenAI's approval_request_id back to our ID so results match.
                const aliasedToolCallId =
                  approvalRequestId != null
                    ? (approvalRequestIdToDummyToolCallIdFromStream.get(
                        approvalRequestId,
                      ) ??
                      approvalRequestIdToDummyToolCallIdFromPrompt[
                        approvalRequestId
                      ] ??
                      value.item.id)
                    : value.item.id;

                const toolName = `mcp.${value.item.name}`;

                controller.enqueue({
                  type: 'tool-call',
                  toolCallId: aliasedToolCallId,
                  toolName,
                  input: value.item.arguments,
                  providerExecuted: true,
                  dynamic: true,
                });

                controller.enqueue({
                  type: 'tool-result',
                  toolCallId: aliasedToolCallId,
                  toolName,
                  result: {
                    type: 'call',
                    serverLabel: value.item.server_label,
                    name: value.item.name,
                    arguments: value.item.arguments,
                    ...(value.item.output != null
                      ? { output: value.item.output }
                      : {}),
                    ...(value.item.error != null
                      ? { error: value.item.error as unknown as JSONValue }
                      : {}),
                  } satisfies InferSchema<typeof mcpOutputSchema>,
                  providerMetadata: {
                    [providerOptionsName]: {
                      itemId: value.item.id,
                    },
                  },
                });
              } else if (value.item.type === 'mcp_list_tools') {
                // Skip listTools - we don't expose this to the UI or send it back
                ongoingToolCalls[value.output_index] = undefined;

                // skip
              } else if (value.item.type === 'apply_patch_call') {
                const toolCall = ongoingToolCalls[value.output_index];
                if (
                  toolCall?.applyPatch &&
                  !toolCall.applyPatch.endEmitted &&
                  value.item.operation.type !== 'delete_file'
                ) {
                  if (!toolCall.applyPatch.hasDiff) {
                    controller.enqueue({
                      type: 'tool-input-delta',
                      id: toolCall.toolCallId,
                      delta: escapeJSONDelta(value.item.operation.diff),
                    });
                  }

                  controller.enqueue({
                    type: 'tool-input-delta',
                    id: toolCall.toolCallId,
                    delta: '"}}',
                  });

                  controller.enqueue({
                    type: 'tool-input-end',
                    id: toolCall.toolCallId,
                  });

                  toolCall.applyPatch.endEmitted = true;
                }

                // Emit the final tool-call with complete diff when status is 'completed'
                if (toolCall && value.item.status === 'completed') {
                  controller.enqueue({
                    type: 'tool-call',
                    toolCallId: toolCall.toolCallId,
                    toolName: toolNameMapping.toCustomToolName('apply_patch'),
                    input: JSON.stringify({
                      callId: value.item.call_id,
                      operation: value.item.operation,
                    } satisfies InferSchema<typeof applyPatchInputSchema>),
                    providerMetadata: {
                      [providerOptionsName]: {
                        itemId: value.item.id,
                      },
                    },
                  });
                }

                ongoingToolCalls[value.output_index] = undefined;
              } else if (value.item.type === 'mcp_approval_request') {
                ongoingToolCalls[value.output_index] = undefined;

                const dummyToolCallId =
                  self.config.generateId?.() ?? generateId();
                const approvalRequestId =
                  value.item.approval_request_id ?? value.item.id;
                approvalRequestIdToDummyToolCallIdFromStream.set(
                  approvalRequestId,
                  dummyToolCallId,
                );

                const toolName = `mcp.${value.item.name}`;

                controller.enqueue({
                  type: 'tool-call',
                  toolCallId: dummyToolCallId,
                  toolName,
                  input: value.item.arguments,
                  providerExecuted: true,
                  dynamic: true,
                });

                controller.enqueue({
                  type: 'tool-approval-request',
                  approvalId: approvalRequestId,
                  toolCallId: dummyToolCallId,
                });
              } else if (value.item.type === 'local_shell_call') {
                ongoingToolCalls[value.output_index] = undefined;

                controller.enqueue({
                  type: 'tool-call',
                  toolCallId: value.item.call_id,
                  toolName: toolNameMapping.toCustomToolName('local_shell'),
                  input: JSON.stringify({
                    action: {
                      type: 'exec',
                      command: value.item.action.command,
                      timeoutMs: value.item.action.timeout_ms,
                      user: value.item.action.user,
                      workingDirectory: value.item.action.working_directory,
                      env: value.item.action.env,
                    },
                  } satisfies InferSchema<typeof localShellInputSchema>),
                  providerMetadata: {
                    [providerOptionsName]: { itemId: value.item.id },
                  },
                });
              } else if (value.item.type === 'shell_call') {
                ongoingToolCalls[value.output_index] = undefined;

                controller.enqueue({
                  type: 'tool-call',
                  toolCallId: value.item.call_id,
                  toolName: toolNameMapping.toCustomToolName('shell'),
                  input: JSON.stringify({
                    action: {
                      commands: value.item.action.commands,
                    },
                  } satisfies InferSchema<typeof shellInputSchema>),
                  ...(isShellProviderExecuted && {
                    providerExecuted: true,
                  }),
                  providerMetadata: {
                    [providerOptionsName]: { itemId: value.item.id },
                  },
                });
              } else if (value.item.type === 'shell_call_output') {
                controller.enqueue({
                  type: 'tool-result',
                  toolCallId: value.item.call_id,
                  toolName: toolNameMapping.toCustomToolName('shell'),
                  result: {
                    output: value.item.output.map(
                      (item: {
                        stdout: string;
                        stderr: string;
                        outcome:
                          | { type: 'exit'; exit_code: number }
                          | { type: 'timeout' };
                      }) => ({
                        stdout: item.stdout,
                        stderr: item.stderr,
                        outcome:
                          item.outcome.type === 'exit'
                            ? {
                                type: 'exit' as const,
                                exitCode: item.outcome.exit_code,
                              }
                            : { type: 'timeout' as const },
                      }),
                    ),
                  } satisfies InferSchema<typeof shellOutputSchema>,
                });
              } else if (value.item.type === 'reasoning') {
                const itemId = resolveOutputItemId({
                  itemId: value.item.id,
                  outputIndex: value.output_index,
                });
                const activeReasoningPart = activeReasoning[itemId];

                if (activeReasoningPart != null) {
                  // get all active or can-conclude summary parts' ids
                  // to conclude ongoing reasoning parts:
                  const summaryPartIndices = Object.entries(
                    activeReasoningPart.summaryParts,
                  )
                    .filter(
                      ([_, status]) =>
                        status === 'active' || status === 'can-conclude',
                    )
                    .map(([summaryIndex]) => summaryIndex);

                  for (const summaryIndex of summaryPartIndices) {
                    controller.enqueue({
                      type: 'reasoning-end',
                      id: `${itemId}:${summaryIndex}`,
                      providerMetadata: {
                        [providerOptionsName]: {
                          itemId,
                          reasoningEncryptedContent:
                            value.item.encrypted_content ?? null,
                        } satisfies ResponsesReasoningProviderMetadata,
                      },
                    });
                  }

                  delete activeReasoning[itemId];
                }
                activeOutputItemIds[value.output_index] = undefined;
              } else if (value.item.type === 'compaction') {
                controller.enqueue({
                  type: 'custom',
                  kind: 'openai.compaction',
                  providerMetadata: {
                    [providerOptionsName]: {
                      type: 'compaction',
                      itemId: value.item.id,
                      encryptedContent: value.item.encrypted_content,
                    } satisfies ResponsesCompactionProviderMetadata,
                  },
                });
              }
            } else if (isResponseFunctionCallArgumentsDeltaChunk(value)) {
              const toolCall = ongoingToolCalls[value.output_index];

              if (toolCall != null) {
                controller.enqueue({
                  type: 'tool-input-delta',
                  id: toolCall.toolCallId,
                  delta: value.delta,
                });
              }
            } else if (isResponseCustomToolCallInputDeltaChunk(value)) {
              const toolCall = ongoingToolCalls[value.output_index];

              if (toolCall != null) {
                controller.enqueue({
                  type: 'tool-input-delta',
                  id: toolCall.toolCallId,
                  delta: value.delta,
                });
              }
            } else if (isResponseApplyPatchCallOperationDiffDeltaChunk(value)) {
              const toolCall = ongoingToolCalls[value.output_index];

              if (toolCall?.applyPatch) {
                controller.enqueue({
                  type: 'tool-input-delta',
                  id: toolCall.toolCallId,
                  delta: escapeJSONDelta(value.delta),
                });

                toolCall.applyPatch.hasDiff = true;
              }
            } else if (isResponseApplyPatchCallOperationDiffDoneChunk(value)) {
              const toolCall = ongoingToolCalls[value.output_index];

              if (toolCall?.applyPatch && !toolCall.applyPatch.endEmitted) {
                if (!toolCall.applyPatch.hasDiff) {
                  controller.enqueue({
                    type: 'tool-input-delta',
                    id: toolCall.toolCallId,
                    delta: escapeJSONDelta(value.diff),
                  });

                  toolCall.applyPatch.hasDiff = true;
                }

                controller.enqueue({
                  type: 'tool-input-delta',
                  id: toolCall.toolCallId,
                  delta: '"}}',
                });

                controller.enqueue({
                  type: 'tool-input-end',
                  id: toolCall.toolCallId,
                });

                toolCall.applyPatch.endEmitted = true;
              }
            } else if (isResponseImageGenerationCallPartialImageChunk(value)) {
              controller.enqueue({
                type: 'tool-result',
                toolCallId: value.item_id,
                toolName: toolNameMapping.toCustomToolName('image_generation'),
                result: {
                  result: value.partial_image_b64,
                } satisfies InferSchema<typeof imageGenerationOutputSchema>,
                preliminary: true,
              });
            } else if (isResponseCodeInterpreterCallCodeDeltaChunk(value)) {
              const toolCall = ongoingToolCalls[value.output_index];

              if (toolCall != null) {
                controller.enqueue({
                  type: 'tool-input-delta',
                  id: toolCall.toolCallId,
                  delta: escapeJSONDelta(value.delta),
                });
              }
            } else if (isResponseCodeInterpreterCallCodeDoneChunk(value)) {
              const toolCall = ongoingToolCalls[value.output_index];

              if (toolCall != null) {
                controller.enqueue({
                  type: 'tool-input-delta',
                  id: toolCall.toolCallId,
                  delta: '"}',
                });

                controller.enqueue({
                  type: 'tool-input-end',
                  id: toolCall.toolCallId,
                });

                // immediately send the tool call after the input end:
                controller.enqueue({
                  type: 'tool-call',
                  toolCallId: toolCall.toolCallId,
                  toolName:
                    toolNameMapping.toCustomToolName('code_interpreter'),
                  input: JSON.stringify({
                    code: value.code,
                    containerId: toolCall.codeInterpreter!.containerId,
                  } satisfies InferSchema<typeof codeInterpreterInputSchema>),
                  providerExecuted: true,
                });
              }
            } else if (isResponseCreatedChunk(value)) {
              responseId = value.response.id;
              controller.enqueue({
                type: 'response-metadata',
                id: value.response.id,
                timestamp: new Date(value.response.created_at * 1000),
                modelId: value.response.model,
              });
            } else if (isTextDeltaChunk(value)) {
              const itemId = resolveOutputItemId({
                itemId: value.item_id,
                outputIndex: value.output_index,
              });
              controller.enqueue({
                type: 'text-delta',
                id: itemId,
                delta: value.delta,
              });

              if (
                options.providerOptions?.[providerOptionsName]?.logprobs &&
                value.logprobs
              ) {
                logprobs.push(value.logprobs);
              }
            } else if (value.type === 'response.reasoning_summary_part.added') {
              const itemId = resolveOutputItemId({
                itemId: value.item_id,
                outputIndex: value.output_index,
              });
              // the first reasoning start is pushed in isResponseOutputItemAddedReasoningChunk
              if (value.summary_index > 0) {
                const activeReasoningPart = activeReasoning[itemId];

                if (activeReasoningPart != null) {
                  activeReasoningPart.summaryParts[value.summary_index] =
                    'active';

                  // since there is a new active summary part, we can conclude all can-conclude summary parts
                  for (const summaryIndex of Object.keys(
                    activeReasoningPart.summaryParts,
                  )) {
                    if (
                      activeReasoningPart.summaryParts[summaryIndex] ===
                      'can-conclude'
                    ) {
                      controller.enqueue({
                        type: 'reasoning-end',
                        id: `${itemId}:${summaryIndex}`,
                        providerMetadata: {
                          [providerOptionsName]: {
                            itemId,
                          } satisfies ResponsesReasoningProviderMetadata,
                        },
                      });
                      activeReasoningPart.summaryParts[summaryIndex] =
                        'concluded';
                    }
                  }

                  controller.enqueue({
                    type: 'reasoning-start',
                    id: `${itemId}:${value.summary_index}`,
                    providerMetadata: {
                      [providerOptionsName]: {
                        itemId,
                        reasoningEncryptedContent:
                          activeReasoningPart.encryptedContent ?? null,
                      } satisfies ResponsesReasoningProviderMetadata,
                    },
                  });
                }
              }
            } else if (value.type === 'response.reasoning_summary_text.delta') {
              const itemId = resolveOutputItemId({
                itemId: value.item_id,
                outputIndex: value.output_index,
              });
              controller.enqueue({
                type: 'reasoning-delta',
                id: `${itemId}:${value.summary_index}`,
                delta: value.delta,
                providerMetadata: {
                  [providerOptionsName]: {
                    itemId,
                  } satisfies ResponsesReasoningProviderMetadata,
                },
              });
            } else if (value.type === 'response.reasoning_summary_part.done') {
              const itemId = resolveOutputItemId({
                itemId: value.item_id,
                outputIndex: value.output_index,
              });
              const activeReasoningPart = activeReasoning[itemId];

              if (activeReasoningPart != null) {
                // when OpenAI stores the message data, we can immediately conclude the reasoning part
                // since we do not need to send the encrypted content.
                if (store) {
                  controller.enqueue({
                    type: 'reasoning-end',
                    id: `${itemId}:${value.summary_index}`,
                    providerMetadata: {
                      [providerOptionsName]: {
                        itemId,
                      } satisfies ResponsesReasoningProviderMetadata,
                    },
                  });

                  // mark the summary part as concluded
                  activeReasoningPart.summaryParts[value.summary_index] =
                    'concluded';
                } else {
                  // mark the summary part as can-conclude only
                  // because we need to have a final summary part with the encrypted content
                  activeReasoningPart.summaryParts[value.summary_index] =
                    'can-conclude';
                }
              }
            } else if (isResponseFinishedChunk(value)) {
              finishReason = {
                unified: mapOpenAIResponseFinishReason({
                  finishReason: value.response.incomplete_details?.reason,
                  hasFunctionCall,
                }),
                raw: value.response.incomplete_details?.reason ?? undefined,
              };
              usage = value.response.usage;
              if (typeof value.response.service_tier === 'string') {
                serviceTier = value.response.service_tier;
              }
              if (value.response.reasoning?.context != null) {
                reasoningContext = value.response.reasoning.context;
              }
            } else if (isResponseFailedChunk(value)) {
              const incompleteReason =
                value.response.incomplete_details?.reason;
              finishReason = {
                unified: incompleteReason
                  ? mapOpenAIResponseFinishReason({
                      finishReason: incompleteReason,
                      hasFunctionCall,
                    })
                  : 'error',
                raw: incompleteReason ?? 'error',
              };
              usage = value.response.usage ?? undefined;
              if (value.response.reasoning?.context != null) {
                reasoningContext = value.response.reasoning.context;
              }

              if (!encounteredStreamError && value.response.error != null) {
                encounteredStreamError = true;
                controller.enqueue({
                  type: 'error',
                  error: {
                    type: 'response.failed',
                    sequence_number: value.sequence_number,
                    response: {
                      error: value.response.error,
                      incomplete_details: value.response.incomplete_details,
                      service_tier: value.response.service_tier,
                    },
                  },
                });
              }
            } else if (isResponseAnnotationAddedChunk(value)) {
              ongoingAnnotations.push(value.annotation);
              if (value.annotation.type === 'url_citation') {
                controller.enqueue({
                  type: 'source',
                  sourceType: 'url',
                  id: self.config.generateId?.() ?? generateId(),
                  url: value.annotation.url,
                  title: value.annotation.title,
                });
              } else if (value.annotation.type === 'file_citation') {
                controller.enqueue({
                  type: 'source',
                  sourceType: 'document',
                  id: self.config.generateId?.() ?? generateId(),
                  mediaType: 'text/plain',
                  title: value.annotation.filename,
                  filename: value.annotation.filename,
                  providerMetadata: {
                    [providerOptionsName]: {
                      type: value.annotation.type,
                      fileId: value.annotation.file_id,
                      index: value.annotation.index,
                    } satisfies Extract<
                      ResponsesSourceDocumentProviderMetadata,
                      { type: 'file_citation' }
                    >,
                  },
                });
              } else if (value.annotation.type === 'container_file_citation') {
                controller.enqueue({
                  type: 'source',
                  sourceType: 'document',
                  id: self.config.generateId?.() ?? generateId(),
                  mediaType: 'text/plain',
                  title: value.annotation.filename,
                  filename: value.annotation.filename,
                  providerMetadata: {
                    [providerOptionsName]: {
                      type: value.annotation.type,
                      fileId: value.annotation.file_id,
                      containerId: value.annotation.container_id,
                    } satisfies Extract<
                      ResponsesSourceDocumentProviderMetadata,
                      { type: 'container_file_citation' }
                    >,
                  },
                });
              } else if (value.annotation.type === 'file_path') {
                controller.enqueue({
                  type: 'source',
                  sourceType: 'document',
                  id: self.config.generateId?.() ?? generateId(),
                  mediaType: 'application/octet-stream',
                  title: value.annotation.file_id,
                  filename: value.annotation.file_id,
                  providerMetadata: {
                    [providerOptionsName]: {
                      type: value.annotation.type,
                      fileId: value.annotation.file_id,
                      index: value.annotation.index,
                    } satisfies Extract<
                      ResponsesSourceDocumentProviderMetadata,
                      { type: 'file_path' }
                    >,
                  },
                });
              }
            } else if (isErrorChunk(value)) {
              encounteredStreamError = true;
              finishReason = { unified: 'error', raw: 'error' };
              controller.enqueue({ type: 'error', error: value });
            }
          },

          flush(controller) {
            const providerMetadata: SharedV4ProviderMetadata = {
              [providerOptionsName]: {
                responseId: responseId,
                ...(logprobs.length > 0 ? { logprobs } : {}),
                ...(serviceTier !== undefined ? { serviceTier } : {}),
                ...(reasoningContext !== undefined ? { reasoningContext } : {}),
              } satisfies ResponsesProviderMetadata,
            };

            controller.enqueue({
              type: 'finish',
              finishReason,
              usage: convertOpenAIResponsesUsage(usage),
              providerMetadata,
            });
          },
        }),
      ),
      request: { body },
      response: { headers: responseHeaders },
    };

    return result;
  }
}

function createBufferedLanguageModelStream(
  result: LanguageModelV4GenerateResult,
): ReadableStream<LanguageModelV4StreamPart> {
  return new ReadableStream<LanguageModelV4StreamPart>({
    start(controller) {
      controller.enqueue({
        type: 'stream-start',
        warnings: result.warnings,
      });

      if (result.response != null) {
        controller.enqueue({
          type: 'response-metadata',
          id: result.response.id,
          timestamp: result.response.timestamp,
          modelId: result.response.modelId,
        });
      }

      for (const content of result.content) {
        switch (content.type) {
          case 'text': {
            const id = generateId();
            controller.enqueue({
              type: 'text-start',
              id,
              providerMetadata: content.providerMetadata,
            });
            controller.enqueue({
              type: 'text-delta',
              id,
              delta: content.text,
              providerMetadata: content.providerMetadata,
            });
            controller.enqueue({
              type: 'text-end',
              id,
              providerMetadata: content.providerMetadata,
            });
            break;
          }
          case 'reasoning': {
            const id = generateId();
            controller.enqueue({
              type: 'reasoning-start',
              id,
              providerMetadata: content.providerMetadata,
            });
            controller.enqueue({
              type: 'reasoning-delta',
              id,
              delta: content.text,
              providerMetadata: content.providerMetadata,
            });
            controller.enqueue({
              type: 'reasoning-end',
              id,
              providerMetadata: content.providerMetadata,
            });
            break;
          }
          case 'tool-call': {
            controller.enqueue({
              type: 'tool-input-start',
              id: content.toolCallId,
              toolName: content.toolName,
              providerExecuted: content.providerExecuted,
              dynamic: content.dynamic,
              providerMetadata: content.providerMetadata,
            });
            controller.enqueue({
              type: 'tool-input-end',
              id: content.toolCallId,
              providerMetadata: content.providerMetadata,
            });
            controller.enqueue(content);
            break;
          }
          default:
            controller.enqueue(content);
        }
      }

      controller.enqueue({
        type: 'finish',
        finishReason: result.finishReason,
        usage: result.usage,
        providerMetadata: result.providerMetadata,
      });
      controller.close();
    },
  });
}

function isTextDeltaChunk(
  chunk: OpenAIResponsesChunk,
): chunk is OpenAIResponsesChunk & { type: 'response.output_text.delta' } {
  return chunk.type === 'response.output_text.delta';
}

function isOpenAIChatCompletionChunk(value: unknown): boolean {
  const chunk = asRecord(value);

  return (
    chunk != null &&
    Array.isArray(chunk.choices) &&
    typeof chunk.type !== 'string'
  );
}

function createOpenAIResponsesChatCompletionsMismatchError({
  value,
  cause,
  url,
  requestBodyValues,
  responseHeaders,
}: {
  value: unknown;
  cause: unknown;
  url: string;
  requestBodyValues: unknown;
  responseHeaders?: Record<string, string>;
}): APICallError {
  return new APICallError({
    message:
      'Received a Chat Completions stream while using the OpenAI Responses API. ' +
      "The default OpenAI provider model uses the Responses API. If your custom baseURL targets a Chat Completions-compatible endpoint, use openai.chat('model-id') or createOpenAI(...).chat('model-id') instead. " +
      'You can also use @ai-sdk/openai-compatible for OpenAI-compatible providers.',
    url,
    requestBodyValues,
    responseHeaders,
    responseBody: JSON.stringify(value),
    cause,
    data: value,
    isRetryable: false,
  });
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return typeof value === 'object' && value != null
    ? (value as Record<string, unknown>)
    : undefined;
}

function isResponseOutputItemDoneChunk(
  chunk: OpenAIResponsesChunk,
): chunk is OpenAIResponsesChunk & { type: 'response.output_item.done' } {
  return chunk.type === 'response.output_item.done';
}

function isResponseFinishedChunk(
  chunk: OpenAIResponsesChunk,
): chunk is OpenAIResponsesChunk & {
  type: 'response.completed' | 'response.incomplete';
} {
  return (
    chunk.type === 'response.completed' || chunk.type === 'response.incomplete'
  );
}

function isResponseFailedChunk(
  chunk: OpenAIResponsesChunk,
): chunk is OpenAIResponsesChunk & { type: 'response.failed' } {
  return chunk.type === 'response.failed';
}

function isResponseCreatedChunk(
  chunk: OpenAIResponsesChunk,
): chunk is OpenAIResponsesChunk & { type: 'response.created' } {
  return chunk.type === 'response.created';
}

function isResponseFunctionCallArgumentsDeltaChunk(
  chunk: OpenAIResponsesChunk,
): chunk is OpenAIResponsesChunk & {
  type: 'response.function_call_arguments.delta';
} {
  return chunk.type === 'response.function_call_arguments.delta';
}

function isResponseCustomToolCallInputDeltaChunk(
  chunk: OpenAIResponsesChunk,
): chunk is OpenAIResponsesChunk & {
  type: 'response.custom_tool_call_input.delta';
} {
  return chunk.type === 'response.custom_tool_call_input.delta';
}

function isResponseImageGenerationCallPartialImageChunk(
  chunk: OpenAIResponsesChunk,
): chunk is OpenAIResponsesChunk & {
  type: 'response.image_generation_call.partial_image';
} {
  return chunk.type === 'response.image_generation_call.partial_image';
}

function isResponseCodeInterpreterCallCodeDeltaChunk(
  chunk: OpenAIResponsesChunk,
): chunk is OpenAIResponsesChunk & {
  type: 'response.code_interpreter_call_code.delta';
} {
  return chunk.type === 'response.code_interpreter_call_code.delta';
}

function isResponseCodeInterpreterCallCodeDoneChunk(
  chunk: OpenAIResponsesChunk,
): chunk is OpenAIResponsesChunk & {
  type: 'response.code_interpreter_call_code.done';
} {
  return chunk.type === 'response.code_interpreter_call_code.done';
}

function isResponseApplyPatchCallOperationDiffDeltaChunk(
  chunk: OpenAIResponsesChunk,
): chunk is OpenAIResponsesApplyPatchOperationDiffDeltaChunk {
  return chunk.type === 'response.apply_patch_call_operation_diff.delta';
}

function isResponseApplyPatchCallOperationDiffDoneChunk(
  chunk: OpenAIResponsesChunk,
): chunk is OpenAIResponsesApplyPatchOperationDiffDoneChunk {
  return chunk.type === 'response.apply_patch_call_operation_diff.done';
}

function isResponseOutputItemAddedChunk(
  chunk: OpenAIResponsesChunk,
): chunk is OpenAIResponsesChunk & { type: 'response.output_item.added' } {
  return chunk.type === 'response.output_item.added';
}

function isResponseAnnotationAddedChunk(
  chunk: OpenAIResponsesChunk,
): chunk is OpenAIResponsesChunk & {
  type: 'response.output_text.annotation.added';
} {
  return chunk.type === 'response.output_text.annotation.added';
}

function isErrorChunk(
  chunk: OpenAIResponsesChunk,
): chunk is OpenAIResponsesChunk & { type: 'error' } {
  return chunk.type === 'error';
}

function isResponseInProgressChunk(
  chunk: OpenAIResponsesChunk,
): chunk is OpenAIResponsesChunk & { type: 'response.in_progress' } {
  return chunk.type === 'response.in_progress';
}

function isResponseOutputChunk(chunk: OpenAIResponsesChunk): boolean {
  return !(
    chunk.type === 'response.created' ||
    chunk.type === 'response.in_progress' ||
    chunk.type === 'response.failed' ||
    chunk.type === 'error' ||
    chunk.type === 'unknown_chunk'
  );
}

function mapWebSearchOutput(
  action: OpenAIResponsesWebSearchAction | null | undefined,
): InferSchema<typeof webSearchOutputSchema> {
  if (action == null) {
    return {};
  }

  switch (action.type) {
    case 'search':
      return {
        action: {
          type: 'search',
          query: action.query ?? undefined,
          ...(action.queries != null && { queries: action.queries }),
        },
        // include sources when provided by the Responses API (behind include flag)
        ...(action.sources != null && { sources: action.sources }),
      };
    case 'open_page':
      return { action: { type: 'openPage', url: action.url } };
    case 'find_in_page':
      return {
        action: {
          type: 'findInPage',
          url: action.url,
          pattern: action.pattern,
        },
      };
  }
}

// The delta is embedded in a JSON string.
// To escape it, we use JSON.stringify and slice to remove the outer quotes.
function escapeJSONDelta(delta: string) {
  return JSON.stringify(delta).slice(1, -1);
}
