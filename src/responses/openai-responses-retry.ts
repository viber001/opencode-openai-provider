import { APICallError } from '@ai-sdk/provider';

// Mirrors OpenCode v1.18.25 packages/opencode/src/session/retry.ts.
// This fork cannot import OpenCode internals, so keep changes synchronized with
// https://github.com/anomalyco/opencode/blob/v1.18.25/packages/opencode/src/session/retry.ts
export const OPEN_CODE_RETRY_INITIAL_DELAY = 2_000;
export const OPEN_CODE_RETRY_BACKOFF_FACTOR = 2;
export const OPEN_CODE_RETRY_JITTER_FACTOR = 0.25;
export const OPEN_CODE_RETRY_MAX_DELAY_NO_HEADERS = 30_000;
export const OPEN_CODE_RETRY_MAX_DELAY = 2_147_483_647;
export const OPEN_CODE_RETRY_MAX_RETRIES = 5;

export function getOpenCodeResponseErrorStatusCode({
  type,
  code,
}: {
  type: string;
  code: string;
}): number {
  const category = `${type} ${code}`.toLowerCase();

  if (
    /rate[_ -]?limit|too[_ -]?many[_ -]?requests|quota|resource[_ -]?exhausted/.test(
      category,
    )
  ) {
    return 429;
  }

  if (
    /overload|service[_ -]?unavailable|server[_ -]?error|internal[_ -]?error|capacity/.test(
      category,
    )
  ) {
    return 503;
  }

  return 400;
}

const RETRYABLE_MESSAGE_PATTERNS = [
  /429|500|502|503|504|524/i,
  /rate increased too quickly|rate limit|rate-limit|rate_limit|too many requests/i,
  /overloaded|service unavailable|service_unavailable|service-unavailable|internal error|internal_error|internal server error|server error|server_error|server-error|provider returned error|provider_returned_error|provider-returned-error/i,
  /terminated|fetch failed|failed to fetch|network[-_\s]error|upstream connect|connection error|connection refused|connection lost|socket connection was closed|socket hang up|reset before headers|getaddrinfo|enotfound|eai_again|econnrefused|econnreset|etimedout/i,
  /^timeout$|\b(?:request|response|connection|network|stream|read) (?:timeout|timed out|time out)\b/i,
  /try your request again|retry your request|resource exhausted|resource_exhausted/i,
  /\btry again (?:later|in\b)|\b(?:currently|temporarily) at capacity\b/i,
];

const CONTEXT_OVERFLOW_PATTERNS = [
  /prompt is too long/i,
  /request_too_large/i,
  /input is too long for requested model/i,
  /exceeds the context window/i,
  /exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))/i,
  /input token count.*exceeds the maximum/i,
  /tokens in request more than max tokens allowed/i,
  /maximum prompt length is \d+/i,
  /reduce the length of the messages/i,
  /maximum context length is \d+ tokens/i,
  /exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?/i,
  /input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)/i,
  /exceeds the limit of \d+/i,
  /exceeds the available context size/i,
  /greater than the context length/i,
  /context window exceeds limit/i,
  /exceeded model token limit/i,
  /context[_ ]length[_ ]exceeded/i,
  /request entity too large/i,
  /context length is only \d+ tokens/i,
  /input length.*exceeds.*context length/i,
  /prompt too long; exceeded (?:max )?context length/i,
  /too large for model with \d+ maximum context length/i,
  /prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?/i,
  /model_context_window_exceeded/i,
  /too many tokens/i,
  /token limit exceeded/i,
];

const CONTEXT_OVERFLOW_EXCLUSIONS = [
  /^(throttling error|service unavailable):/i,
  /rate limit/i,
  /too many requests/i,
];

const NON_RETRYABLE_PROVIDER_ERROR_CODES = new Set([
  'insufficient_quota',
  'usage_not_included',
  'invalid_prompt',
]);

type RetryErrorInfo = {
  message: string;
  statusCode?: number;
  isRetryable?: boolean;
  responseHeaders?: Record<string, string>;
  responseBody?: string;
};

export type OpenCodeRetryErrorEvent = {
  error: unknown;
  attempt: number;
  retryable: boolean;
  willRetry: boolean;
  delayMs?: number;
};

function isAbortError(error: unknown): boolean {
  return (
    error instanceof Error && error.name === 'AbortError'
  );
}

function getProviderErrorCode(error: APICallError): string | undefined {
  const data = error.data as
    | { error?: { code?: unknown; type?: unknown } }
    | undefined;
  for (const value of [data?.error?.code, data?.error?.type]) {
    if (typeof value === 'string') {
      return value.toLowerCase();
    }
  }

  if (error.responseBody != null) {
    try {
      const parsed = JSON.parse(error.responseBody) as {
        error?: { code?: unknown; type?: unknown };
      };
      for (const value of [parsed.error?.code, parsed.error?.type]) {
        if (typeof value === 'string') {
          return value.toLowerCase();
        }
      }
    } catch {
      return undefined;
    }
  }

  return undefined;
}

function matchesRetryableMessage(value: unknown): boolean {
  return (
    typeof value === 'string' &&
    RETRYABLE_MESSAGE_PATTERNS.some(pattern => pattern.test(value))
  );
}

function isContextOverflow(message: string): boolean {
  return (
    !CONTEXT_OVERFLOW_EXCLUSIONS.some(pattern => pattern.test(message)) &&
    (CONTEXT_OVERFLOW_PATTERNS.some(pattern => pattern.test(message)) ||
      /^4(00|13)\s*(status code)?\s*\(no body\)/i.test(message))
  );
}

function getErrorInfo(error: unknown): RetryErrorInfo | undefined {
  if (APICallError.isInstance(error)) {
    return {
      message: error.message,
      statusCode: error.statusCode,
      isRetryable: error.isRetryable,
      responseHeaders: error.responseHeaders,
      responseBody: error.responseBody,
    };
  }

  if (!(error instanceof Error)) {
    return undefined;
  }

  return { message: error.message };
}

export function isOpenCodeRetryableError(error: unknown): boolean {
  if (isAbortError(error)) {
    return false;
  }

  if (
    error instanceof Error &&
    (error.name === 'ProviderHeaderTimeoutError' ||
      error.name === 'ResponseStreamError')
  ) {
    return true;
  }

  const info = getErrorInfo(error);
  if (info == null) {
    return false;
  }

  if (isContextOverflow(info.message)) {
    return false;
  }

  if (APICallError.isInstance(error)) {
    const providerErrorCode = getProviderErrorCode(error);
    if (
      providerErrorCode != null &&
      NON_RETRYABLE_PROVIDER_ERROR_CODES.has(providerErrorCode)
    ) {
      return false;
    }

    return Boolean(
      info.isRetryable ||
        (info.statusCode != null && info.statusCode >= 500) ||
        matchesRetryableMessage(info.message) ||
        matchesRetryableMessage(info.responseBody),
    );
  }

  const code = (error as Error & { code?: unknown }).code;
  if (code === 'ECONNRESET' || code === 'ZlibError') {
    return true;
  }

  const lower = info.message.toLowerCase();
  return (
    lower.includes('too_many_requests') ||
    lower.includes('exhausted') ||
    lower.includes('unavailable') ||
    matchesRetryableMessage(info.message)
  );
}

function capRetryDelay(ms: number): number {
  return Math.min(ms, OPEN_CODE_RETRY_MAX_DELAY);
}

function exponentialRetryDelay(attempt: number, random: number): number {
  const base =
    OPEN_CODE_RETRY_INITIAL_DELAY *
    Math.pow(OPEN_CODE_RETRY_BACKOFF_FACTOR, attempt - 1);
  return Math.ceil(base + base * OPEN_CODE_RETRY_JITTER_FACTOR * random);
}

export function getOpenCodeRetryDelay(
  attempt: number,
  error?: unknown,
  random = Math.random(),
): number {
  const headers = getErrorInfo(error)?.responseHeaders;
  if (headers != null) {
    const retryAfterMs = headers['retry-after-ms'];
    if (retryAfterMs != null) {
      const parsedMs = Number.parseFloat(retryAfterMs);
      if (!Number.isNaN(parsedMs)) {
        return capRetryDelay(parsedMs);
      }
    }

    const retryAfter = headers['retry-after'];
    if (retryAfter != null) {
      const parsedSeconds = Number.parseFloat(retryAfter);
      if (!Number.isNaN(parsedSeconds)) {
        return capRetryDelay(Math.ceil(parsedSeconds * 1_000));
      }

      const parsedDate = Date.parse(retryAfter) - Date.now();
      if (!Number.isNaN(parsedDate) && parsedDate > 0) {
        return capRetryDelay(Math.ceil(parsedDate));
      }
    }

    return capRetryDelay(exponentialRetryDelay(attempt, random));
  }

  return capRetryDelay(
    Math.min(
      exponentialRetryDelay(attempt, random),
      OPEN_CODE_RETRY_MAX_DELAY_NO_HEADERS,
    ),
  );
}

function abortReason(signal: AbortSignal): unknown {
  return signal.reason ?? new DOMException('Aborted', 'AbortError');
}

async function sleepWithAbort(
  delayMs: number,
  abortSignal?: AbortSignal,
): Promise<void> {
  if (abortSignal?.aborted) {
    throw abortReason(abortSignal);
  }

  await new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(finish, delayMs);

    function finish() {
      abortSignal?.removeEventListener('abort', abort);
      resolve();
    }

    function abort() {
      clearTimeout(timeout);
      reject(abortReason(abortSignal!));
    }

    abortSignal?.addEventListener('abort', abort, { once: true });
  });
}

export async function retryWithOpenCodePolicy<T>({
  execute,
  maxRetries = OPEN_CODE_RETRY_MAX_RETRIES,
  abortSignal,
  onError,
}: {
  execute: (attempt: number) => Promise<T>;
  maxRetries?: number;
  abortSignal?: AbortSignal;
  onError?: (event: OpenCodeRetryErrorEvent) => void | Promise<void>;
}): Promise<T> {
  let attempt = 0;

  while (true) {
    try {
      return await execute(attempt);
    } catch (error) {
      const retryable = isOpenCodeRetryableError(error);
      const willRetry = retryable && attempt < maxRetries;
      const delayMs = willRetry
        ? getOpenCodeRetryDelay(attempt + 1, error)
        : undefined;

      await onError?.({
        error,
        attempt,
        retryable,
        willRetry,
        delayMs,
      });

      if (!willRetry) {
        throw error;
      }

      await sleepWithAbort(delayMs!, abortSignal);
      attempt += 1;
    }
  }
}
