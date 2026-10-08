import { APICallError } from '@ai-sdk/provider';
import {
  extractResponseHeaders,
  safeParseJSON,
  type FlexibleSchema,
  type ResponseHandler,
} from '@ai-sdk/provider-utils';

// Compatibility endpoints are not required to honour `stream: false`. Some
// relays (and overloaded gateways in front of them) answer a non-streaming
// Responses request with an SSE body — `event: response.in_progress\ndata:
// {...}` — or fail the stream with a `response.failed` / `error` frame instead
// of an HTTP error status. `createJsonResponseHandler` parses the whole body as
// a single JSON object, so an SSE body surfaces as the opaque
// `Invalid JSON response` error and the real upstream message (for example
// `server_is_overloaded`) is lost.
//
// The handler below keeps the JSON fast-path untouched and only adds an SSE
// fallback: it reconstructs the final response from the terminal
// `response.completed` event, or throws a retryable `APICallError` carrying the
// upstream error frame.

type SseEvent = {
  event?: string;
  data: string;
};

type ReconstructedError = {
  message: string;
  code?: string | number;
  type?: string;
};

export function isEventStreamResponse(
  contentType: string | null | undefined,
  bodyText: string,
): boolean {
  if (contentType != null && /text\/event-stream/i.test(contentType)) {
    return true;
  }

  const head = bodyText.replace(/^\uFEFF/, '').trimStart();
  if (head.startsWith('event:') || head.startsWith('data:')) {
    return true;
  }

  return /(^|\n)data:/.test(bodyText.slice(0, 4096));
}

export function parseSseEvents(bodyText: string): SseEvent[] {
  const normalized = bodyText.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  const events: SseEvent[] = [];

  for (const block of normalized.split('\n\n')) {
    let eventName: string | undefined;
    const dataLines: string[] = [];

    for (const line of block.split('\n')) {
      if (line.startsWith(':')) {
        continue;
      }
      if (line.startsWith('event:')) {
        eventName = line.slice('event:'.length).trim();
        continue;
      }
      if (line.startsWith('data:')) {
        dataLines.push(line.slice('data:'.length).replace(/^ /, ''));
      }
    }

    if (dataLines.length > 0) {
      events.push({ event: eventName, data: dataLines.join('\n') });
    }
  }

  return events;
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return typeof value === 'object' && value != null
    ? (value as Record<string, unknown>)
    : undefined;
}

function getString(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

function getStringOrNumber(value: unknown): string | number | undefined {
  return typeof value === 'string' || typeof value === 'number'
    ? value
    : undefined;
}

function extractError(frame: Record<string, unknown>): ReconstructedError | undefined {
  if (frame.type === 'response.failed') {
    const response = asRecord(frame.response);
    const responseError = asRecord(response?.error);
    const message =
      getString(responseError?.message) ?? getString(frame.message);
    if (message == null) {
      return undefined;
    }
    return {
      message,
      code: getStringOrNumber(responseError?.code),
      type: 'response.failed',
    };
  }

  const nested = asRecord(frame.error);
  const error = nested ?? frame;
  const message = getString(error.message);

  if (message == null) {
    return undefined;
  }

  const looksLikeError =
    nested != null ||
    typeof frame.type === 'string' ||
    'code' in error ||
    'param' in error;

  if (!looksLikeError) {
    return undefined;
  }

  return {
    message,
    code: getStringOrNumber(error.code),
    type: getString(error.type) ?? getString(frame.type),
  };
}

function statusFromError(error: ReconstructedError): number {
  if (typeof error.code === 'number' && error.code >= 400 && error.code <= 599) {
    return error.code;
  }
  if (typeof error.code === 'string' && /^\d{3}$/.test(error.code)) {
    const numeric = Number(error.code);
    if (numeric >= 400 && numeric <= 599) {
      return numeric;
    }
  }

  const discriminator = `${error.code ?? ''} ${error.type ?? ''} ${error.message}`.toLowerCase();

  if (/rate[_ -]?limit|too[_ -]?many|quota|resource[_ -]?exhausted/.test(discriminator)) {
    return 429;
  }
  if (/overload|service[_ -]?unavailable|server[_ -]?error|internal[_ -]?error|capacity/.test(discriminator)) {
    return 503;
  }
  if (/timeout|timed out/.test(discriminator)) {
    return 504;
  }
  if (/auth/.test(discriminator)) {
    return 401;
  }
  if (/permission|forbidden/.test(discriminator)) {
    return 403;
  }
  if (/not[_ -]?found/.test(discriminator)) {
    return 404;
  }
  if (/invalid|bad[_ -]?request|context[_ -]?length|too long/.test(discriminator)) {
    return 400;
  }

  return 500;
}

export function reconstructResponseFromSse(events: SseEvent[]): {
  response?: unknown;
  error?: ReconstructedError;
  sawEvents: boolean;
} {
  let completedResponse: unknown;
  let latestSnapshot: unknown;
  let sawEvents = false;

  for (const event of events) {
    const trimmed = event.data.trim();
    if (trimmed === '' || trimmed === '[DONE]') {
      continue;
    }

    let parsed: unknown;
    try {
      parsed = JSON.parse(trimmed);
    } catch {
      continue;
    }

    const frame = asRecord(parsed);
    if (frame == null) {
      continue;
    }
    sawEvents = true;

    const error = extractError(frame);
    if (error != null && frame.type !== 'response.completed') {
      return { error, sawEvents };
    }

    const snapshot = asRecord(frame.response);
    if (snapshot == null) {
      continue;
    }

    if (snapshot.output != null) {
      latestSnapshot = snapshot;
    }

    if (frame.type === 'response.completed' || frame.type === 'response.incomplete') {
      completedResponse = snapshot;
    }
  }

  return {
    response: completedResponse ?? latestSnapshot,
    sawEvents,
  };
}

export function createTolerantResponsesResponseHandler<T>(
  schema: FlexibleSchema<T>,
): ResponseHandler<T> {
  return async ({ response, url, requestBodyValues }) => {
    const bodyText = await response.text();
    const responseHeaders = extractResponseHeaders(response);

    const contentType = response.headers.get('content-type');
    if (isEventStreamResponse(contentType, bodyText)) {
      const events = parseSseEvents(bodyText);
      const reconstructed = reconstructResponseFromSse(events);

      if (reconstructed.error != null) {
        throw new APICallError({
          message: reconstructed.error.message,
          url,
          requestBodyValues,
          statusCode: statusFromError(reconstructed.error),
          responseHeaders,
          responseBody: bodyText,
          data: { error: reconstructed.error },
        });
      }

      if (reconstructed.response != null) {
        return validateJsonText({
          text: JSON.stringify(reconstructed.response),
          schema,
          response,
          responseHeaders,
          url,
          requestBodyValues,
          bodyText,
        });
      }

      // The body looked like SSE but contained no usable events. Fall through
      // to the JSON parser only when it still resembles JSON, otherwise report
      // a retryable truncated-stream error instead of a misleading
      // "Invalid JSON response".
      if (!bodyText.trimStart().startsWith('{')) {
        throw new APICallError({
          message:
            'OpenAI Responses returned an event stream without a completed response',
          url,
          requestBodyValues,
          statusCode: 503,
          responseHeaders,
          responseBody: bodyText,
          data: { events },
        });
      }
    }

    return validateJsonText({
      text: bodyText,
      schema,
      response,
      responseHeaders,
      url,
      requestBodyValues,
      bodyText,
    });
  };
}

async function validateJsonText<T>({
  text,
  schema,
  response,
  responseHeaders,
  url,
  requestBodyValues,
  bodyText,
}: {
  text: string;
  schema: FlexibleSchema<T>;
  response: Response;
  responseHeaders: Record<string, string>;
  url: string;
  requestBodyValues: unknown;
  bodyText: string;
}): Promise<{ value: T; rawValue?: unknown; responseHeaders: Record<string, string> }> {
  const parsed = await safeParseJSON({ text, schema });

  if (!parsed.success) {
    throw new APICallError({
      message: 'Invalid JSON response',
      cause: parsed.error,
      statusCode: response.status,
      responseHeaders,
      responseBody: bodyText,
      url,
      requestBodyValues,
    });
  }

  return {
    responseHeaders,
    value: parsed.value,
    rawValue: parsed.rawValue,
  };
}
