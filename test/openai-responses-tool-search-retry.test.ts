import assert from 'node:assert/strict';
import { once } from 'node:events';
import { createServer, type ServerResponse } from 'node:http';
import { APICallError, type LanguageModelV4Prompt } from '@ai-sdk/provider';
import { createOpenAI } from '../src/openai-provider';
import { convertToOpenAIResponsesInput } from '../src/responses/convert-to-openai-responses-input';
import {
  getOpenCodeRetryDelay,
  getOpenCodeResponseErrorStatusCode,
  isOpenCodeRetryableError,
  OPEN_CODE_RETRY_MAX_RETRIES,
} from '../src/responses/openai-responses-retry';

type RequestBody = {
  input?: Array<Record<string, unknown>>;
};

type Step = (response: ServerResponse, index: number) => void | Promise<void>;

const tests: Array<{
  name: string;
  run: () => void | Promise<void>;
}> = [];

function test(name: string, run: () => void | Promise<void>) {
  tests.push({ name, run });
}

function responsePayload(output: Array<Record<string, unknown>>) {
  return {
    id: `resp_${crypto.randomUUID()}`,
    created_at: Date.now() / 1_000,
    model: 'gpt-5.6-luna',
    output,
    service_tier: null,
    status: 'completed',
    usage: {
      input_tokens: 1,
      input_tokens_details: { cached_tokens: 0 },
      output_tokens: 1,
      output_tokens_details: { reasoning_tokens: 0 },
    },
  };
}

function sendJson(
  response: ServerResponse,
  statusCode: number,
  body: unknown,
  headers: Record<string, string> = {},
) {
  response.writeHead(statusCode, {
    'content-type': 'application/json',
    ...headers,
  });
  response.end(JSON.stringify(body));
}

function toolSearchCall() {
  return {
    type: 'tool_search_call',
    id: 'tsc_test',
    execution: 'client',
    call_id: 'call_tsc_test',
    status: 'completed',
    arguments: { query: 'test' },
  };
}

function successfulMessage() {
  return {
    type: 'message',
    role: 'assistant',
    id: 'msg_success',
    content: [
      {
        type: 'output_text',
        text: 'OK',
        annotations: [],
      },
    ],
  };
}

function errorBody({
  message,
  type,
  code,
}: {
  message: string;
  type: string;
  code: string;
}) {
  return { error: { message, type, code, param: null } };
}

async function runScenario({
  steps,
  fetch,
}: {
  steps: Step[];
  fetch?: typeof globalThis.fetch;
}) {
  const bodies: RequestBody[] = [];
  const server = createServer(async (request, response) => {
    let rawBody = '';
    for await (const chunk of request) {
      rawBody += chunk;
    }
    bodies.push(JSON.parse(rawBody) as RequestBody);

    const index = bodies.length - 1;
    const step = steps[index];
    if (step == null) {
      sendJson(
        response,
        500,
        errorBody({
          message: `Unexpected request ${index}`,
          type: 'server_error',
          code: 'server_error',
        }),
      );
      return;
    }
    await step(response, index);
  });

  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const address = server.address();
  assert(address != null && typeof address === 'object');

  const provider = createOpenAI({
    name: 'headroom-openai-branching',
    apiKey: 'test-key',
    baseURL: `http://127.0.0.1:${address.port}/v1`,
    fetch,
  });
  const model = provider.responses('gpt-5.6-luna');
  const call = () =>
    model.doGenerate({
      prompt: [
        {
          role: 'user',
          content: [{ type: 'text', text: 'test' }],
        },
      ],
    });

  return {
    bodies,
    call,
    async close() {
      server.close();
      await once(server, 'close');
    },
  };
}

function initialToolSearch(response: ServerResponse) {
  sendJson(response, 200, responsePayload([toolSearchCall()]));
}

function successfulFollowUp(response: ServerResponse) {
  sendJson(response, 200, responsePayload([successfulMessage()]));
}

function responseErrorPayload({
  message,
  type,
  code,
}: {
  message: string;
  type: string;
  code: string;
}) {
  return {
    ...responsePayload([]),
    error: { message, type, code, param: null },
  };
}

function assertSuccessfulResult(result: Awaited<ReturnType<ReturnType<typeof runScenario>['call']>>) {
  assert(
    result.content.some(
      part => part.type === 'text' && part.text === 'OK',
    ),
  );
}

function assertFollowUpInput(body: RequestBody) {
  const input = body.input ?? [];
  assert.equal(
    input.filter(item => item.type === 'tool_search_call').length,
    1,
  );
  assert.equal(
    input.filter(item => item.type === 'tool_search_output').length,
    1,
  );
}

test('TSC follow-up succeeds without retry', async () => {
  const scenario = await runScenario({
    steps: [initialToolSearch, successfulFollowUp],
  });
  try {
    assertSuccessfulResult(await scenario.call());
    assert.equal(scenario.bodies.length, 2);
    assertFollowUpInput(scenario.bodies[1]);
  } finally {
    await scenario.close();
  }
});

test('TSC follow-up retries a response-header timeout and succeeds', async () => {
  let fetchCount = 0;
  const timeoutFetch: typeof globalThis.fetch = async (input, init) => {
    fetchCount += 1;
    const request = globalThis.fetch(input, init);
    if (fetchCount !== 2) {
      return request;
    }

    return Promise.race([
      request,
      new Promise<Response>((_, reject) => {
        setTimeout(() => {
          const error = new Error('Response headers timed out');
          error.name = 'ProviderHeaderTimeoutError';
          reject(error);
        }, 20);
      }),
    ]);
  };
  const scenario = await runScenario({
    fetch: timeoutFetch,
    steps: [
      initialToolSearch,
      async response => {
        await new Promise(resolve => setTimeout(resolve, 100));
        successfulFollowUp(response);
      },
      successfulFollowUp,
    ],
  });
  try {
    assertSuccessfulResult(await scenario.call());
    assert.equal(scenario.bodies.length, 3);
    assertFollowUpInput(scenario.bodies[1]);
    assertFollowUpInput(scenario.bodies[2]);
  } finally {
    await scenario.close();
  }
});

test('TSC follow-up retries HTTP 503 and honors Retry-After', async () => {
  const scenario = await runScenario({
    steps: [
      initialToolSearch,
      response =>
        sendJson(
          response,
          503,
          errorBody({
            message: 'Service unavailable',
            type: 'server_error',
            code: 'server_error',
          }),
          { 'retry-after-ms': '0' },
        ),
      successfulFollowUp,
    ],
  });
  try {
    assertSuccessfulResult(await scenario.call());
    assert.equal(scenario.bodies.length, 3);
  } finally {
    await scenario.close();
  }
});

test('TSC follow-up retries transient response.error in a successful HTTP response', async () => {
  const scenario = await runScenario({
    steps: [
      initialToolSearch,
      response =>
        sendJson(
          response,
          200,
          responseErrorPayload({
            message: 'Our servers are currently overloaded. Please try again later.',
            type: 'server_error',
            code: 'server_is_overloaded',
          }),
          { 'retry-after-ms': '0' },
        ),
      successfulFollowUp,
    ],
  });
  try {
    assertSuccessfulResult(await scenario.call());
    assert.equal(scenario.bodies.length, 3);
    assertFollowUpInput(scenario.bodies[1]);
    assertFollowUpInput(scenario.bodies[2]);
  } finally {
    await scenario.close();
  }
});

test('TSC follow-up returns fatal response.error without retry', async () => {
  const scenario = await runScenario({
    steps: [
      initialToolSearch,
      response =>
        sendJson(
          response,
          200,
          responseErrorPayload({
            message: 'You exceeded your current quota',
            type: 'insufficient_quota',
            code: 'insufficient_quota',
          }),
        ),
    ],
  });
  try {
    await assert.rejects(scenario.call(), error => {
      assert(APICallError.isInstance(error));
      assert.equal(error.statusCode, 429);
      assert.equal(error.data?.error?.code, 'insufficient_quota');
      return true;
    });
    assert.equal(scenario.bodies.length, 2);
  } finally {
    await scenario.close();
  }
});

for (const code of ['rate_limit', 'too_many_requests']) {
  test(`TSC follow-up retries HTTP 429 ${code}`, async () => {
    const scenario = await runScenario({
      steps: [
        initialToolSearch,
        response =>
          sendJson(
            response,
            429,
            errorBody({
              message:
                code === 'rate_limit'
                  ? 'Rate limit exceeded'
                  : 'Too many requests',
              type: code,
              code,
            }),
            { 'retry-after-ms': '0' },
          ),
        successfulFollowUp,
      ],
    });
    try {
      assertSuccessfulResult(await scenario.call());
      assert.equal(scenario.bodies.length, 3);
    } finally {
      await scenario.close();
    }
  });
}

test('OpenCode Retry-After delay formats are preserved', () => {
  const retryAfterMs = new APICallError({
    message: 'busy',
    url: 'https://example.test',
    requestBodyValues: {},
    statusCode: 503,
    responseHeaders: { 'retry-after-ms': '1234' },
  });
  assert.equal(getOpenCodeRetryDelay(1, retryAfterMs, 0), 1234);

  const retryAfterSeconds = new APICallError({
    message: 'busy',
    url: 'https://example.test',
    requestBodyValues: {},
    statusCode: 429,
    responseHeaders: { 'retry-after': '2.5' },
  });
  assert.equal(getOpenCodeRetryDelay(1, retryAfterSeconds, 0), 2500);
});

test('retry exhaustion throws the final TSC error without a provider result', async () => {
  const failures = Array.from(
    { length: OPEN_CODE_RETRY_MAX_RETRIES + 1 },
    (): Step => response =>
      sendJson(
        response,
        503,
        errorBody({
          message: 'Service unavailable',
          type: 'server_error',
          code: 'server_error',
        }),
        { 'retry-after-ms': '0' },
      ),
  );
  const scenario = await runScenario({
    steps: [initialToolSearch, ...failures],
  });
  try {
    await assert.rejects(scenario.call(), error => {
      assert(APICallError.isInstance(error));
      assert.equal(error.statusCode, 503);
      return true;
    });
    assert.equal(
      scenario.bodies.length,
      1 + OPEN_CODE_RETRY_MAX_RETRIES + 1,
    );
    for (const body of scenario.bodies.slice(1)) {
      assertFollowUpInput(body);
    }
  } finally {
    await scenario.close();
  }
});

for (const failure of [
  {
    name: 'HTTP 401 authentication error',
    status: 401,
    message: 'Invalid API key',
    type: 'authentication_error',
    code: 'invalid_api_key',
  },
  {
    name: 'insufficient quota',
    status: 429,
    message: 'You exceeded your current quota',
    type: 'insufficient_quota',
    code: 'insufficient_quota',
  },
  {
    name: 'invalid request',
    status: 400,
    message: 'Invalid request parameter',
    type: 'invalid_request_error',
    code: 'invalid_request',
  },
]) {
  test(`${failure.name} is thrown without TSC retry`, async () => {
    const scenario = await runScenario({
      steps: [
        initialToolSearch,
        response =>
          sendJson(
            response,
            failure.status,
            errorBody({
              message: failure.message,
              type: failure.type,
              code: failure.code,
            }),
          ),
      ],
    });
    try {
      await assert.rejects(scenario.call(), error => {
        assert(APICallError.isInstance(error));
        assert.equal(error.statusCode, failure.status);
        return true;
      });
      assert.equal(scenario.bodies.length, 2);
    } finally {
      await scenario.close();
    }
  });
}

test('OpenCode retry of the original request re-executes TSC from clean state', async () => {
  const scenario = await runScenario({
    steps: [
      initialToolSearch,
      response =>
        sendJson(
          response,
          401,
          errorBody({
            message: 'Invalid API key',
            type: 'authentication_error',
            code: 'invalid_api_key',
          }),
        ),
      initialToolSearch,
      successfulFollowUp,
    ],
  });
  try {
    await assert.rejects(scenario.call(), error => {
      assert(APICallError.isInstance(error));
      return true;
    });
    assertSuccessfulResult(await scenario.call());
    assert.equal(scenario.bodies.length, 4);
    assert.deepEqual(scenario.bodies[0], scenario.bodies[2]);
    assertFollowUpInput(scenario.bodies[1]);
    assertFollowUpInput(scenario.bodies[3]);
  } finally {
    await scenario.close();
  }
});

test('OpenCode retry re-executes TSC after hidden 503 retries are exhausted', async () => {
  const failures = Array.from(
    { length: OPEN_CODE_RETRY_MAX_RETRIES + 1 },
    (): Step => response =>
      sendJson(
        response,
        503,
        errorBody({
          message: 'Service unavailable',
          type: 'server_error',
          code: 'server_error',
        }),
        { 'retry-after-ms': '0' },
      ),
  );
  const scenario = await runScenario({
    steps: [
      initialToolSearch,
      ...failures,
      initialToolSearch,
      successfulFollowUp,
    ],
  });
  try {
    await assert.rejects(scenario.call(), error => {
      assert(APICallError.isInstance(error));
      assert.equal(error.statusCode, 503);
      return true;
    });
    assertSuccessfulResult(await scenario.call());
    assert.equal(scenario.bodies.length, 9);
    assert.deepEqual(scenario.bodies[0], scenario.bodies[7]);
    for (const body of scenario.bodies.slice(1, 7)) {
      assertFollowUpInput(body);
    }
    assertFollowUpInput(scenario.bodies[8]);
  } finally {
    await scenario.close();
  }
});

test('retry classification matches OpenCode fatal and transient boundaries', () => {
  assert.equal(
    getOpenCodeResponseErrorStatusCode({
      type: 'server_error',
      code: 'server_is_overloaded',
    }),
    503,
  );
  assert.equal(
    getOpenCodeResponseErrorStatusCode({
      type: 'insufficient_quota',
      code: 'insufficient_quota',
    }),
    429,
  );
  assert.equal(
    isOpenCodeRetryableError(
      new APICallError({
        message: 'Rate limit exceeded',
        url: 'https://example.test',
        requestBodyValues: {},
        statusCode: 429,
      }),
    ),
    true,
  );
  assert.equal(
    isOpenCodeRetryableError(
      new APICallError({
        message: 'You exceeded your current quota',
        url: 'https://example.test',
        requestBodyValues: {},
        statusCode: 429,
        data: {
          error: {
            code: 'insufficient_quota',
            type: 'insufficient_quota',
          },
        },
      }),
    ),
    false,
  );
  assert.equal(
    isOpenCodeRetryableError(
      new APICallError({
        message: 'Invalid request',
        url: 'https://example.test',
        requestBodyValues: {},
        statusCode: 400,
        isRetryable: false,
      }),
    ),
    false,
  );
});

test('converts legacy raw file parts without serializing null content', async () => {
  const prompt = [
    {
      role: 'user' as const,
      content: [
        { type: 'text' as const, text: 'observer images' },
        {
          type: 'file' as const,
          mediaType: 'image/png',
          data: 'data:image/png;base64,AQID',
        },
        {
          type: 'file' as const,
          mediaType: 'image/png',
          data: new Uint8Array([4, 5, 6]),
        },
        {
          type: 'file' as const,
          mediaType: 'image/png',
          data: new URL('https://example.test/observer.png'),
        },
      ],
    },
  ] as unknown as LanguageModelV4Prompt;

  const { input } = await convertToOpenAIResponsesInput({
    prompt,
    toolNameMapping: {
      toProviderToolName: name => name,
      toCustomToolName: name => name,
    },
    systemMessageMode: 'system',
    providerOptionsName: 'openai.responses',
    store: true,
  });

  const content = (input[0] as { content: Array<Record<string, unknown>> })
    .content;
  assert.deepEqual(
    content.map(part => part.type),
    ['input_text', 'input_image', 'input_image', 'input_image'],
  );
  assert.equal(content[1]?.image_url, 'data:image/png;base64,AQID');
  assert.equal(content[2]?.image_url, 'data:image/png;base64,BAUG');
  assert.equal(content[3]?.image_url, 'https://example.test/observer.png');
  assert.ok(content.every(part => part != null && typeof part === 'object'));
});

let failures = 0;
for (const item of tests) {
  try {
    await item.run();
    console.log(`ok - ${item.name}`);
  } catch (error) {
    failures += 1;
    console.error(`not ok - ${item.name}`);
    console.error(error);
  }
}

if (failures > 0) {
  process.exitCode = 1;
}
