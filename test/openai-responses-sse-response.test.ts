import assert from 'node:assert/strict';
import { once } from 'node:events';
import { createServer, type ServerResponse } from 'node:http';
import { APICallError } from '@ai-sdk/provider';
import { createOpenAI } from '../src/openai-provider';
import { isOpenCodeRetryableError } from '../src/responses/openai-responses-retry';
import {
  isEventStreamResponse,
  parseSseEvents,
  reconstructResponseFromSse,
} from '../src/responses/openai-responses-tolerant-response';

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

function successfulMessage() {
  return {
    type: 'message',
    role: 'assistant',
    id: 'msg_success',
    content: [{ type: 'output_text', text: 'OK', annotations: [] }],
  };
}

function sseFrame(event: string, payload: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`;
}

function sendSse(response: ServerResponse, frames: string[]) {
  response.writeHead(200, { 'content-type': 'text/event-stream' });
  response.end(frames.join(''));
}

function sendJson(response: ServerResponse, statusCode: number, body: unknown) {
  response.writeHead(statusCode, { 'content-type': 'application/json' });
  response.end(JSON.stringify(body));
}

async function runScenario({
  steps,
}: {
  steps: Array<(response: ServerResponse, index: number) => void>;
}) {
  let stepCount = 0;
  const server = createServer(async (request, response) => {
    let rawBody = '';
    for await (const chunk of request) {
      rawBody += chunk;
    }
    const index = stepCount;
    stepCount += 1;
    const step = steps[index];
    if (step == null) {
      sendJson(response, 500, { error: { message: 'unexpected', type: 'server_error', code: 'server_error' } });
      return;
    }
    step(response, index);
  });

  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const address = server.address();
  assert(address != null && typeof address === 'object');

  const provider = createOpenAI({
    name: 'headroom-openai-branching',
    apiKey: 'test-key',
    baseURL: `http://127.0.0.1:${address.port}/v1`,
  });
  const model = provider.responses('gpt-5.6-luna');

  return {
    call: () =>
      model.doGenerate({
        prompt: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
      }),
    async close() {
      server.close();
      await once(server, 'close');
    },
  };
}

test('isEventStreamResponse detects content-type and body sniffing', () => {
  assert.equal(isEventStreamResponse('text/event-stream', '{}'), true);
  assert.equal(isEventStreamResponse('text/event-stream; charset=utf-8', ''), true);
  assert.equal(isEventStreamResponse('application/json', 'event: response.created\ndata: {}'), true);
  assert.equal(isEventStreamResponse('application/json', 'data: {"a":1}'), true);
  assert.equal(isEventStreamResponse('application/json', '{"output":[]}'), false);
});

test('parseSseEvents joins multi-line data and ignores comments', () => {
  const events = parseSseEvents(
    ': keep-alive\nevent: response.completed\ndata: {"a":\ndata: 1}\n\n',
  );
  assert.equal(events.length, 1);
  assert.equal(events[0]?.event, 'response.completed');
  assert.equal(events[0]?.data, '{"a":\n1}');
});

test('reconstructResponseFromSse picks the terminal completed response', () => {
  const completed = responsePayload([successfulMessage()]);
  const events = parseSseEvents(
    sseFrame('response.created', { type: 'response.created', response: { id: 'resp_x' } }) +
      sseFrame('response.completed', { type: 'response.completed', response: completed }),
  );
  const result = reconstructResponseFromSse(events);
  assert.equal(result.error, undefined);
  assert.deepEqual(result.response, completed);
});

test('reconstructResponseFromSse surfaces response.failed as error', () => {
  const events = parseSseEvents(
    sseFrame('response.failed', {
      type: 'response.failed',
      sequence_number: 1,
      response: {
        id: 'resp_x',
        error: {
          code: 'server_is_overloaded',
          message: 'stream failed type=service_unavailable_error code=server_is_overloaded',
        },
      },
    }),
  );
  const result = reconstructResponseFromSse(events);
  assert.equal(result.error?.code, 'server_is_overloaded');
  assert.match(result.error?.message ?? '', /server_is_overloaded/);
});

test('plain JSON response still works (fast path)', async () => {
  const scenario = await runScenario({
    steps: [response => sendJson(response, 200, responsePayload([successfulMessage()]))],
  });
  try {
    const result = await scenario.call();
    assert(result.content.some(part => part.type === 'text' && part.text === 'OK'));
  } finally {
    await scenario.close();
  }
});

test('SSE response.completed body is reconstructed on the non-stream path', async () => {
  const completed = responsePayload([successfulMessage()]);
  const scenario = await runScenario({
    steps: [
      response =>
        sendSse(response, [
          sseFrame('response.created', { type: 'response.created', response: { id: 'resp_x' } }),
          sseFrame('response.in_progress', { type: 'response.in_progress', response: { id: 'resp_x' } }),
          sseFrame('response.completed', { type: 'response.completed', response: completed }),
        ]),
    ],
  });
  try {
    const result = await scenario.call();
    assert(result.content.some(part => part.type === 'text' && part.text === 'OK'));
  } finally {
    await scenario.close();
  }
});

test('SSE response.failed surfaces the real upstream message as retryable', async () => {
  const scenario = await runScenario({
    steps: [
      response =>
        sendSse(response, [
          sseFrame('response.in_progress', { type: 'response.in_progress', response: { id: 'resp_x' } }),
          sseFrame('response.failed', {
            type: 'response.failed',
            sequence_number: 2,
            response: {
              id: 'resp_x',
              error: {
                code: 'server_is_overloaded',
                message: 'stream failed type=service_unavailable_error code=server_is_overloaded',
              },
            },
          }),
        ]),
    ],
  });
  try {
    await assert.rejects(scenario.call(), (error: unknown) => {
      assert(APICallError.isInstance(error));
      assert.match((error as APICallError).message, /server_is_overloaded/);
      assert.equal((error as APICallError).statusCode, 503);
      assert.equal(isOpenCodeRetryableError(error), true);
      return true;
    });
  } finally {
    await scenario.close();
  }
});

test('SSE error frame (service_unavailable_error) is retryable', async () => {
  const scenario = await runScenario({
    steps: [
      response =>
        sendSse(response, [
          ': keep-alive\n\n',
          sseFrame('error', {
            type: 'error',
            error: {
              type: 'service_unavailable_error',
              code: 'server_is_overloaded',
              message: 'stream failed type=service_unavailable_error code=server_is_overloaded',
            },
          }),
        ]),
    ],
  });
  try {
    await assert.rejects(scenario.call(), (error: unknown) => {
      assert(APICallError.isInstance(error));
      assert.match((error as APICallError).message, /service_unavailable_error|overloaded/);
      assert.equal((error as APICallError).statusCode, 503);
      assert.equal(isOpenCodeRetryableError(error), true);
      return true;
    });
  } finally {
    await scenario.close();
  }
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
