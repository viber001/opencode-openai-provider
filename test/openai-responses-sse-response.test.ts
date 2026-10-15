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
  const bodies: Array<Record<string, unknown>> = [];
  const server = createServer(async (request, response) => {
    let rawBody = '';
    for await (const chunk of request) {
      rawBody += chunk;
    }
    bodies.push(JSON.parse(rawBody) as Record<string, unknown>);
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

  const streamRequest = {
    prompt: [{ role: 'user', content: [{ type: 'text' as const, text: 'test' }] }],
  };

  return {
    bodies,
    call: () =>
      model.doGenerate({
        prompt: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
      }),
    stream: () => model.doStream(streamRequest),
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

test('reconstructResponseFromSse rebuilds output from streamed output_item frames when the terminal snapshot is empty', () => {
  const reasoning = {
    type: 'reasoning',
    id: 'rs_1',
    summary: [],
    encrypted_content: 'enc',
  };
  const message = {
    type: 'message',
    id: 'msg_1',
    role: 'assistant',
    status: 'completed',
    phase: 'commentary',
    content: [{ type: 'output_text', text: 'hi', annotations: [] }],
  };
  const call = {
    type: 'function_call',
    id: 'fc_1',
    call_id: 'call_1',
    name: 'bash',
    arguments: '{}',
  };
  const events = parseSseEvents(
    sseFrame('response.created', { type: 'response.created', response: { id: 'resp_x', output: [] } }) +
      sseFrame('response.output_item.added', {
        type: 'response.output_item.added',
        output_index: 0,
        item: { type: 'reasoning', id: 'rs_1' },
      }) +
      sseFrame('response.output_item.done', {
        type: 'response.output_item.done',
        output_index: 0,
        item: reasoning,
      }) +
      sseFrame('response.output_item.done', {
        type: 'response.output_item.done',
        output_index: 1,
        item: message,
      }) +
      sseFrame('response.output_item.done', {
        type: 'response.output_item.done',
        output_index: 2,
        item: call,
      }) +
      sseFrame('response.completed', {
        type: 'response.completed',
        response: { id: 'resp_x', status: 'completed', output: [] },
      }),
  );
  const result = reconstructResponseFromSse(events);
  assert.equal(result.error, undefined);
  assert.deepEqual((result.response as { output?: unknown[] })?.output, [
    reasoning,
    message,
    call,
  ]);
});

test('reconstructResponseFromSse prefers a populated terminal output over streamed items', () => {
  const terminalItem = successfulMessage();
  const streamed = { type: 'message', id: 'msg_other', role: 'assistant', content: [] };
  const completed = responsePayload([terminalItem]);
  const events = parseSseEvents(
    sseFrame('response.output_item.done', {
      type: 'response.output_item.done',
      output_index: 0,
      item: streamed,
    }) +
      sseFrame('response.completed', { type: 'response.completed', response: completed }),
  );
  const result = reconstructResponseFromSse(events);
  assert.deepEqual((result.response as { output?: unknown[] })?.output, [terminalItem]);
});

test('streamed output_item frames survive an empty terminal output (no silent empty retry)', async () => {
  const scenario = await runScenario({
    steps: [
      response =>
        sendSse(response, [
          sseFrame('response.created', {
            type: 'response.created',
            response: { id: 'resp_x', output: [] },
          }),
          sseFrame('response.output_item.done', {
            type: 'response.output_item.done',
            output_index: 0,
            item: successfulMessage(),
          }),
          sseFrame('response.completed', {
            type: 'response.completed',
            response: responsePayload([]),
          }),
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

test('completed response whose output_text omits annotations validates (Tencent DeepSeek)', async () => {
  const messageWithoutAnnotations = {
    type: 'message',
    id: 'msg_tc',
    role: 'assistant',
    status: 'completed',
    phase: 'final_answer',
    content: [{ type: 'output_text', text: '你好' }],
  };
  const scenario = await runScenario({
    steps: [
      response =>
        sendSse(response, [
          sseFrame('response.created', {
            type: 'response.created',
            response: { id: 'resp_tc', output: [] },
          }),
          sseFrame('response.output_item.done', {
            type: 'response.output_item.done',
            output_index: 0,
            item: messageWithoutAnnotations,
          }),
          sseFrame('response.completed', {
            type: 'response.completed',
            response: responsePayload([messageWithoutAnnotations]),
          }),
        ]),
    ],
  });
  try {
    const result = await scenario.call();
    assert(result.content.some(part => part.type === 'text' && part.text === '你好'));
  } finally {
    await scenario.close();
  }
});

test('reasoning_text deltas surface as reasoning stream parts (DeepSeek)', async () => {
  const scenario = await runScenario({
    steps: [
      response =>
        sendSse(response, [
          sseFrame('response.created', {
            type: 'response.created',
            response: { id: 'resp_x', output: [] },
          }),
          sseFrame('response.output_item.added', {
            type: 'response.output_item.added',
            output_index: 0,
            item: {
              type: 'reasoning',
              id: 'rs_1',
              status: 'in_progress',
              summary: [],
              encrypted_content: 'enc-0',
            },
          }),
          sseFrame('response.content_part.added', {
            type: 'response.content_part.added',
            output_index: 0,
            item_id: 'rs_1',
            content_index: 0,
            part: { type: 'reasoning_text' },
          }),
          sseFrame('response.reasoning_text.delta', {
            type: 'response.reasoning_text.delta',
            output_index: 0,
            item_id: 'rs_1',
            content_index: 0,
            delta: 'We',
          }),
          sseFrame('response.reasoning_text.delta', {
            type: 'response.reasoning_text.delta',
            output_index: 0,
            item_id: 'rs_1',
            content_index: 0,
            delta: ' need answer.',
          }),
          sseFrame('response.reasoning_text.done', {
            type: 'response.reasoning_text.done',
            output_index: 0,
            item_id: 'rs_1',
            content_index: 0,
            text: 'We need answer.',
          }),
          sseFrame('response.output_item.done', {
            type: 'response.output_item.done',
            output_index: 0,
            item: {
              type: 'reasoning',
              id: 'rs_1',
              status: 'completed',
              content: [{ type: 'reasoning_text', text: 'We need answer.' }],
              summary: [],
              encrypted_content: 'enc-0',
            },
          }),
          sseFrame('response.output_item.added', {
            type: 'response.output_item.added',
            output_index: 1,
            item: { type: 'message', id: 'msg_1' },
          }),
          sseFrame('response.output_text.delta', {
            type: 'response.output_text.delta',
            item_id: 'msg_1',
            output_index: 1,
            content_index: 0,
            delta: 'OK',
          }),
          sseFrame('response.output_item.done', {
            type: 'response.output_item.done',
            output_index: 1,
            item: successfulMessage(),
          }),
          sseFrame('response.completed', {
            type: 'response.completed',
            response: responsePayload([
              {
                type: 'reasoning',
                id: 'rs_1',
                status: 'completed',
                content: [{ type: 'reasoning_text', text: 'We need answer.' }],
                summary: [],
                encrypted_content: 'enc-0',
              },
              successfulMessage(),
            ]),
          }),
        ]),
    ],
  });
  try {
    const { stream } = await scenario.stream();
    const parts: Array<Record<string, unknown>> = [];
    const reader = stream.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value as Record<string, unknown>);
    }

    const reasoningDeltas = parts.filter(part => part.type === 'reasoning-delta');
    assert.equal(
      reasoningDeltas.map(part => part.delta).join(''),
      'We need answer.',
    );
    assert(reasoningDeltas.length > 0, 'reasoning deltas are emitted');
    assert(
      reasoningDeltas.every(part => part.id === reasoningDeltas[0]?.id),
      'reasoning deltas share the reasoning part id',
    );
    const reasoningStart = parts.find(part => part.type === 'reasoning-start');
    const reasoningEnd = parts.find(part => part.type === 'reasoning-end');
    assert(reasoningStart != null, 'reasoning-start is emitted');
    assert(reasoningEnd != null, 'reasoning-end is emitted');
    assert.equal(reasoningStart?.id, reasoningDeltas[0]?.id);
    assert.equal(reasoningEnd?.id, reasoningDeltas[0]?.id);
    assert(parts.some(part => part.type === 'text-delta' && part.delta === 'OK'));
  } finally {
    await scenario.close();
  }
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

test('requests are sent with stream: true on the wire', async () => {
  const scenario = await runScenario({
    steps: [response => sendJson(response, 200, responsePayload([successfulMessage()]))],
  });
  try {
    await scenario.call();
    assert.equal(scenario.bodies[0]?.stream, true);
  } finally {
    await scenario.close();
  }
});

test('SSE stream that never completes is retryable (not "Invalid JSON response")', async () => {
  const scenario = await runScenario({
    steps: [
      response =>
        sendSse(response, [
          sseFrame('response.created', { type: 'response.created', response: { id: 'resp_x' } }),
          sseFrame('response.in_progress', { type: 'response.in_progress', response: { id: 'resp_x' } }),
        ]),
    ],
  });
  try {
    await assert.rejects(scenario.call(), (error: unknown) => {
      assert(APICallError.isInstance(error));
      assert.doesNotMatch((error as APICallError).message, /Invalid JSON response/);
      assert.equal((error as APICallError).statusCode, 503);
      assert.equal(isOpenCodeRetryableError(error), true);
      return true;
    });
  } finally {
    await scenario.close();
  }
});

test('JSON keepalive placeholder is retryable (not "Invalid JSON response")', async () => {
  const keepalive = {
    id: 'resp_photonmark_keepalive',
    object: 'response',
    created_at: 0,
    status: 'in_progress',
    output: [],
    error: null,
    usage: null,
  };
  const scenario = await runScenario({
    steps: [response => sendJson(response, 200, keepalive)],
  });
  try {
    await assert.rejects(scenario.call(), (error: unknown) => {
      assert(APICallError.isInstance(error));
      assert.doesNotMatch((error as APICallError).message, /Invalid JSON response/);
      assert.match((error as APICallError).message, /in_progress|non-terminal/);
      assert.equal((error as APICallError).statusCode, 503);
      assert.equal(isOpenCodeRetryableError(error), true);
      return true;
    });
  } finally {
    await scenario.close();
  }
});

test('completed response with empty output array is retryable (not a silent stop)', async () => {
  const scenario = await runScenario({
    steps: [response => sendJson(response, 200, responsePayload([]))],
  });
  try {
    await assert.rejects(scenario.call(), (error: unknown) => {
      assert(APICallError.isInstance(error));
      assert.match((error as APICallError).message, /empty output/);
      assert.equal((error as APICallError).statusCode, 503);
      assert.equal(isOpenCodeRetryableError(error), true);
      return true;
    });
  } finally {
    await scenario.close();
  }
});

test('SSE completed response with empty output array is retryable', async () => {
  const completed = responsePayload([]);
  const scenario = await runScenario({
    steps: [
      response =>
        sendSse(response, [
          sseFrame('response.created', { type: 'response.created', response: { id: 'resp_x' } }),
          sseFrame('response.completed', { type: 'response.completed', response: completed }),
        ]),
    ],
  });
  try {
    await assert.rejects(scenario.call(), (error: unknown) => {
      assert(APICallError.isInstance(error));
      assert.match((error as APICallError).message, /empty output/);
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
