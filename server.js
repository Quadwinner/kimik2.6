const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");
const { AnthropicFoundry } = require("@anthropic-ai/foundry-sdk");
const { OpenAI } = require("openai");

const rootDir = __dirname;
const publicDir = path.join(rootDir, "public");
const PORT = process.env.PORT || 3000;

function loadEnvFile(filePath) {
  if (!fs.existsSync(filePath)) return;

  const lines = fs.readFileSync(filePath, "utf8").split(/\r?\n/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;

    const separatorIndex = trimmed.indexOf("=");
    if (separatorIndex === -1) continue;

    const key = trimmed.slice(0, separatorIndex).trim();
    const rawValue = trimmed.slice(separatorIndex + 1).trim();
    const value = rawValue.replace(/^["']|["']$/g, "");

    if (!process.env[key]) {
      process.env[key] = value;
    }
  }
}

loadEnvFile(path.join(rootDir, ".env.local"));

let azureClient;
let anthropicClient;

function getAzureConfig() {
  return {
    endpoint: process.env.AZURE_OPENAI_ENDPOINT || process.env.AZURE_ENDPOINT,
    apiKey: process.env.AZURE_OPENAI_API_KEY || process.env.AZURE_API_KEY,
    deployment: process.env.AZURE_OPENAI_DEPLOYMENT || process.env.AZURE_DEPLOYMENT || "Kimi-K2.6",
  };
}

function getAzureClient() {
  const { endpoint, apiKey } = getAzureConfig();

  if (!endpoint || !apiKey) {
    return null;
  }

  if (!azureClient) {
    azureClient = new OpenAI({
      baseURL: endpoint,
      apiKey,
    });
  }

  return azureClient;
}

function getAnthropicConfig() {
  return {
    endpoint:
      process.env.ANTHROPIC_FOUNDRY_ENDPOINT ||
      process.env.ANTHROPIC_FOUNDRY_BASE_URL,
    apiKey: process.env.ANTHROPIC_FOUNDRY_API_KEY,
    deployment: process.env.ANTHROPIC_FOUNDRY_DEPLOYMENT || "claude-opus-4-7",
    apiVersion: process.env.ANTHROPIC_FOUNDRY_API_VERSION || "2023-06-01",
  };
}

function getAnthropicClient() {
  const { endpoint, apiKey, apiVersion } = getAnthropicConfig();

  if (!endpoint || !apiKey) {
    return null;
  }

  if (!anthropicClient) {
    anthropicClient = new AnthropicFoundry({
      apiKey,
      baseURL: endpoint,
      apiVersion,
    });
  }

  return anthropicClient;
}

function getMissingAzureConfigMessage() {
  return "Missing Azure configuration. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY, or AZURE_ENDPOINT and AZURE_API_KEY.";
}

function getMissingAnthropicConfigMessage() {
  return "Missing Anthropic Foundry configuration. Set ANTHROPIC_FOUNDRY_ENDPOINT, ANTHROPIC_FOUNDRY_API_KEY, and ANTHROPIC_FOUNDRY_DEPLOYMENT.";
}

function sendJson(res, statusCode, payload) {
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
  });
  res.end(JSON.stringify(payload));
}

function sendOpenAIError(res, statusCode, message, type = "server_error") {
  sendJson(res, statusCode, {
    error: {
      message,
      type,
      code: null,
    },
  });
}

function getRequestBody(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk;
      if (body.length > 16 * 1024 * 1024) {
        req.destroy();
        reject(new Error("Request is too large. Please upload a smaller image."));
      }
    });
    req.on("end", () => resolve(body));
    req.on("error", reject);
  });
}

async function getJsonPayload(req) {
  const body = await getRequestBody(req);

  try {
    return JSON.parse(body || "{}");
  } catch {
    const error = new Error("Invalid JSON request body.");
    error.statusCode = 400;
    error.type = "invalid_request_error";
    throw error;
  }
}

function getDeploymentName() {
  return getAzureConfig().deployment;
}

function getAnthropicDeploymentName() {
  return getAnthropicConfig().deployment;
}

function shouldUseAnthropic(payload) {
  return payload.model === getAnthropicDeploymentName();
}

function buildAzureChatParams(payload, options = {}) {
  const messages = Array.isArray(payload.messages) ? payload.messages : [];

  if (messages.length === 0) {
    const error = new Error("Message is required.");
    error.statusCode = 400;
    error.type = "invalid_request_error";
    throw error;
  }

  const allowedParams = [
    "messages",
    "temperature",
    "top_p",
    "n",
    "stream",
    "stop",
    "max_tokens",
    "max_completion_tokens",
    "presence_penalty",
    "frequency_penalty",
    "response_format",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "seed",
    "user",
    "logprobs",
    "top_logprobs",
  ];

  const params = {};
  for (const key of allowedParams) {
    if (payload[key] !== undefined) {
      params[key] = payload[key];
    }
  }

  params.model = getDeploymentName();
  params.messages = messages;

  if (
    options.defaultMaxTokens &&
    params.max_tokens === undefined &&
    params.max_completion_tokens === undefined
  ) {
    params.max_tokens = options.defaultMaxTokens;
  }

  return params;
}

function toPlainObject(value) {
  return JSON.parse(JSON.stringify(value));
}

function normalizeChatCompletion(completion) {
  const normalized = toPlainObject(completion);

  normalized.choices = (normalized.choices || []).map((choice) => {
    const message = choice.message || {};
    const openAIMessage = { ...message };
    delete openAIMessage.reasoning_content;

    return {
      ...choice,
      message: {
        ...openAIMessage,
        role: openAIMessage.role || "assistant",
        content: openAIMessage.content || "",
      },
    };
  });

  return normalized;
}

function normalizeChatCompletionChunk(chunk) {
  const normalized = toPlainObject(chunk);

  normalized.choices = (normalized.choices || [])
    .map((choice) => {
      const delta = choice.delta || {};
      const openAIDelta = { ...delta };
      delete openAIDelta.reasoning_content;
      const hasDelta = Object.keys(openAIDelta).length > 0;
      const hasFinishReason = choice.finish_reason !== null && choice.finish_reason !== undefined;

      if (!hasDelta && !hasFinishReason) {
        return null;
      }

      return {
        ...choice,
        delta: openAIDelta,
      };
    })
    .filter(Boolean);

  return normalized.choices.length > 0 ? normalized : null;
}

function getTextFromContent(content) {
  if (typeof content === "string") {
    return content;
  }

  if (!Array.isArray(content)) {
    return "";
  }

  return content
    .map((part) => {
      if (typeof part === "string") return part;
      if (part?.type === "text" || part?.type === "input_text") return part.text || "";
      return "";
    })
    .filter(Boolean)
    .join("\n");
}

function convertOpenAIContentToAnthropic(content) {
  if (typeof content === "string") {
    return content;
  }

  if (!Array.isArray(content)) {
    return "";
  }

  return content
    .map((part) => {
      if (typeof part === "string") {
        return { type: "text", text: part };
      }

      if (part?.type === "text" || part?.type === "input_text") {
        return { type: "text", text: part.text || "" };
      }

      return null;
    })
    .filter(Boolean);
}

function convertOpenAIToolsToAnthropic(tools) {
  if (!Array.isArray(tools)) {
    return undefined;
  }

  const convertedTools = tools
    .filter((tool) => tool?.type === "function" && tool.function?.name)
    .map((tool) => ({
      name: tool.function.name,
      description: tool.function.description || "",
      input_schema: tool.function.parameters || {
        type: "object",
        properties: {},
      },
    }));

  return convertedTools.length > 0 ? convertedTools : undefined;
}

function convertOpenAIToolChoiceToAnthropic(toolChoice) {
  if (!toolChoice || toolChoice === "auto") {
    return undefined;
  }

  if (toolChoice === "none") {
    return { type: "none" };
  }

  if (toolChoice === "required") {
    return { type: "any" };
  }

  if (toolChoice?.type === "function" && toolChoice.function?.name) {
    return { type: "tool", name: toolChoice.function.name };
  }

  return undefined;
}

function convertOpenAIMessagesToAnthropic(messages) {
  const anthropicMessages = [];
  const systemMessages = [];

  for (const message of messages) {
    if (message.role === "system" || message.role === "developer") {
      const systemText = getTextFromContent(message.content);
      if (systemText) systemMessages.push(systemText);
      continue;
    }

    if (message.role === "tool") {
      anthropicMessages.push({
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: message.tool_call_id,
            content: getTextFromContent(message.content),
          },
        ],
      });
      continue;
    }

    const role = message.role === "assistant" ? "assistant" : "user";
    const content = convertOpenAIContentToAnthropic(message.content);
    const contentBlocks = Array.isArray(content)
      ? [...content]
      : content
        ? [{ type: "text", text: content }]
        : [];

    if (message.role === "assistant" && Array.isArray(message.tool_calls)) {
      for (const toolCall of message.tool_calls) {
        if (toolCall?.type !== "function" || !toolCall.function?.name) continue;
        let input = {};
        try {
          input = JSON.parse(toolCall.function.arguments || "{}");
        } catch {
          input = { arguments: toolCall.function.arguments || "" };
        }

        contentBlocks.push({
          type: "tool_use",
          id: toolCall.id,
          name: toolCall.function.name,
          input,
        });
      }
    }

    anthropicMessages.push({
      role,
      content: contentBlocks.length > 0 ? contentBlocks : "",
    });
  }

  return {
    messages: anthropicMessages,
    system: systemMessages.length > 0 ? systemMessages.join("\n\n") : undefined,
  };
}

function buildAnthropicMessageParams(payload, options = {}) {
  const messages = Array.isArray(payload.messages) ? payload.messages : [];

  if (messages.length === 0) {
    const error = new Error("Message is required.");
    error.statusCode = 400;
    error.type = "invalid_request_error";
    throw error;
  }

  const converted = convertOpenAIMessagesToAnthropic(messages);
  const params = {
    model: getAnthropicDeploymentName(),
    messages: converted.messages,
    max_tokens:
      payload.max_tokens ||
      payload.max_completion_tokens ||
      options.defaultMaxTokens ||
      1024,
  };

  if (converted.system) {
    params.system = converted.system;
  }

  if (payload.stop !== undefined) {
    params.stop_sequences = Array.isArray(payload.stop) ? payload.stop : [payload.stop];
  }

  const tools = convertOpenAIToolsToAnthropic(payload.tools);
  if (tools) {
    params.tools = tools;
  }

  const toolChoice = convertOpenAIToolChoiceToAnthropic(payload.tool_choice);
  if (toolChoice) {
    params.tool_choice = toolChoice;
  }

  if (payload.temperature === 1) {
    params.temperature = 1;
  }

  return params;
}

function getAnthropicTextContent(message) {
  return (message.content || [])
    .filter((block) => block.type === "text")
    .map((block) => block.text || "")
    .join("");
}

function getAnthropicToolCalls(message) {
  return (message.content || [])
    .filter((block) => block.type === "tool_use")
    .map((block) => ({
      id: block.id,
      type: "function",
      function: {
        name: block.name,
        arguments: JSON.stringify(block.input || {}),
      },
    }));
}

function mapAnthropicFinishReason(stopReason) {
  switch (stopReason) {
    case "end_turn":
    case "stop_sequence":
    case "pause_turn":
      return "stop";
    case "max_tokens":
      return "length";
    case "tool_use":
      return "tool_calls";
    case "refusal":
      return "content_filter";
    default:
      return "stop";
  }
}

function normalizeAnthropicUsage(usage) {
  if (!usage) return null;

  const promptTokens = usage.input_tokens || 0;
  const completionTokens = usage.output_tokens || 0;

  return {
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: promptTokens + completionTokens,
  };
}

function normalizeAnthropicCompletion(message) {
  const toolCalls = getAnthropicToolCalls(message);
  const openAIMessage = {
    role: "assistant",
    content: getAnthropicTextContent(message),
  };

  if (toolCalls.length > 0) {
    openAIMessage.tool_calls = toolCalls;
  }

  return {
    id: message.id,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: getAnthropicDeploymentName(),
    choices: [
      {
        index: 0,
        message: openAIMessage,
        finish_reason: mapAnthropicFinishReason(message.stop_reason),
      },
    ],
    usage: normalizeAnthropicUsage(message.usage),
  };
}

function getContentType(filePath) {
  const extension = path.extname(filePath);
  switch (extension) {
    case ".html":
      return "text/html; charset=utf-8";
    case ".css":
      return "text/css; charset=utf-8";
    case ".js":
      return "text/javascript; charset=utf-8";
    case ".svg":
      return "image/svg+xml";
    default:
      return "application/octet-stream";
  }
}

function serveStatic(req, res) {
  const requestUrl = new URL(req.url, `http://${req.headers.host}`);
  const safePath = path.normalize(decodeURIComponent(requestUrl.pathname)).replace(/^(\.\.[/\\])+/, "");
  const filePath = path.join(publicDir, safePath === "/" ? "index.html" : safePath);

  if (!filePath.startsWith(publicDir)) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }

  fs.readFile(filePath, (error, data) => {
    if (error) {
      res.writeHead(404);
      res.end("Not found");
      return;
    }

    res.writeHead(200, {
      "Content-Type": getContentType(filePath),
      "Cache-Control": "no-store",
    });
    res.end(data);
  });
}

async function handleChat(req, res) {
  const client = getAzureClient();

  if (!client) {
    sendJson(res, 500, { error: getMissingAzureConfigMessage() });
    return;
  }

  try {
    const payload = await getJsonPayload(req);
    const completion = normalizeChatCompletion(
      await client.chat.completions.create(
        buildAzureChatParams({ ...payload, stream: false }, { defaultMaxTokens: 1024 })
      )
    );

    sendJson(res, 200, {
      message: completion.choices?.[0]?.message?.content || "",
      usage: completion.usage || null,
    });
  } catch (error) {
    const statusCode = error.status || error.statusCode || 500;
    sendJson(res, statusCode, {
      error: error.message || "Request failed.",
      statusCode,
    });
  }
}

function handleModels(req, res) {
  const kimiDeployment = getDeploymentName();
  const anthropicDeployment = getAnthropicDeploymentName();

  sendJson(res, 200, {
    object: "list",
    data: [
      {
        id: "kimi",
        object: "model",
        created: 0,
        owned_by: "local-azure-proxy",
      },
      {
        id: kimiDeployment,
        object: "model",
        created: 0,
        owned_by: "local-azure-proxy",
      },
      {
        id: anthropicDeployment,
        object: "model",
        created: 0,
        owned_by: "local-anthropic-foundry-proxy",
      },
    ],
  });
}

async function streamOpenAICompletion(res, params) {
  const client = getAzureClient();

  if (!client) {
    throw new Error(getMissingAzureConfigMessage());
  }

  res.writeHead(200, {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no",
  });

  const stream = await client.chat.completions.create({ ...params, stream: true });

  for await (const chunk of stream) {
    if (res.destroyed) return;

    const normalized = normalizeChatCompletionChunk(chunk);
    if (normalized) {
      res.write(`data: ${JSON.stringify(normalized)}\n\n`);
    }
  }

  res.write("data: [DONE]\n\n");
  res.end();
}

async function createAnthropicCompletion(params) {
  const client = getAnthropicClient();

  if (!client) {
    throw new Error(getMissingAnthropicConfigMessage());
  }

  const message = await client.messages.create({ ...params, stream: false });
  return normalizeAnthropicCompletion(message);
}

async function streamAnthropicCompletion(res, params) {
  res.writeHead(200, {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no",
  });

  const completion = await createAnthropicCompletion(params);
  const choice = completion.choices[0];

  res.write(
    `data: ${JSON.stringify({
      id: completion.id,
      object: "chat.completion.chunk",
      created: completion.created,
      model: completion.model,
      choices: [
        {
          index: 0,
          delta: choice.message,
          finish_reason: choice.finish_reason,
        },
      ],
      usage: completion.usage,
    })}\n\n`
  );
  res.write("data: [DONE]\n\n");
  res.end();
}

async function handleOpenAIChatCompletion(req, res) {
  try {
    const payload = await getJsonPayload(req);

    if (shouldUseAnthropic(payload)) {
      const params = buildAnthropicMessageParams(payload);

      if (payload.stream === true) {
        await streamAnthropicCompletion(res, params);
        return;
      }

      const completion = await createAnthropicCompletion(params);
      sendJson(res, 200, completion);
      return;
    }

    const client = getAzureClient();

    if (!client) {
      sendOpenAIError(res, 500, getMissingAzureConfigMessage());
      return;
    }

    const params = buildAzureChatParams(payload);

    if (payload.stream === true) {
      await streamOpenAICompletion(res, params);
      return;
    }

    const completion = normalizeChatCompletion(
      await client.chat.completions.create({ ...params, stream: false })
    );
    sendJson(res, 200, completion);
  } catch (error) {
    const statusCode = error.status || error.statusCode || 500;
    const message = error.message || "Request failed.";

    if (res.headersSent) {
      const errorPayload = {
        error: { message, type: error.type || "server_error", code: null },
      };
      res.write(`data: ${JSON.stringify(errorPayload)}\n\n`);
      res.write("data: [DONE]\n\n");
      res.end();
      return;
    }

    sendOpenAIError(res, statusCode, message, error.type || "server_error");
  }
}

const server = http.createServer((req, res) => {
  const requestUrl = new URL(req.url, `http://${req.headers.host}`);

  if (req.method === "POST" && requestUrl.pathname === "/api/chat") {
    handleChat(req, res);
    return;
  }

  if (req.method === "POST" && requestUrl.pathname === "/v1/chat/completions") {
    handleOpenAIChatCompletion(req, res);
    return;
  }

  if (req.method === "GET" && requestUrl.pathname === "/v1/models") {
    handleModels(req, res);
    return;
  }

  if (req.method === "GET") {
    serveStatic(req, res);
    return;
  }

  res.writeHead(405);
  res.end("Method not allowed");
});

server.listen(PORT, () => {
  console.log("running on", PORT);
});
