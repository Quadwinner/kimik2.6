const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");
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

function getMissingAzureConfigMessage() {
  return "Missing Azure configuration. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY, or AZURE_ENDPOINT and AZURE_API_KEY.";
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
  sendJson(res, 200, {
    object: "list",
    data: [
      {
        id: "kimi",
        object: "model",
        created: 0,
        owned_by: "local-azure-proxy",
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

async function handleOpenAIChatCompletion(req, res) {
  const client = getAzureClient();

  if (!client) {
    sendOpenAIError(res, 500, getMissingAzureConfigMessage());
    return;
  }

  try {
    const payload = await getJsonPayload(req);
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
