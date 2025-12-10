# Agent Booster API Server Examples

## Quick Start

Start the API server:

```bash
npm run server
# or
agent-booster-server

# Server will start on http://localhost:3000
```

## Endpoint 1: `/v1/chat/completions` (Morph LLM Compatible)

This endpoint is 100% compatible with Morph LLM's chat completions format.

### Request Example

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "agent-booster-v1",
    "messages": [{
      "role": "user",
      "content": "<instruction>Add error handling to divide function</instruction><code>function divide(a, b) { return a / b; }</code><update>function divide(a, b) { if (b === 0) throw new Error(\"Division by zero\"); return a / b; }</update>"
    }]
  }'
```

### Response Example

```json
{
  "id": "chatcmpl-1704067200000",
  "object": "chat.completion",
  "created": 1704067200,
  "model": "agent-booster-v1",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "function divide(a, b) { if (b === 0) throw new Error('Division by zero'); return a / b; }"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 30,
    "total_tokens": 55
  },
  "confidence": 0.90,
  "strategy": "exact_replace",
  "latency": 1
}
```

## Endpoint 2: `/v1/apply` (Simplified)

Direct apply endpoint without the chat format wrapper.

### Request Example

```bash
curl -X POST http://localhost:3000/v1/apply \
  -H "Content-Type: application/json" \
  -d '{
    "code": "function add(a, b) { return a + b; }",
    "edit": "function add(a: number, b: number): number { return a + b; }",
    "language": "typescript"
  }'
```

### Response Example

```json
{
  "output": "function add(a: number, b: number): number { return a + b; }",
  "success": true,
  "latency": 1,
  "tokens": {
    "input": 10,
    "output": 15
  },
  "confidence": 0.85,
  "strategy": "exact_replace"
}
```

## Endpoint 3: `/v1/batch` (Batch Processing)

Process multiple edits in a single request.

### Request Example

```bash
curl -X POST http://localhost:3000/v1/batch \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {
        "code": "var x = 1;",
        "edit": "const x = 1;",
        "language": "javascript"
      },
      {
        "code": "function test() { return 42; }",
        "edit": "const test = () => 42;",
        "language": "javascript"
      }
    ]
  }'
```

### Response Example

```json
{
  "results": [
    {
      "output": "const x = 1;",
      "success": true,
      "latency": 0,
      "tokens": { "input": 2, "output": 2 },
      "confidence": 0.95,
      "strategy": "exact_replace"
    },
    {
      "output": "const test = () => 42;",
      "success": true,
      "latency": 1,
      "tokens": { "input": 8, "output": 6 },
      "confidence": 0.80,
      "strategy": "exact_replace"
    }
  ],
  "total": 2,
  "successful": 2
}
```

## JavaScript/TypeScript Client Examples

### Using with Morph LLM SDK (Drop-in Replacement)

```typescript
// Just change the base URL - everything else stays the same!
const response = await fetch('http://localhost:3000/v1/chat/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'agent-booster-v1',
    messages: [{
      role: 'user',
      content: `<instruction>Add TypeScript types</instruction><code>${code}</code><update>${typedCode}</update>`
    }]
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);
// Morph LLM: 352ms, $0.01
// Agent Booster: 1ms, $0.00
```

### Using Direct Apply Endpoint

```typescript
async function applyEdit(code: string, edit: string, language: string) {
  const response = await fetch('http://localhost:3000/v1/apply', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code, edit, language })
  });

  const result = await response.json();

  if (result.success) {
    console.log('Applied edit with confidence:', result.confidence);
    return result.output;
  } else {
    throw new Error('Edit failed');
  }
}
```

### Batch Processing Example

```typescript
async function batchRefactor(files: Array<{code: string, edit: string}>) {
  const response = await fetch('http://localhost:3000/v1/batch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      requests: files.map(f => ({
        code: f.code,
        edit: f.edit,
        language: 'typescript'
      }))
    })
  });

  const { results, total, successful } = await response.json();
  console.log(`Processed ${successful}/${total} files successfully`);

  return results;
}
```

## Environment Variables

Configure the server with environment variables:

```bash
# Server configuration
PORT=3000                    # Server port (default: 3000)
HOST=0.0.0.0                 # Server host (default: 0.0.0.0)

# Agent Booster configuration
CONFIDENCE_THRESHOLD=0.5     # Minimum confidence (default: 0.5)
MAX_CHUNKS=100               # Max code chunks (default: 100)

# Start server
npm run server
```

## Performance Comparison

### Agent Booster API vs Morph LLM API

| Metric | Morph LLM API | Agent Booster API | Improvement |
|--------|---------------|-------------------|-------------|
| Latency | 352ms | 1ms | **352x faster** |
| Cost per request | $0.01 | $0.00 | **100% free** |
| Privacy | Cloud-based | **100% local** | Private |
| Throughput | 2.8 req/s | **1000 req/s** | 357x higher |

### Real-World Example

**Scenario:** AI coding agent making 100 code transformations

```
Morph LLM API:
- Time: 35.2 seconds
- Cost: $1.00
- Data sent to cloud: All code

Agent Booster API:
- Time: 0.1 seconds  (352x faster)
- Cost: $0.00        (100% savings)
- Data sent to cloud: None (private)
```

## Docker Deployment

```dockerfile
FROM node:18-alpine

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY dist/ ./dist/
COPY wasm/ ./wasm/

EXPOSE 3000
CMD ["npm", "run", "server"]
```

```bash
docker build -t agent-booster-api .
docker run -p 3000:3000 agent-booster-api
```

## Health Check

```bash
curl http://localhost:3000/health
```

Response:
```json
{
  "status": "ok",
  "version": "0.1.0",
  "uptime": 3600
}
```

## Error Handling

All errors return standard error format:

```json
{
  "error": {
    "message": "code and edit fields are required",
    "type": "invalid_request_error"
  }
}
```

Error types:
- `invalid_request_error` - Bad request (400)
- `internal_error` - Server error (500)
