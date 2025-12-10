# Edge Functions Integration

This directory contains the edge function implementations integrated from the Supabase project. These functions provide real-time processing capabilities for meeting transcripts and related functionality.

## Structure

```
edge/
├── types.ts           # Shared TypeScript types
├── server.ts          # Main edge server implementation
├── intent-detection.ts # Intent detection service
├── meeting-info.ts    # Meeting information handler
├── webhook-handler.ts # Webhook processing
└── deploy.ts         # Deployment utilities
```

## Features

- **Intent Detection**: AI-powered intent detection for meeting transcripts
- **Meeting Info**: Retrieval and processing of meeting information
- **Webhook Handler**: Processing of meeting-related events

## Development

1. Install Deno if not already installed:
   ```bash
   curl -fsSL https://deno.land/x/install/install.sh | sh
   ```

2. Run the development server:
   ```bash
   deno task dev
   ```

3. Test the implementation:
   ```bash
   deno task test
   ```

## Deployment

Deploy all functions:
```bash
deno run --allow-run --allow-env deploy.ts --all
```

Deploy specific function:
```bash
deno run --allow-run --allow-env deploy.ts intent-detection
```

## Environment Variables

Required environment variables:
- `OPENAI_API_KEY`: API key for OpenAI services
- `DENO_DEPLOY_TOKEN`: Token for Deno Deploy

## API Endpoints

### Intent Detection
- **POST** `/intent-detection`
- Detects intents in meeting transcripts
- Requires: `meetingId`, `transcriptionText`

### Meeting Info
- **POST** `/meeting-info`
- Retrieves meeting information and summaries
- Requires: `meetingId`

### Webhook Handler
- **POST** `/webhook`
- Handles incoming meeting events
- Requires: `meetingId`, `eventType`

## Integration Notes

1. The edge functions are designed to work with the existing federation system
2. All functions use TypeScript for type safety
3. CORS is enabled for all endpoints
4. Error handling follows a consistent pattern
5. Responses follow the standard EdgeResponse format

## Testing

Each function includes its own test file. Run all tests:

```bash
deno test --allow-net --allow-env
```

## Error Handling

All functions use a standardized error response format:
```typescript
{
  success: false,
  message: string,
  data?: any
}
