# Edge Functions Guide

## Overview
The system supports deploying serverless edge functions across multiple providers:
- Supabase Edge Functions
- Cloudflare Workers  
- Fly.io

## Available Functions

### 1. Intent Detection
Endpoint: `/intent-detection`
- Detects intents in meeting transcripts using AI
- Uses OpenAI GPT-4 for analysis
- Returns structured intent data

### 2. Meeting Info
Endpoint: `/meeting-info`
- Retrieves meeting information and summaries
- Integrates with Fireflies API
- Provides comprehensive meeting data

### 3. Webhook Handler
Endpoint: `/webhook`
- Processes meeting-related webhooks
- Handles event notifications
- Manages asynchronous updates

## Deployment

### Provider Configuration
1. Select provider from menu
2. Configure authentication credentials
3. Deploy functions
4. Monitor status

### Environment Variables
Required variables per provider:
- Supabase:
  - SUPABASE_PROJECT_ID
  - SUPABASE_ACCESS_TOKEN
- Cloudflare:
  - CLOUDFLARE_API_TOKEN
  - CLOUDFLARE_ACCOUNT_ID
- Fly.io:
  - FLY_API_TOKEN
  - FLY_APP_NAME

## Monitoring

### Status Checking
```typescript
// View function status
viewEdgeFunctionStatus();

// View function logs
viewEdgeFunctionLogs();

// List deployed functions
listDeployedFunctions();
```

### Logs
- Real-time log streaming
- Error tracking
- Performance monitoring
