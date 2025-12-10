# Supabase Edge Function Deployment Guide

This guide explains how to set up and deploy edge functions to Supabase using the MCP server.

## Prerequisites

1. Install Supabase CLI:
```bash
# macOS
brew install supabase/tap/supabase

# Windows (requires scoop)
scoop bucket add supabase https://github.com/supabase/scoop-bucket.git
scoop install supabase

# Linux
curl -fsSL https://cli.supabase.com/install.sh | sh
```

2. Log in to Supabase:
```bash
supabase login
```

## Configuration

1. Create a `.env` file in the project root:
```bash
touch .env
```

2. Add the following environment variables to `.env`:
```env
SUPABASE_PROJECT_ID=your_project_id
SUPABASE_ACCESS_TOKEN=your_access_token
```

To get these values:

1. **Project ID**:
   - Go to your Supabase project dashboard
   - Click on Settings -> API
   - Copy the "Project ID" value

2. **Access Token**:
   - Go to https://supabase.com/dashboard/account/tokens
   - Generate a new access token
   - Copy the token value

## Project Setup

1. Initialize Supabase in your project:
```bash
supabase init
```

2. Link to your Supabase project:
```bash
supabase link --project-ref your_project_id
```

## Using the MCP Server

1. Start the server with environment variables:
```bash
export $(cat .env | xargs) && deno run --allow-net --allow-env --allow-read --allow-write --allow-run apps/deno/server.ts
```

2. Deploy Edge Functions:
   - Select option [5] from the menu
   - Choose the function to deploy
   - The server will handle the deployment process

## Available Edge Functions

1. Intent Detection (`intent-detection`)
   - AI-powered intent detection for meeting transcripts
   - Uses OpenAI GPT-4

2. Meeting Info (`meeting-info`)
   - Meeting information and summary handler
   - Integrates with Fireflies API

3. Webhook Handler (`webhook-handler`)
   - Processes meeting-related webhooks
   - Handles event notifications

## Deployment Process

When deploying a function:
1. The server creates necessary Supabase function directories
2. Copies the function code to the Supabase structure
3. Deploys using Supabase CLI
4. Provides the deployed function URL

## Monitoring

- Use option [6] to view function status
- Use option [7] to view function logs
- Use option [8] to list all deployed functions

## Troubleshooting

1. **Missing Configuration**
   ```
   Error: Missing Supabase configuration
   ```
   - Ensure `.env` file exists with required variables
   - Verify environment variables are properly exported

2. **Deployment Failures**
   - Check Supabase CLI is installed and logged in
   - Verify project linking is correct
   - Check function logs for detailed error messages

3. **Permission Issues**
   - Ensure proper Supabase access token permissions
   - Verify project access rights

## Security Notes

- Keep your access token secure
- Don't commit `.env` file to version control
- Regularly rotate access tokens
- Monitor function access logs

## Additional Resources

- [Supabase Edge Functions Documentation](https://supabase.com/docs/guides/functions)
- [Supabase CLI Reference](https://supabase.com/docs/reference/cli)
- [Edge Functions Examples](https://github.com/supabase/supabase/tree/master/examples/edge-functions)
