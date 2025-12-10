# n8n OpenWebUI Agent Integration

This repository contains an n8n workflow template that allows you to integrate n8n agents with [Open WebUI](https://openwebui.com), enabling you to chat with your n8n-powered AI agents directly through the Open WebUI interface. 

If you self-host n8n and have local LLMs through something like Ollama, this solution can be used to run n8n powered agents entirely offline!

This template needs to be used alongside [this Open WebUI function](https://openwebui.com/f/coleam/n8n_pipe) that integrates with n8n.

## Quickstart

This quickstart assumes you already have n8n installed and running.

1. Install Open WebUI - instructions in the [Open WebUI GitHub repository](https://github.com/open-webui/open-webui
)
2. Go to [this Open WebUI function](https://openwebui.com/f/coleam/n8n_pipe), click "Get", and enter in the URL for your Open WebUI instance (i.e. http://localhost:3000)
3. Import `Open_WebUI_Agent_Template.json` into your n8n instance, configure the webhook endpoint & security, connect the different services like OpenAI (or your provider of choice), and customize the agent to your needs.
4. Go to "Profile (bottom left) -> Admin Panel -> Functions" within Open WebUI, click on the settings icon for the n8n Pipe function, and enter your n8n webhook URL, bearer token, input field key, and output field key. These are the "valves" for your function so it can connect specifically to your n8n agent.

## Overview

The `Open_WebUI_Agent_Template.json` workflow provides a ready-to-use template for creating an n8n agent that can be accessed through Open WebUI's function calling system. This integration enables:

- Exposing your n8n AI agents as a service through a webhook
- Maintaining conversation history with Postgres memory
- Utilizing OpenAI's language models for responses
- Including tools like web search capabilities
- Seamless integration with the Open WebUI chat interface

## How It Works

1. **Webhook Endpoint**: The workflow exposes a webhook endpoint (`invoke-n8n-agent`) that receives requests from Open WebUI.

2. **Authentication**: The webhook is secured with header authentication to ensure only authorized requests are processed.

3. **Conversation Memory**: The workflow uses Postgres Chat Memory to maintain context across conversation turns, using the session ID provided by Open WebUI.

4. **AI Processing**: The workflow uses OpenAI's language models to process user inputs and generate responses.

5. **Tool Integration**: The template includes a web search tool that allows the agent to search the web for information when needed.

6. **Response Handling**: The agent's response is formatted and sent back to Open WebUI for display to the user.

## Setup Instructions

### Prerequisites

- An n8n instance (self-hosted or cloud)
- OpenAI API credentials
- PostgreSQL database for conversation memory
- Open WebUI

### Installation Steps

1. **Import the Template**:
   - In your n8n instance, go to "Workflows"
   - Click "Import from File" and select the `Open_WebUI_Agent_Template.json` file after downloading it

2. **Configure Credentials**:
   - Set up your OpenAI API credentials
   - Configure your PostgreSQL database connection
   - Set up header authentication for the webhook

3. **Activate the Workflow**:
   - Activate the workflow to make the webhook endpoint available

4. **Connect to Open WebUI**:
   - [Visit this Open WebUI function](https://openwebui.com/f/coleam/n8n_pipe)
   - Click "Get" and insert the URL for your Open WebUI instance
   - Edit the "valves" (settings icon) to set the necessary configuration for your n8n agent

## Customization

You can customize this template by build really any AI agent you want in place of the very simple agent currently in the template that just has a single tool. You can use any LLM, any database for chat history, and any tool you would typically use with an n8n agent!
