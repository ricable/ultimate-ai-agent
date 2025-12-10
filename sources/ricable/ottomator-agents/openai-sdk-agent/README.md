# OpenAI Agents SDK Demo

This repository contains examples of using the OpenAI Agents SDK to build intelligent travel planning agents with progressively advanced capabilities.

## Project Structure

- `v1_basic_agent.py` - A simple agent example that generates a haiku about recursion
- `v2_structured_output.py` - Travel agent with structured output using Pydantic models
- `v3_tool_calls.py` - Travel agent with tool calls for weather forecasting
- `v4_handoffs.py` - Travel agent with specialized sub-agents for flights and hotels
- `v5_guardrails_and_context.py` - Travel agent with budget guardrails and user context
- `v6_streamlit_agent.py` - A Streamlit web interface for the travel agent with chat memory

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
MODEL_CHOICE=gpt-4o-mini  # or another model of your choice
```

## Running the Examples

### Basic Agent (v1)

Run the basic agent example:

```bash
python v1_basic_agent.py
```

This will execute a simple agent that generates a haiku about recursion.

### Structured Output Agent (v2)

Run the structured output travel agent example:

```bash
python v2_structured_output.py
```

This demonstrates using Pydantic models to create structured travel plans with destinations, activities, and budget information.

### Tool Calls Agent (v3)

Run the tool calls travel agent example:

```bash
python v3_tool_calls.py
```

This version adds a weather forecasting tool to provide weather information for travel destinations.

### Handoffs Agent (v4)

Run the handoffs travel agent example:

```bash
python v4_handoffs.py
```

This version introduces specialized sub-agents for flight and hotel recommendations, demonstrating agent handoffs.

### Guardrails and Context Agent (v5)

Run the guardrails and context travel agent example:

```bash
python v5_guardrails_and_context.py
```

This version adds:
- Budget analysis guardrails to validate if a travel budget is realistic
- User context to store and use preferences like preferred airlines and hotel amenities

[Optional] Follow the [Logfire setup intructions](https://logfire.pydantic.dev/docs/#logfire) (free to get started) for tracing in this version and version 6. This example will still work with Logfire configured but you won't get tracing.

### Streamlit Chat Interface (v6)

Launch the Streamlit web interface:

```bash
streamlit run v6_streamlit_agent.py
```

This will start a web server and open a browser window with the travel agent chat interface. Features include:

- Persistent chat history within a session
- User preference management in the sidebar
- Beautifully formatted responses for different types of travel information
- Support for conversation memory across multiple turns

## Environment Variables

The following environment variables can be configured in your `.env` file:

- `OPENAI_API_KEY` (required): Your OpenAI API key
- `MODEL_CHOICE` (optional): The OpenAI model to use (default: gpt-4o-mini)

## Features Demonstrated

1. **Basic Agent Configuration (v1)**
   - Instructions and model settings
   - Simple agent execution

2. **Structured Output (v2)**
   - Using Pydantic models for structured responses
   - Travel planning with organized information

3. **Tool Calls (v3)**
   - Custom tools for retrieving external data
   - Weather forecasting integration

4. **Agent Handoffs (v4)**
   - Specialized agents for flights and hotels
   - Delegation to domain-specific experts

5. **Guardrails and Context (v5)**
   - Input validation with budget guardrails
   - User context for personalized recommendations
   - Preference-based sorting of results

6. **Chat Interface (v6)**
   - Conversation history and context
   - User preference management
   - Formatted responses for different output types
   - Thread management for persistent conversations

## Notes

This is a demonstration project and uses simulated data for weather, flights, and hotels. In a production environment, you would integrate with real APIs for this information.
