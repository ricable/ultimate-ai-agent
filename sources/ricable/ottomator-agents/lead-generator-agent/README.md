# Pydantic AI: Lead Generator Agent

Author: [Asvin Kumar](https://www.linkedin.com/in/asvin-kumar-1107/)

An intelligent lead generation agent built using Pydantic AI and Hunter.io API, capable of finding business email addresses, verifying emails, and generating leads. The agent can search domains for email addresses, verify email validity, and provide detailed statistics about email distribution within organizations.

**Note**: For the Live Agent Studio (see the studio_ versions of the scripts), the Hunter.io calls have been mocked since the API does not scale for the entire community trying this agent on the Studio.

## Features

- Domain email search and analysis
- Email address verification
- Email count statistics by department
- Real-time processing status updates
- Conversation history tracking with Supabase
- Support for both OpenAI and OpenRouter models
- Available as API endpoint with FastAPI

## Prerequisites

- Python 3.11+
- Hunter.io API key
- OpenRouter API key
- Supabase account for conversation storage

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ottomator-agents.git
cd ottomator-agents/lead-generator-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - Create a `.env` file with your API keys and preferences:
   ```env
   HUNTER_API_KEY=your_hunter_api_key
   OPEN_ROUTER_API_KEY=your_openrouter_api_key
   LLM_MODEL=your_chosen_model  # e.g., deepseek/deepseek-chat
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_KEY=your_supabase_key
   API_BEARER_TOKEN=your_bearer_token
   ```

## Usage

### Running with Docker

1. Build the Docker image:
```bash
docker build -t lead-generator-agent .
```

2. Run the container:
```bash
docker run -d -p 8001:8001 --env-file .env lead-generator-agent
```

### API Endpoints

The main endpoint is available at: `POST http://localhost:8001/api/lead-generator`

Example request body:
```json
{
    "query": "Find email addresses for google.com",
    "user_id": "test_user",
    "request_id": "123",
    "session_id": "test_session"
}
```

### Example Queries

1. Domain Email Search:
```json
{
    "query": "Find all email addresses for microsoft.com",
    "user_id": "user1",
    "request_id": "req1",
    "session_id": "session1"
}
```

2. Email Statistics:
```json
{
    "query": "Get email count and department statistics for apple.com",
    "user_id": "user1",
    "request_id": "req2",
    "session_id": "session1"
}
```

3. Email Verification:
```json
{
    "query": "Verify if john.doe@example.com is valid",
    "user_id": "user1",
    "request_id": "req3",
    "session_id": "session1"
}
```

4. Department-Specific Search:
```json
{
    "query": "Find IT department emails in amazon.com",
    "user_id": "user1",
    "request_id": "req4",
    "session_id": "session1"
}
```

## Response Format

The agent provides detailed responses including:
- Domain Information
- Email Addresses Found
- Verification Status
- Department Statistics
- Confidence Scores
- Position Information

Example response in Supabase:
```json
{
    "type": "ai",
    "content": {
        "domain": "example.com",
        "emails": [
            {
                "email": "john.doe@example.com",
                "type": "personal",
                "confidence": 95,
                "position": "Software Engineer"
            }
        ],
        "statistics": {
            "total": 150,
            "personal": 120,
            "generic": 30
        }
    }
}
```

## Error Handling

The agent includes built-in error handling for:
- Invalid domains
- Rate limiting
- Authentication issues
- API timeouts
- Invalid email formats

## Project Structure

- `leadgen_agent.py`: Core agent implementation with Hunter.io API integration
- `leadgen_agent_endpoint.py`: FastAPI endpoint for the agent
- `requirements.txt`: Project dependencies
- `Dockerfile`: Container configuration
- `.env`: Environment variables

## Configuration

### LLM Models

The agent uses OpenRouter as the API endpoint, supporting various models:
```env
LLM_MODEL=deepseek/deepseek-chat  # Default model
```

### API Keys

- **Hunter.io API Key**: Get your API key from [Hunter.io](https://hunter.io/api-keys)
- **OpenRouter API Key**: Get your API key from [OpenRouter](https://openrouter.ai/)
- **Supabase Keys**: Get your project URL and service key from [Supabase](https://supabase.com/dashboard)

## Limitations

- Hunter.io API rate limits apply
- Some email addresses might be protected or hidden
- Accuracy depends on Hunter.io's database
- Some domains might block email harvesting

## Contributing

This agent is part of the oTTomator agents collection. For contributions or issues, please refer to the main repository guidelines.
