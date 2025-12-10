# Graphiti Agent Demo

Here we demonstrate the power of Graphiti, a temporal knowledge graph solution that enables AI agents to maintain and query evolving knowledge over time. The implementation showcases how to use Graphiti with Pydantic AI to build intelligent agents that can reason about changing facts.

## Overview

This demo includes three main components:

1. **Quickstart Example (`quickstart.py`)**: A comprehensive tutorial demonstrating Graphiti's core features.
2. **Agent Interface (`agent.py`)**: A conversational agent powered by Pydantic AI that can search and query the Graphiti knowledge graph.
3. **LLM Evolution Demo (`llm_evolution.py`)**: A simulation showing how knowledge evolves over time, with three phases of LLM development that update the knowledge graph.

## Prerequisites

- Python 3.10 or higher
- Neo4j 5.26 or higher (for storing the knowledge graph)
- OpenAI API key (for LLM inference and embedding)

## Installation

### 1. Set up a virtual environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up Neo4j

You have a couple easy options for setting up Neo4j:

#### Option A: Using Local-AI-Packaged (Simplified setup)
1. Clone the repository: `git clone https://github.com/coleam00/local-ai-packaged`
2. Follow the installation instructions to set up Neo4j through the package
3. Note the username and password you set in .env and the URI will be bolt://localhost:7687

#### Option B: Using Neo4j Desktop
1. Download and install [Neo4j Desktop](https://neo4j.com/download/)
2. Create a new project and add a local DBMS
3. Start the DBMS and set a password
4. Note the connection details (URI, username, password)

### 4. Configure environment variables

Create a `.env` file in the project root with the following variables:

```
# Neo4j Connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# OpenAI API
OPENAI_API_KEY=your_openai_api_key
MODEL_CHOICE=gpt-4.1-mini  # Or another OpenAI model
```

## Running the Demo

### 1. Run the Quickstart Example

To get familiar with Graphiti's core features:

```bash
python quickstart.py
```

This will demonstrate:
- Adding episodes to the knowledge graph
- Performing basic searches
- Using center node search for context-aware results
- Utilizing search recipes for node retrieval

### 2. Experience the Power of Temporal Knowledge

To see how knowledge evolves over time, run the LLM evolution demo in one terminal:

```bash
python llm_evolution.py
```

**⚠️ WARNING: Running this script will clear all existing data in your Neo4j database!**

This interactive demo will:
1. Add information about current top LLMs (Gemini, Claude, GPT-4.1)
2. Update the knowledge graph when Claude 4 emerges as the best LLM
3. Update again when MLMs make traditional LLMs obsolete

The script will pause between phases, allowing you to interact with the agent to see how its knowledge changes.

### 3. Interact with the Agent

In a separate terminal, run the agent interface:

```bash
python agent.py
```

This will start a conversational interface where you can:
1. Ask questions about LLMs
2. See the agent retrieve information from the knowledge graph
3. Experience how the agent's responses change as the knowledge graph evolves

## Demo Workflow

For the best demonstration experience:

1. Start with a fresh Neo4j database
2. In Terminal 1: Run `python llm_evolution.py` and complete Phase 1
3. In Terminal 2: Run `python agent.py` and ask "Which is the best LLM?"
4. In Terminal 1: Continue to Phase 2 by typing "continue"
5. In Terminal 2: Ask the same question again to see the updated knowledge
6. In Terminal 1: Continue to Phase 3
7. In Terminal 2: Ask "Are LLMs still relevant?" to see the final evolution

This workflow demonstrates how Graphiti maintains temporal knowledge and how the agent's responses adapt to the changing knowledge graph.

## Key Features

- **Temporal Knowledge**: Graphiti tracks when facts become valid and invalid
- **Hybrid Search**: Combines semantic similarity and BM25 text retrieval
- **Context-Aware Queries**: Reranks results based on graph distance
- **Structured Data Support**: Works with both text and JSON episodes
- **Easy Integration**: Seamlessly works with Pydantic AI for agent development

## Project Structure

- `agent.py`: Pydantic AI agent with Graphiti search capabilities
- `quickstart.py`: Tutorial demonstrating core Graphiti features
- `llm_evolution.py`: Demo showing how knowledge evolves over time
- `requirements.txt`: Project dependencies
- `.env`: Configuration for API keys and Neo4j connection

## Additional Resources

- [Graphiti Documentation](https://help.getzep.com/graphiti/graphiti/overview)
- [Pydantic AI Documentation](https://ai.pydantic.dev/)
- [Neo4j Documentation](https://neo4j.com/docs/)

## License

This project includes code from Zep Software, Inc. under the Apache License 2.0.
