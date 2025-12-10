# Travel Planning Agent System

A demonstration of the parallel agent architecture using Pydantic AI and LangGraph. This project implements a multi-agent travel planning system that helps users plan their perfect trip through an interactive Streamlit UI.

![Travel Agent Graph](extras/TravelAgentGraph.png)

## Overview

This project implements a sophisticated travel planning system that uses multiple specialized AI agents working in parallel to create comprehensive travel plans. The system collects user preferences and travel details through a conversational interface, then simultaneously processes flight, hotel, and activity recommendations before combining them into a final travel plan.

All tool calls for the agents are mocked, so this isn't using real data! This is simply built as an example, focusing on the agent architecture instead of the tooling.

## Features

- **Interactive Streamlit UI** with real-time streaming responses
- **Multi-agent architecture** with specialized agents for different aspects of travel planning
- **Parallel processing** of recommendations for improved efficiency
- **User preference management** for airlines, hotel amenities, and budget levels
- **Conversational interface** for gathering travel details
- **Comprehensive travel plans** with flights, accommodations, and activities

## Architecture

The system consists of five specialized agents:

1. **Info Gathering Agent**: Collects necessary travel details (destination, origin, dates, budget)
2. **Flight Agent**: Recommends flights based on travel details and user preferences
3. **Hotel Agent**: Suggests accommodations based on destination, dates, budget, and amenity preferences
4. **Activity Agent**: Recommends activities based on destination, dates, and weather forecasts
5. **Final Planner Agent**: Aggregates all recommendations into a comprehensive travel plan

These agents are orchestrated through a LangGraph workflow that enables parallel execution and dynamic routing based on the completeness of gathered information.

## Technical Stack

- **Pydantic AI**: For structured agent definitions and type validation
- **LangGraph**: For orchestrating the agent workflow and parallel execution
- **Streamlit**: For building the interactive user interface

## Setup Instructions

### Prerequisites

- Python 3.11 or higher
- OpenAI or OpenRouter API key (can use Ollama too)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/travel-planning-agent.git
   cd travel-planning-agent
   ```

2. Set up a virtual environment:

   **Windows**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   **macOS/Linux**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory and follow the instructions given in `.env.example`:
   ```
   PROVIDER=
   BASE_URL=
   LLM_API_KEY=
   MODEL_CHOICE=
   ```

### Running the Application

1. Start the Streamlit UI:
   ```bash
   streamlit run streamlit_ui.py
   ```

2. Open your browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

## Usage

1. **Set Your Preferences**: Use the sidebar to set your preferred airlines, hotel amenities, and budget level.

2. **Start a Conversation**: Type your travel request in the chat input. For example:
   ```
   I want to go to Tokyo from Minneapolis. Jun 1st, returning on 6th. Max price for hotel is $300 per night.
   ```

3. **Interact with the Agent**: The system will ask follow-up questions if any details are missing.

4. **Review Your Plan**: Once all details are collected, the system will generate a comprehensive travel plan with flight, hotel, and activity recommendations.

Note that with this demonstration, once the final plan is given to you, the conversation is over. This can of course be extended to allow for editing the trip, asking more questions, etc.

## Project Structure

```
travel-planning-agent/
├── agents/                      # Individual agent definitions
│   ├── activity_agent.py        # Agent for recommending activities
│   ├── final_planner_agent.py   # Agent for creating the final travel plan
│   ├── flight_agent.py          # Agent for flight recommendations
│   ├── hotel_agent.py           # Agent for hotel recommendations
│   └── info_gathering_agent.py  # Agent for collecting travel details
├── agent_graph.py               # LangGraph workflow definition
├── streamlit_ui.py              # Streamlit user interface
├── utils.py                     # Utility functions
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## How It Works

1. The system starts by gathering all necessary information from the user through the Info Gathering Agent.
2. Once all required details are collected, the system simultaneously calls the Flight, Hotel, and Activity agents to get recommendations.
3. Each specialized agent uses its tools to search for and recommend options based on the user's preferences.
4. After all recommendations are collected, the Final Planner Agent creates a comprehensive travel plan.
5. The entire process is streamed in real-time to the user through the Streamlit UI.

## Inspired by Anthropic's Agent Architecture

This project is a demonstration of the parallelization workflow showcased in [Anthropic's Agent Architecture blog](https://www.anthropic.com/engineering/building-effective-agents). The implementation follows a similar pattern where multiple specialized agents work in parallel to solve different aspects of a complex task.

![Anthropic Parallelization Workflow](extras/AnthropicParallelizationWorkflow.png)

The key architectural pattern demonstrated here is the ability to:
1. Gather initial information
2. Fan out to multiple specialized agents working in parallel
3. Aggregate results into a final, comprehensive response

This approach significantly improves efficiency compared to sequential processing, especially for complex tasks with independent subtasks.

## Customization

You can customize the system by:

- Modifying agent prompts in the respective agent files
- Adding new specialized agents for additional travel aspects
- Enhancing the tools with real API integrations for flights, hotels, and activities
- Extending the user preference system with additional options

## License

This project is licensed under the MIT License.

## Acknowledgments

- Built with [Pydantic AI](https://github.com/pydantic/pydantic-ai)
- Powered by [LangGraph](https://github.com/langchain-ai/langgraph)
- UI created with [Streamlit](https://streamlit.io/)
