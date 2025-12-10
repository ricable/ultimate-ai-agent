# RAG AI Agent with Pydantic AI and Supabase

A simple Retrieval-Augmented Generation (RAG) AI agent using Pydantic AI and Supabase with pgvector for document storage and retrieval.

## Features

- Document ingestion pipeline for TXT and PDF files
- Vector embeddings using OpenAI
- Document storage in Supabase with pgvector
- Pydantic AI agent with knowledge base search capabilities
- Streamlit UI for document uploads and agent interaction

## Project Structure

```
foundational-rag-agent/
├── database/
│   └── setup.py          # Database setup and connection utilities
├── document_processing/
│   ├── __init__.py
│   ├── chunker.py        # Text chunking functionality
│   ├── embeddings.py     # Embeddings generation with OpenAI
│   ├── ingestion.py      # Document ingestion pipeline
│   └── processors.py     # TXT and PDF processing
├── agent/
│   ├── __init__.py
│   ├── agent.py          # Main agent definition
│   ├── prompts.py        # System prompts
│   └── tools.py          # Knowledge base search tool
├── ui/
│   └── app.py            # Streamlit application
├── tests/
│   ├── test_chunker.py
│   ├── test_embeddings.py
│   ├── test_ingestion.py
│   ├── test_processors.py
│   └── test_agent.py
├── .env.example          # Example environment variables
├── requirements.txt      # Project dependencies
├── PLANNING.md           # Project planning document
├── TASK.md               # Task tracking
└── README.md             # Project documentation
```

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in your API keys and configuration
5. Run the Streamlit application:
   ```
   streamlit run ui/app.py
   ```
6. Run the SQL in `rag-example.sql` to create the table and matching function for RAG

## Usage

1. Upload documents (TXT or PDF) through the Streamlit UI
2. Ask questions to the AI agent
3. View responses with source attribution

## Dependencies

- Python 3.11+
- Pydantic AI
- Supabase
- OpenAI
- PyPDF2
- Streamlit

## License

MIT
