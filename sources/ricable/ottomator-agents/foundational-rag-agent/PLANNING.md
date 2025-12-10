# Project Planning: RAG AI Agent with Pydantic AI and Supabase

## Project Overview
We're building a simple Retrieval-Augmented Generation (RAG) AI agent using the Pydantic AI library. The agent will have access to a knowledge base stored in Supabase, allowing it to retrieve relevant information to answer user queries. The system will include functionality to ingest local text and PDF files, process them, and store them in Supabase for later retrieval.

## Architecture

### Core Components:
1. **Document Ingestion Pipeline**
   - Accept local files (TXT, PDF)
   - Simple text processing and chunking (without external libraries)
   - Generate embeddings using OpenAI embeddings API
   - Store documents and embeddings in Supabase

2. **Supabase Database**
   - Store document chunks and their embeddings with pgvector
   - Support semantic search for efficient retrieval
   - Tables will be created and managed via Supabase MCP server

3. **Pydantic AI Agent**
   - Define a tool to query the knowledge base
   - Use OpenAI models for generating responses
   - Integrate knowledge base search results into responses

4. **Streamlit User Interface**
   - Interface for uploading documents
   - Interface for querying the AI agent
   - Display agent responses

### Technology Stack:
- **Language**: Python 3.11+
- **AI Framework**: Pydantic AI for agent implementation
- **Database**: Supabase with pgvector extension
- **Embeddings**: OpenAI embeddings API
- **LLM Provider**: OpenAI (GPT-4.1 mini or similar)
- **UI**: Streamlit
- **Document Processing**: Simple text processing with PyPDF2 for PDF extraction

## Development Process

The development will follow a task-based approach where each component will be implemented sequentially. We should:

1. Start by setting up the project structure
2. Create database tables using Supabase MCP server
3. Implement simple document ingestion pipeline
4. Create the Pydantic AI agent with knowledge base search tool
5. Develop Streamlit UI
6. Connect all components and ensure they work together
7. Test the complete system

## Design Principles

1. **Modularity**: Keep components decoupled for easier maintenance
2. **Simplicity**: Focus on making the system easy to understand and modify
3. **Performance**: Optimize for response time in knowledge retrieval
4. **User Experience**: Make the Streamlit interface intuitive

## Environment Configuration

Create a `.env.example` file with the following variables:
- `OPENAI_API_KEY`: For embeddings and LLM
- `OPENAI_MODEL`: e.g., "gpt-4.1-mini" or other models
- `SUPABASE_URL`: URL for Supabase instance
- `SUPABASE_KEY`: API key for Supabase

This file will serve as a template for users to create their own `.env` file.

## Expected Output

A functional RAG system where users can:
- Upload local text or PDF documents to build a knowledge base
- Ask questions to the AI agent
- Receive responses that incorporate information from the knowledge base

## Notes

When implementing this project, make sure to:
- Mark tasks complete in the task.md file as you finish them
- Use the Supabase MCP server to create and manage database tables
- Build a simple document ingestion pipeline without complex libraries
- Focus on creating a working Pydantic AI agent that can effectively retrieve and use information from the knowledge base
- Create a clean, intuitive Streamlit interface