from constants.constants import MODEL_RETRIES, PYDANTIC_AI_MODEL
from constants.prompts import RESPONDER_SYSTEM_PROMPT
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings
from utils import supabase_util
from utils.rag_util import get_embedding

# Create Agent Instance with System Prompt and Result Type
responder_agent = Agent(
    model=PYDANTIC_AI_MODEL,
    name="responder_agent",
    end_strategy="early",
    model_settings=ModelSettings(temperature=0.5),
    system_prompt=RESPONDER_SYSTEM_PROMPT,
    result_type=str,
    result_tool_name="respond",
    result_tool_description="respond queries using RAG. If no sufficient "
    "information is found in knowledge base then "
    "default to LLM to generate the response",
    result_retries=MODEL_RETRIES,
)


@responder_agent.tool
async def respond(ctx: RunContext[str], user_query: str) -> str | None:
    """Responds to a user query using Retrieval Augmented Generation (RAG).

    This tool leverages a knowledge base stored in Supabase to provide contextually
    relevant answers to user queries. It first checks if a knowledge base exists for
    the given session ID. If found, it retrieves relevant document chunks by
    embedding the user query and comparing it to pre-computed embeddings of the
    document chunks. The retrieved chunks are then formatted and returned as a
    single string. If no knowledge base is found, or if no relevant chunks are
    retrieved, the function returns None, signaling the agent to fall back to
    the LLM for a response.

    Args:
        ctx: The context of the current agent run. This includes dependencies
            passed to the agent, such as the session ID, accessible via `ctx.deps`.
        user_query: The user's question or query string.

    Returns:
        A string containing formatted document chunks relevant to the user query,
        or None if no relevant information is found or an error occurs. The chunks
        are formatted with a header containing the document title, followed by the
        document content, with a separator ("\\n---\\n") between chunks. If an error
        occurs during retrieval, a string containing the error message is returned.

    Raises:
        Exception: If an error occurs during the retrieval process, such as issues
            accessing the database or embedding the query. The error message is
            logged to the console and returned as a string.
    """
    try:
        # Check if session_id has knowledge base
        file_name: str = await supabase_util.get_kb_file_name(session_id=ctx.deps)
        if not file_name:
            return None

        # Get the embedding for the query
        query_embedding = await get_embedding(user_query)

        # Query Supabase for relevant documents
        result = await supabase_util.get_matching_chunks(
            query_embedding=query_embedding, session_id=ctx.deps
        )

        if not result:
            return None

        # Format the results
        formatted_chunks = []
        for doc in result:
            chunk_text = f"""# {doc['title']}\n{doc['content']}"""
            formatted_chunks.append(chunk_text)

        # Join all chunks with a separator
        return "\n---\n".join(formatted_chunks)

    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"
