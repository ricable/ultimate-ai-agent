import asyncio
import base64
import json
import os
import re
from typing import Any, Dict, List, Tuple

import google.generativeai as genai
from dotenv import load_dotenv

from agents.buzz_intern import buzz_intern_agent
from constants.constants import (ACCEPTED_FILE_EXTENSION, ACCEPTED_FILE_MIME,
                                 ACCEPTED_FILE_QUANTITY, CHUNK_SIZE,
                                 EMBEDDING_DIMENSIONS, EMBEDDING_MODEL_NAME,
                                 MAX_FILE_SIZE_B, MAX_FILE_SIZE_MB, SUMMARY, TITLE)
from constants.prompts import TITLE_SUMMARY_PROMPT
from exceptions.user_error import UserError
from logger import log_method
from models.agent_models import AgentRequest, ProcessedChunk
from utils import supabase_util

load_dotenv()


@log_method
async def get_title_and_summary(chunk: str) -> dict:
    """Extracts the title and summary from a text chunk using a buzz_intern_agent.

    This function uses buzz_intern_agent to process the input chunk and extract
    the title and summary. It uses the `TITLE_SUMMARY_PROMPT` to guide the agent.

    Args:
        chunk: The input text string from which to extract the title and summary.

    Returns:
        A dictionary containing the extracted title and summary. If there's an
        error during extraction, the dictionary will contain default error messages
        for title and summary. The keys are 'title' and 'summary'.

    Raises:
        Any exception encountered during the extraction process is caught, printed to
        the console, and default error messages are returned.
    """
    try:
        completions = await buzz_intern_agent.run(
            user_prompt=f"{TITLE_SUMMARY_PROMPT}\n{chunk[:500]}",
            result_type=str
        )
        response_text = completions.data
        if response_text.startswith("```json"):
            response_text = re.sub(
                r"```json\s(.*?)\s*```", r"\1", response_text, flags=re.DOTALL
            )
        response: dict = json.loads(response_text)
        response[TITLE] = response.get(TITLE, "Error processing title")
        response[SUMMARY] = response.get(SUMMARY, "Error processing summary")
        return response
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {
            TITLE: "Error processing title",
            SUMMARY: "Error processing summary",
        }


@log_method
async def validate_file(files: List[Dict[str, Any]]) -> str:
    """Validates the uploaded file(s) against size, type, and quantity constraints.

    This function checks if the number of uploaded files, their extensions, MIME types,
    and sizes are within the allowed limits specified by constants.

    Args:
        files: A list of dictionaries, where each dictionary represents a file
            and contains keys like 'name', 'type', and 'base64'.

    Returns:
        The base64 encoded string of the file content if the file is valid.

    Raises:
        UserError: If any of the following conditions are met:
            - The number of files exceeds `ACCEPTED_FILE_QUANTITY`.
            - The file extension does not match `ACCEPTED_FILE_EXTENSION`.
            - The MIME type does not match `ACCEPTED_FILE_MIME`.
            - The decoded file size exceeds `MAX_FILE_SIZE_B`.
    """
    if len(files) != ACCEPTED_FILE_QUANTITY:
        raise UserError(
            f"Only {ACCEPTED_FILE_QUANTITY} file is allowed. You uploaded "
            f"{len(files)} files."
        )
    file = files[0]
    if not file["name"].endswith(ACCEPTED_FILE_EXTENSION):
        raise UserError(
            f"Only {ACCEPTED_FILE_QUANTITY} {ACCEPTED_FILE_EXTENSION} file is allowed."
        )
    if file["type"] != ACCEPTED_FILE_MIME:
        raise UserError(
            f"Only {ACCEPTED_FILE_QUANTITY} {ACCEPTED_FILE_MIME} file is allowed."
        )
    base64_content = file.get("base64")
    estimated_size = (len(base64_content) * 3) // 4 - base64_content.count("=")
    if estimated_size > MAX_FILE_SIZE_B:
        raise UserError(
            f"File exceeds the maximum allowed size of {MAX_FILE_SIZE_MB} MB."
        )
    return base64_content


@log_method
async def get_file_contents(files: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Retrieves and decodes the content of a file from its base64 representation.

    This function validates the file using `validate_file` and then decodes the
    base64 content to a UTF-8 string. It also checks if the decoded file size
    exceeds the allowed limit.

    Args:
        files: A list of dictionaries, where each dictionary represents a file
            and contains keys like 'name' and 'base64'.

    Returns:
        A tuple containing the file name and the decoded file content as a string.

    Raises:
        UserError: If any of the following conditions are met:
            - The file fails validation by `validate_file`.
            - There's an error decoding the base64 content.
            - The decoded file size exceeds `MAX_FILE_SIZE_B`.
    """
    base64_content = await validate_file(files)
    try:
        file_content = base64.b64decode(base64_content).decode("utf-8").strip()
    except Exception:
        raise UserError("Error decoding file content, invalid base64 encoding")
    if len(file_content) > MAX_FILE_SIZE_B:
        raise UserError(
            f"File exceeds the maximum allowed size of {MAX_FILE_SIZE_MB} MB."
        )
    return files[0]["name"], file_content


@log_method
async def get_embedding(text: str) -> List[float]:
    """Generates an embedding vector for a given text using the Gemini model.

    This function uses the `google.generativeai` library to generate an embedding
    vector for the provided text. It configures the API key from the environment
    and uses the specified embedding model.

    Args:
        text: The input text string for which the embedding is to be generated.

    Returns:
        A list of floats representing the embedding vector. Returns a zero vector
        of size `EMBEDDING_DIMENSIONS` if there's an error during embedding.

    Raises:
         Any exception encountered during the embedding process is caught, printed to the console, and a zero vector is returned.
    """
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        response = genai.embed_content(model=EMBEDDING_MODEL_NAME, content=text)
        return response["embedding"]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * EMBEDDING_DIMENSIONS  # Return zero vector on error


@log_method
async def process_chunk(
    chunk_number: int, session_id: str, file_name: str, chunk: str
) -> ProcessedChunk:
    """Processes a single chunk of text to extract metadata and generate embeddings.

    This function orchestrates the extraction of title and summary using
    `get_title_and_summary`, generates embeddings using `get_embedding`, and
    packages the results into a `ProcessedChunk` object.

    Args:
        chunk_number: The index of the chunk in the original document.
        session_id: The unique ID of the user session.
        file_name: The name of the file the chunk belongs to.
        chunk: The text content of the chunk.

    Returns:
        A `ProcessedChunk` object containing the session ID, file name, chunk
        number, extracted title, summary, content, and embedding vector.
    """
    # Get title and summary
    extracted = await get_title_and_summary(chunk)

    # Get embedding
    embedding = await get_embedding(chunk)

    return ProcessedChunk(
        session_id=session_id,
        file_name=file_name,
        chunk_number=chunk_number,
        title=extracted[TITLE],
        summary=extracted[SUMMARY],
        content=chunk,
        embedding=embedding,
    )


@log_method
async def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Splits a text into chunks, respecting code blocks, paragraphs, and sentences.

    This function splits a text into chunks of a specified size, prioritizing
    splits at code blocks (```), then paragraphs (\n\n), and finally sentences
    (. ). It aims to create semantically meaningful chunks.

    Args:
        text: The input text string to be split into chunks.
        chunk_size: The desired maximum size of each chunk. Defaults to `CHUNK_SIZE`.

    Returns:
        A list of strings, where each string is a chunk of the original text.
        Chunks are split at code blocks (```), paragraphs (\n\n), or sentences (. )
        when possible, prioritizing code blocks, then paragraphs, then sentences.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind("```")
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif "\n\n" in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind("\n\n")
            if (
                last_break > chunk_size * 0.3
            ):  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif ". " in chunk:
            # Find the last sentence break
            last_period = chunk.rfind(". ")
            if (
                last_period > chunk_size * 0.3
            ):  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks


@log_method
async def process_and_store_document(
    session_id: str, file_name: str, file_content: str
):
    """Processes a document by splitting it into chunks, extracting metadata, and storing it.

    This function orchestrates the processing of a document by first splitting it into
    chunks using `chunk_text`, then processes each chunk in parallel using
    `process_chunk`, and finally stores the processed chunks in a database
    in parallel using `supabase_util.insert_chunk`.

    Args:
        session_id: The unique ID of the user session.
        file_name: The name of the file being processed.
        file_content: The text content of the file.
    """
    # Split into chunks
    chunks = await chunk_text(file_content)

    # Process chunks in parallel
    tasks = [
        process_chunk(index, session_id, file_name, chunk)
        for index, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)

    # Store chunks in parallel
    insert_tasks = [supabase_util.insert_chunk(chunk) for chunk in processed_chunks]
    await asyncio.gather(*insert_tasks)


@log_method
async def create_knowledge_base(request: AgentRequest) -> None:
    """Creates a knowledge base from a user-uploaded document.

    This function handles the creation of a knowledge base from a document,
    including retrieving the file content, checking for and deleting any existing
    knowledge base associated with the session, processing the new document to
    create a new knowledge base, and storing a success message in the database.

    Args:
        request: An `AgentRequest` object containing the session ID and file
            information.
    """
    response_string = ""
    file_name, file_content = await get_file_contents(request.files)
    previous_file_name: str = await supabase_util.get_kb_file_name(request.session_id)
    if previous_file_name:
        await supabase_util.delete_previous_kb_entries(request.session_id)
        response_string += f"Discarding previous knowledge base {previous_file_name}.\n"
    await process_and_store_document(request.session_id, file_name, file_content)
    response_string += (
        f"Knowledge base built with {file_name} successfully and ready for use. "
        f"Analyzing a response to your query ..."
    )
    # Display this to user and use his query in unknown buzz_type
    await supabase_util.store_message(
        session_id=request.session_id,
        message_type="ai",
        content=response_string,
        data={"request_id": request.request_id},
    )
