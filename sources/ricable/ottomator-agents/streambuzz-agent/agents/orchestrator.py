from typing import List

from pydantic_ai.messages import ModelRequest, ModelResponse

from constants.constants import (START_STREAM_APPEND)
from constants.enums import StreamerIntentEnum
from exceptions.user_error import UserError
from models.agent_models import AgentRequest
from utils import intent_util
from utils.rag_util import create_knowledge_base
from .buzz_master import buzz_master_agent
from .responder import responder_agent
from .stream_starter import stream_starter_agent


async def get_response(
    request: AgentRequest,
    human_messages: list[str],
    messages: List[ModelRequest | ModelResponse],
) -> str:
    """Orchestrates the response generation process based on the user's intent.

    This function acts as a central dispatcher, receiving a user request and
    determining the appropriate agent to handle it. It first processes any
    provided files using RAG (Retrieval Augmented Generation) if available.
    Then, it classifies the user's intent using a dedicated utility. Based on
    the identified intent, it calls a specific agent to generate a response.

    Args:
        request: An AgentRequest object containing the user's query, session ID, and
            any associated files.
        human_messages: A list of strings representing the history of human
            messages in the current conversation.
        messages: A list of ModelRequest or ModelResponse objects representing
            the history of model messages in the current conversation.

    Returns:
        A string containing the generated response from the selected agent.

    Raises:
        UserError: If a user-related error occurs during processing, such as
            invalid input or a problem with user-specific data.
        Exception: If any other unexpected error occurs during the execution of
            this function, such as network issues or unexpected model responses.
    """
    try:
        # Make file RAG ready
        if request.files:
            await create_knowledge_base(request)

        # Get streamer's buzz_type
        streamer_intent: StreamerIntentEnum = (
            await intent_util.classify_streamer_intent(
                messages=human_messages, query=request.query
            )
        )
        print(f"{request.query=}>> {streamer_intent.name=}")

        # Perform task based on streamer's buzz_type
        if streamer_intent == StreamerIntentEnum.START_STREAM:
            agent_result = await stream_starter_agent.run(
                user_prompt=request.query, deps=request.session_id, result_type=str
            )
            response = agent_result.data + START_STREAM_APPEND
        elif streamer_intent == StreamerIntentEnum.GET_CURRENT_CHAT:
            agent_result = await buzz_master_agent.run(
                user_prompt="Get current buzz.",
                deps=request.session_id,
                result_type=str,
            )
            response = agent_result.data
        elif streamer_intent == StreamerIntentEnum.GET_NEXT_CHAT:
            agent_result = await buzz_master_agent.run(
                user_prompt="Get next buzz.", deps=request.session_id, result_type=str
            )
            response = agent_result.data
        elif streamer_intent == StreamerIntentEnum.REPLY_CHAT:
            agent_result = await buzz_master_agent.run(
                user_prompt=f"Extract and store reply from this message:\n{request.query}",
                deps=request.session_id,
                result_type=str,
            )
            response = agent_result.data
        else:
            agent_result = await responder_agent.run(
                user_prompt=request.query,
                deps=request.session_id,
                result_type=str,
                message_history=messages,
            )
            response = agent_result.data

        return response
    except UserError as ue:
        print(f"Error>> get_response: {str(ue)}")
        raise
    except Exception as e:
        print(f"Error>> get_response: {str(e)}")
        raise
