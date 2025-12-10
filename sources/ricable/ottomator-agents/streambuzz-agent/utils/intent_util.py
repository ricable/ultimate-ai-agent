import re
from typing import List

from constants.constants import (CONFIDENCE_THRESHOLD, STREAMER_INTENT_EXAMPLES,
                                 YOUTUBE_URL_REGEX)
from constants.enums import StreamerIntentEnum
from logger import log_method


def contains_valid_youtube_url(user_query: str) -> bool:
    """Checks if a given string contains a valid YouTube URL.

    This function uses a regular expression to determine if the input string
    matches the expected format of a YouTube URL.

    Args:
        user_query: The string to check for a YouTube URL.

    Returns:
        True if the string contains a valid YouTube URL, False otherwise.
    """
    return re.search(YOUTUBE_URL_REGEX, user_query) is not None


def _simple_intent_match(text: str) -> tuple[StreamerIntentEnum, float]:
    """Simple intent matching using string matching.
    
    Args:
        text: Input text to classify
        
    Returns:
        Tuple of (intent, confidence score)
    """
    text = text.lower()
    max_score = 0
    best_intent = StreamerIntentEnum.UNKNOWN
    
    for intent_str, examples in STREAMER_INTENT_EXAMPLES.items():
        for example in examples:
            # Simple word overlap score
            example_words = set(example.lower().split())
            text_words = set(text.split())
            overlap = len(example_words.intersection(text_words))
            total = len(example_words.union(text_words))
            score = overlap / total if total > 0 else 0
            
            if score > max_score:
                max_score = score
                best_intent = StreamerIntentEnum[intent_str]
    
    return best_intent, max_score


@log_method
async def classify_streamer_intent(
    messages: List[str], query: str
) -> StreamerIntentEnum:
    """Classifies the intent of a streamer based on previous messages and the latest query.

    This function uses simple string matching to determine the streamer's intent.
    It gives more weight to the latest query compared to previous messages.

    Args:
        messages: List of previous messages in the conversation
        query: The latest query to classify

    Returns:
        The classified StreamerIntentEnum
    """
    # Give more weight to the latest query
    query_intent, query_score = _simple_intent_match(query)
    
    # If query score is high enough, use it directly
    if query_score >= CONFIDENCE_THRESHOLD:
        # Special case for START_STREAM - must have YouTube URL
        if query_intent == StreamerIntentEnum.START_STREAM and not contains_valid_youtube_url(query):
            return StreamerIntentEnum.UNKNOWN
        return query_intent
    
    # Otherwise consider previous messages
    if messages:
        prev_intents = [_simple_intent_match(msg)[0] for msg in messages[-3:]]
        # If there's a consistent intent in previous messages, use it
        if len(set(prev_intents)) == 1 and prev_intents[0] != StreamerIntentEnum.UNKNOWN:
            return prev_intents[0]
    
    return StreamerIntentEnum.UNKNOWN
