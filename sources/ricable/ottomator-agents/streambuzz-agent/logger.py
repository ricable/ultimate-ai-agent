import logging

# Configure logging
logging.basicConfig(
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    level=logging.INFO
)

logger = logging.getLogger(__name__)

from functools import wraps


def log_method(func):
    """Decorator to log entry, exit, and errors for functions."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        logger.info(f"Entering: {func.__name__}()")
        try:
            result = await func(*args, **kwargs)
            logger.info(f"Exiting: {func.__name__}()")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}(): {e}", exc_info=True)
            raise

    return wrapper
