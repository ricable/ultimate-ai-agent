"""
Environment configuration management for the MCP server.
"""

import os
from dataclasses import dataclass
from urllib.parse import urlparse


class ConfigurationError(Exception):
    """Raised when there's an error in configuration."""

    pass


@dataclass
class EnvironmentConfig:
    """Configuration loaded from environment variables."""

    supabase_url: str
    supabase_service_key: str
    port: int  # Required - no default
    openai_api_key: str | None = None
    host: str = "0.0.0.0"
    transport: str = "sse"


@dataclass
class RAGStrategyConfig:
    """Configuration for RAG strategies."""

    use_contextual_embeddings: bool = False
    use_hybrid_search: bool = True
    use_agentic_rag: bool = True
    use_reranking: bool = True


def validate_openai_api_key(api_key: str) -> bool:
    """Validate OpenAI API key format."""
    if not api_key:
        raise ConfigurationError("OpenAI API key cannot be empty")

    if not api_key.startswith("sk-"):
        raise ConfigurationError("OpenAI API key must start with 'sk-'")

    return True


def validate_supabase_url(url: str) -> bool:
    """Validate Supabase URL format."""
    if not url:
        raise ConfigurationError("Supabase URL cannot be empty")

    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ConfigurationError("Supabase URL must use HTTPS")

    if not parsed.netloc:
        raise ConfigurationError("Invalid Supabase URL format")

    return True


def load_environment_config() -> EnvironmentConfig:
    """Load and validate environment configuration."""
    # OpenAI API key is optional at startup - can be set via API
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Required environment variables for database access
    supabase_url = os.getenv("SUPABASE_URL")
    if not supabase_url:
        raise ConfigurationError("SUPABASE_URL environment variable is required")

    supabase_service_key = os.getenv("SUPABASE_SERVICE_KEY")
    if not supabase_service_key:
        raise ConfigurationError("SUPABASE_SERVICE_KEY environment variable is required")

    # Validate required fields
    if openai_api_key:
        validate_openai_api_key(openai_api_key)
    validate_supabase_url(supabase_url)

    # Optional environment variables with defaults
    host = os.getenv("HOST", "0.0.0.0")
    port_str = os.getenv("PORT")
    if not port_str:
        # This appears to be for MCP configuration based on default 8051
        port_str = os.getenv("ARCHON_MCP_PORT")
        if not port_str:
            raise ConfigurationError(
                "PORT or ARCHON_MCP_PORT environment variable is required. "
                "Please set it in your .env file or environment. "
                "Default value: 8051"
            )
    transport = os.getenv("TRANSPORT", "sse")

    # Validate and convert port
    try:
        port = int(port_str)
    except ValueError:
        raise ConfigurationError(f"PORT must be a valid integer, got: {port_str}")

    return EnvironmentConfig(
        openai_api_key=openai_api_key,
        supabase_url=supabase_url,
        supabase_service_key=supabase_service_key,
        host=host,
        port=port,
        transport=transport,
    )


def get_rag_strategy_config() -> RAGStrategyConfig:
    """Load RAG strategy configuration from environment variables."""

    def str_to_bool(value: str | None) -> bool:
        """Convert string environment variable to boolean."""
        if value is None:
            return False
        return value.lower() in ("true", "1", "yes", "on")

    return RAGStrategyConfig(
        use_contextual_embeddings=str_to_bool(os.getenv("USE_CONTEXTUAL_EMBEDDINGS")),
        use_hybrid_search=str_to_bool(os.getenv("USE_HYBRID_SEARCH")),
        use_agentic_rag=str_to_bool(os.getenv("USE_AGENTIC_RAG")),
        use_reranking=str_to_bool(os.getenv("USE_RERANKING")),
    )
