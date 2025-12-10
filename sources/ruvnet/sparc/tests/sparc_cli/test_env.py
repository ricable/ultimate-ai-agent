import pytest
from sparc_cli.env import validate_environment

def test_validate_environment():
    """Test environment validation."""
    class Args:
        provider = "anthropic"
        expert_provider = "openai"
    
    expert_enabled, missing = validate_environment(Args())
    assert isinstance(expert_enabled, bool)
    assert isinstance(missing, list)

def test_validate_environment_missing_keys():
    """Test environment validation with missing keys."""
    class Args:
        provider = "invalid"
        expert_provider = "invalid"
    
    expert_enabled, missing = validate_environment(Args())
    assert expert_enabled is False
    assert len(missing) > 0

def test_validate_environment_openai_compatible():
    """Test environment validation for openai-compatible provider."""
    class Args:
        provider = "openai-compatible"
        expert_provider = "openai"
    
    expert_enabled, missing = validate_environment(Args())
    assert isinstance(expert_enabled, bool)
    assert isinstance(missing, list)

def test_validate_environment_expert_fallback():
    """Test expert provider fallback to base keys."""
    class Args:
        provider = "anthropic"
        expert_provider = "anthropic"
    
    expert_enabled, missing = validate_environment(Args())
    assert isinstance(expert_enabled, bool)
    assert isinstance(missing, list)
