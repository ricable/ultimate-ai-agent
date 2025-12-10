import pytest
from pathlib import Path
from src.agentic_security.cache import SecurityCache
from src.agentic_security.prompts import PromptManager
from src.agentic_security.progress import ProgressReporter

def test_security_cache(tmp_path):
    """Test cache functionality with security controls"""
    # Validate tmp_path is safe
    safe_path = Path(tmp_path).resolve()
    assert safe_path.is_relative_to(tmp_path), "Path traversal detected"
    
    cache = SecurityCache(str(safe_path))
    
    try:
        # Test with size-limited data
        results = {"test": "data" * 100}  # Limit test data size
        assert len(str(results)) < 1024 * 1024, "Test data too large"
        
        # Test saving results with sanitized scan ID
        scan_id = "test_scan_123"
        assert scan_id.isalnum(), "Scan ID must be alphanumeric"
        cache.save_scan_results(scan_id, results)
        
        # Test retrieving results
        cached = cache.get_scan_results(scan_id)
        assert cached == results
        
        # Test clearing old results
        cache.clear_old_results(days=0)
        assert cache.get_scan_results(scan_id) is None
    finally:
        # Cleanup sensitive test data
        if safe_path.exists():
            for f in safe_path.glob("**/*"):
                if f.is_file():
                    f.write_bytes(b'\x00' * f.stat().st_size)
                    f.unlink()

def test_prompt_manager():
    """Test prompt management with security controls"""
    # Validate custom prompt content
    custom_prompts = {
        'custom_review': "Custom review prompt"
    }
    
    for key, value in custom_prompts.items():
        assert len(value) < 10000, "Prompt too long"
        assert all(ord(c) < 128 for c in value), "Non-ASCII characters in prompt"
        assert not any(c in value for c in ['<', '>', '{%', '%}']), "Potential template injection"
    
    manager = PromptManager(custom_prompts)
    
    # Test default prompt with validation
    arch_prompt = manager.get_prompt('architecture_review')
    assert "Review the architecture" in arch_prompt
    assert len(arch_prompt) < 10000, "Default prompt too long"
    
    # Test custom prompt
    custom_prompt = manager.get_prompt('custom_review')
    assert custom_prompt == "Custom review prompt"
    
    # Test formatted prompt with sanitized inputs
    safe_vuln_type = "SQL Injection".replace(';', '').replace('--', '')
    safe_file_path = Path("test.py").name  # Extract filename only
    
    fix_prompt = manager.get_prompt('fix_generation',
                                  vulnerability_type=safe_vuln_type,
                                  file_path=safe_file_path)
    assert safe_vuln_type in fix_prompt
    assert safe_file_path in fix_prompt

def test_progress_reporter(capsys):
    """Test progress reporting with security controls"""
    # Validate input parameters
    total_steps = 100
    assert 0 < total_steps <= 1000000, "Invalid step count"
    
    reporter = ProgressReporter(total_steps=total_steps)
    
    try:
        # Test progress updates with sanitized messages
        def validate_message(msg):
            assert len(msg) < 1000, "Message too long"
            assert all(ord(c) < 128 for c in msg), "Non-ASCII characters in message"
            assert '\n' not in msg, "Newline in message"
            return msg
        
        reporter.start(validate_message("Starting"))
        captured = capsys.readouterr()
        assert "Starting" in captured.out
        assert "0%" in captured.out
        
        # Validate step number
        step = 50
        assert 0 <= step <= total_steps, "Invalid step number"
        reporter.update(step, validate_message("Halfway"))
        captured = capsys.readouterr()
        assert "Halfway" in captured.out
        assert "50%" in captured.out
        
        reporter.finish(validate_message("Done"))
        captured = capsys.readouterr()
        assert "Done" in captured.out
        assert "100%" in captured.out
    except Exception as e:
        reporter.finish(f"Error: {str(e)[:100]}")  # Limit error message length
        raise
