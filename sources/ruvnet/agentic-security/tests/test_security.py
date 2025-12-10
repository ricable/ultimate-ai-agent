import pytest
import os
import yaml
import json
import time
import sys
import random
from pathlib import Path
from unittest.mock import Mock, patch
from src.agentic_security.security_pipeline import SecurityPipeline

# Cyberpunk styling
CYAN = '\033[0;36m'
MAGENTA = '\033[0;35m'
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[0;33m'
NC = '\033[0m'
BOLD = '\033[1m'
DIM = '\033[2m'

CYBER_BANNER = f"""
{CYAN}
    █████╗  ██████╗ ███████╗███╗   ██╗████████╗██╗ ██████╗
   ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██║██╔════╝
   ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ██║██║     
   ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ██║██║     
   ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ██║╚██████╗
   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝ ╚═════╝
{MAGENTA}   ████████╗███████╗███████╗████████╗███████╗
   ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██╔════╝
      ██║   █████╗  ███████╗   ██║   ███████╗
      ██║   ██╔══╝  ╚════██║   ██║   ╚════██║
      ██║   ███████╗███████║   ██║   ███████║
      ╚═╝   ╚══════╝╚══════╝   ╚═╝   ╚══════╝{NC}

{CYAN}[ Security Pipeline Test Suite ]{NC}
{MAGENTA}[ Initializing Cybersecurity Validation Sequence ]{NC}

{DIM}Executing comprehensive security validation protocol...{NC}
"""

def matrix_rain(duration=1.0):
    """Display Matrix-style rain animation"""
    chars = "ﾊﾐﾋｰｳｼﾅﾓﾆｻﾜﾂｵﾘｱﾎﾃﾏｹﾒｴｶｷﾑﾕﾗｾﾈｽﾀﾇﾍ"
    lines = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        # Generate new line
        if random.random() < 0.2:
            lines.append([0, random.randint(0, 79)])
        
        # Update and print lines
        sys.stdout.write("\033[2J\033[H")  # Clear screen
        for line in lines[:]:
            if line[0] < 24:  # Screen height
                sys.stdout.write(f"\033[{line[0]};{line[1]}H{CYAN}{random.choice(chars)}{NC}")
                line[0] += 1
            else:
                lines.remove(line)
        
        sys.stdout.flush()
        time.sleep(0.05)

def print_section(name):
    """Print cyberpunk-styled section header"""
    print(f"\n{CYAN}╔{'═' * 50}╗{NC}")
    print(f"{CYAN}║{NC} {BOLD}{name}{' ' * (49 - len(name))}{CYAN}║{NC}")
    print(f"{CYAN}╠{'═' * 50}╣{NC}")
    print(f"{CYAN}║{NC} {DIM}Initializing test sequence...{' ' * 25}{CYAN}║{NC}")
    print(f"{CYAN}╚{'═' * 50}╝{NC}\n")

def animate_loading(message, duration=0.5):
    """Show cyberpunk-styled loading animation"""
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    start_time = time.time()
    i = 0
    
    # Progress bar characters
    fill = "█"
    empty = "░"
    
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        bar_length = 20
        filled_length = int(bar_length * progress)
        
        # Create progress bar
        bar = (fill * filled_length + empty * (bar_length - filled_length))
        
        # Create loading message
        sys.stdout.write(f"\r{CYAN}[{frames[i % len(frames)]}]{NC} {message} {CYAN}|{NC} {bar} {CYAN}{int(progress * 100)}%{NC}")
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1
    
    sys.stdout.write(f"\r{GREEN}[✓]{NC} {message} {CYAN}|{NC} {fill * bar_length} {CYAN}100%{NC}\n")
    sys.stdout.flush()

# Print banner at start of tests
def pytest_configure(config):
    print(CYBER_BANNER)
    matrix_rain(1.0)  # Show Matrix rain animation for 1 second

# Test fixtures with animations
@pytest.fixture
def mock_config():
    animate_loading("Initializing mock configuration")
    return {
        'security': {
            'critical_threshold': 7.5,
            'max_fix_attempts': 3,
            'scan_targets': [
                {'type': 'web', 'url': 'http://localhost:8080'},
                {'type': 'code', 'path': './src'}
            ]
        },
        'notifications': {
            'enabled': True,
            'channels': [
                {'type': 'github'},
                {'type': 'slack', 'webhook': 'test_webhook'}
            ]
        },
        'aider': {
            'architect_mode': True,
            'model': 'o1-preview',
            'fix_mode': 'sonnet'
        }
    }

@pytest.fixture
def mock_env_vars(monkeypatch):
    animate_loading("Setting up environment variables")
    monkeypatch.setenv('OPENAI_API_KEY', 'test_openai_key')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test_anthropic_key')
    monkeypatch.setenv('SLACK_WEBHOOK', 'test_slack_webhook')

@pytest.fixture
def pipeline(mock_config, mock_env_vars, tmp_path):
    animate_loading("Initializing security pipeline")
    config_file = tmp_path / 'config.yml'
    with open(config_file, 'w') as f:
        yaml.dump(mock_config, f)
    return SecurityPipeline(str(config_file))

# Configuration Tests
def test_load_config(pipeline, mock_config):
    """Test configuration loading functionality"""
    print_section("Configuration Tests")
    animate_loading("Testing configuration loading")
    assert pipeline.config == mock_config
    assert pipeline.critical_threshold == mock_config['security']['critical_threshold']
    assert pipeline.max_fix_attempts == mock_config['security']['max_fix_attempts']

def test_load_config_missing_file():
    """Test handling of missing configuration file"""
    animate_loading("Testing missing configuration handling")
    with pytest.raises(FileNotFoundError):
        SecurityPipeline('nonexistent.yml')

# Environment Tests
def test_setup_environment(mock_env_vars):
    """Test environment setup functionality"""
    print_section("Environment Tests")
    animate_loading("Testing environment setup")
    pipeline = SecurityPipeline()
    pipeline.setup_environment()

def test_setup_environment_missing_vars(monkeypatch):
    """Test handling of missing environment variables"""
    animate_loading("Testing missing environment variables")
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with pytest.raises(EnvironmentError):
        pipeline = SecurityPipeline()
        pipeline.setup_environment()

# Security Scanning Tests
@patch('subprocess.run')
def test_run_architecture_review(mock_run, pipeline):
    """Test architecture review functionality"""
    print_section("Security Scanning Tests")
    animate_loading("Testing architecture review")
    mock_run.return_value.stdout = "- Fix XSS vulnerability\n- Update dependencies"
    result = pipeline.run_architecture_review()
    assert 'output' in result
    assert 'suggestions' in result
    assert len(result['suggestions']) == 2
    mock_run.assert_called_once()

@patch('subprocess.run')
def test_run_security_checks(mock_run, pipeline):
    """Test security scanning functionality"""
    animate_loading("Testing security checks")
    mock_run.return_value.stdout = "{}"
    results = pipeline.run_security_checks()
    assert 'web' in results
    assert 'code' in results

# Fix Implementation Tests
@patch('subprocess.run')
def test_implement_fixes(mock_run, pipeline):
    """Test fix implementation functionality"""
    print_section("Fix Implementation Tests")
    animate_loading("Testing fix implementation")
    mock_run.return_value.stdout = "Changes applied"
    suggestions = ["Fix XSS vulnerability", "Update dependencies"]
    assert pipeline.implement_fixes(suggestions)
    assert mock_run.call_count == len(suggestions)

@patch('subprocess.run')
def test_validate_fixes(mock_run, pipeline):
    """Test fix validation functionality"""
    animate_loading("Testing fix validation")
    mock_run.return_value.stdout = "{}"
    assert pipeline.validate_fixes()

# Git Integration Tests
@patch('subprocess.run')
def test_create_fix_branch(mock_run, pipeline):
    """Test branch creation functionality"""
    print_section("Git Integration Tests")
    animate_loading("Testing branch creation")
    assert pipeline.create_fix_branch()
    mock_run.assert_called_once()

@patch('subprocess.run')
def test_create_pull_request(mock_run, pipeline):
    """Test pull request creation functionality"""
    animate_loading("Testing pull request creation")
    mock_run.return_value.stdout = "test.py\nREADME.md"
    mock_run.return_value.stderr = ""
    assert pipeline.create_pull_request()
    assert mock_run.call_count >= 2

# Pipeline Integration Tests
@patch('subprocess.run')
def test_run_pipeline(mock_run, pipeline):
    """Test complete pipeline functionality"""
    print_section("Pipeline Integration Tests")
    animate_loading("Testing complete pipeline")
    mock_run.return_value.stdout = "{}"
    assert pipeline.run_pipeline()

# Result Processing Tests
def test_get_max_severity(pipeline):
    """Test severity calculation functionality"""
    print_section("Result Processing Tests")
    animate_loading("Testing severity calculation")
    test_cases = [
        ({'zap': {'alerts': [{'riskcode': '3'}, {'riskcode': '1'}]}}, 3.0),
        ({'nuclei': [{'severity': 'high'}, {'severity': 'low'}]}, 7.0),
        ({'dependency': {'vulnerabilities': [{'cvssScore': '8.5'}, {'cvssScore': '6.2'}]}}, 8.5),
    ]
    
    for result, expected in test_cases:
        assert abs(pipeline._get_max_severity(result) - expected) < 0.1

def test_parse_results(pipeline, tmp_path):
    """Test result parsing functionality"""
    print_section("Result Parsing Tests")
    animate_loading("Testing result parsing")
    
    # Test ZAP results
    report_file = tmp_path / 'zap-report.json'
    test_data = {'alerts': [{'riskcode': '3'}]}
    with open(report_file, 'w') as f:
        json.dump(test_data, f)
    result = pipeline._parse_zap_results(str(report_file))
    assert result == test_data
    
    # Test Nuclei results
    report_file = tmp_path / 'nuclei-report.jsonl'
    test_data = [{'severity': 'high'}, {'severity': 'low'}]
    with open(report_file, 'w') as f:
        for item in test_data:
            json.dump(item, f)
            f.write('\n')
    result = pipeline._parse_nuclei_results(str(report_file))
    assert len(result) == len(test_data)

def test_error_handling(pipeline):
    """Test error handling functionality"""
    print_section("Error Handling Tests")
    animate_loading("Testing error handling")
    result = pipeline._parse_zap_results('nonexistent.json')
    assert 'error' in result
    assert 'Failed to parse ZAP results' in result['error']

# Print completion banner
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    passed = len([i for i in terminalreporter.stats.get('passed', [])])
    failed = len([i for i in terminalreporter.stats.get('failed', [])])
    total = passed + failed
    
    # Calculate pass percentage
    pass_percentage = (passed / total) * 100 if total > 0 else 0
    
    # Create progress bar
    bar_length = 30
    filled_length = int(bar_length * pass_percentage / 100)
    bar = "█" * filled_length + "░" * (bar_length - filled_length)
    
    print(f"\n{CYAN}╔{'═' * 50}╗{NC}")
    print(f"{CYAN}║{NC} {BOLD}Test Execution Complete{' ' * 31}{CYAN}║{NC}")
    print(f"{CYAN}╠{'═' * 50}╣{NC}")
    print(f"{CYAN}║{NC} Progress: {bar} {pass_percentage:0.1f}%{' ' * (6 - len(f'{pass_percentage:0.1f}'))}{CYAN}║{NC}")
    print(f"{CYAN}║{NC} Total Tests: {total}{' ' * (38 - len(str(total)))}{CYAN}║{NC}")
    print(f"{CYAN}║{NC} Passed: {GREEN}{passed}{NC}{' ' * (42 - len(str(passed)))}{CYAN}║{NC}")
    print(f"{CYAN}║{NC} Failed: {RED}{failed}{NC}{' ' * (42 - len(str(failed)))}{CYAN}║{NC}")
    print(f"{CYAN}╚{'═' * 50}╝{NC}")
    
    if failed == 0:
        print(f"\n{GREEN}[✓] All security validation protocols completed successfully{NC}")
    else:
        print(f"\n{RED}[!] Security validation protocols detected {failed} failure(s){NC}")
    
    print(f"\n{DIM}End of test sequence. Awaiting further commands...{NC}\n")
