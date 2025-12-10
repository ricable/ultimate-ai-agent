# User Guide

Welcome to the Agentic Security user guide. This guide covers everything from basic usage to advanced features.

## Advanced Features

### Pattern-Based Security Analysis

The system uses advanced pattern detection to identify:
- SQL Injection vulnerabilities (including string formatting)
- Command Injection and RCE vulnerabilities
- Cross-Site Scripting (XSS) and DOM manipulation
- Weak Cryptography implementations
- Insecure Deserialization (pickle, yaml, etc)
- XXE and XML Parser vulnerabilities
- Path Traversal and file access issues
- Authentication and session weaknesses

### Caching System

- Performance-optimized result caching
- Configurable cache duration
- Automatic cache cleanup
- Cache validation checks

### Review System

Three output modes available:
1. **Basic Mode**: Simple vulnerability listing
2. **Verbose Mode**: Detailed findings with descriptions
3. **Report Mode**: Comprehensive markdown reports

### AI Integration

- **Architecture Review**: Uses GPT-4-1106-preview for high-level security analysis
- **Fix Generation**: Uses Claude-3-sonnet-20240229 for precise fix implementation
- **PR Description**: AI-generated detailed pull request descriptions with context
- **Vulnerability Analysis**: Smart severity scoring and risk prioritization
- **Multi-Model Orchestration**: Leverages both models' strengths

### Performance Optimization

- Smart caching for repeated scans
- Parallel processing capabilities
- Resource-aware scanning
- Incremental analysis support

## Command Overview

```mermaid
mindmap
  root((Commands))
    scan
      Web scanning
      Code scanning
      Dependency checks
    analyze
      Architecture review
      Vulnerability analysis
      Fix suggestions
    run
      Full pipeline
      Automated fixes
      PR creation
    validate
      Config validation
      Environment check
      System check
```

## Basic Usage

### CLI Interface

The CLI provides a cyberpunk-themed interface with advanced scanning options:

#### Basic Commands

```bash
╔══════════════════════════════════════════════════════════════╗
║                     Available Commands                      ║
╚══════════════════════════════════════════════════════════════╝

[>] scan     - Run security scans
[>] analyze  - AI-powered analysis
[>] run      - Full pipeline execution
[>] validate - Config validation
[>] version  - Show version
```

### Common Commands

1. **Run Security Scan**
   ```bash
   agentic-security scan --config config.yml
   ```

2. **Analyze Security Issues**
   ```bash
   agentic-security analyze --config config.yml
   ```

3. **Run Full Pipeline**
   ```bash
   # Full pipeline with architecture review
   agentic-security run --config config.yml --with-architecture-review

   # Skip architecture review
   agentic-security run --config config.yml --skip-architecture-review

   # Custom model selection
   agentic-security run --config config.yml --arch-model gpt-4 --fix-model claude-3
   ```

## Advanced Features

### Custom Scanning

```mermaid
flowchart TB
    subgraph Input["Scan Configuration"]
        Config[Custom Config]
        Targets[Scan Targets]
        Rules[Custom Rules]
    end

    subgraph Process["Scanning Process"]
        Load[Load Config]
        Validate[Validate Rules]
        Execute[Execute Scan]
    end

    subgraph Output["Results"]
        Report[Scan Report]
        Analysis[AI Analysis]
        Fixes[Fix Suggestions]
    end

    Input --> Process
    Process --> Output

    classDef default fill:#1a1a1a,stroke:#00ff00,color:#fff
    classDef process fill:#2a2a2a,stroke:#00ffff,color:#fff
    classDef output fill:#2a2a2a,stroke:#ff00ff,color:#fff
    
    class Input default
    class Process process
    class Output output
```

### AI Integration

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant GPT4
    participant Claude3
    participant Security
    
    User->>CLI: Run Analysis
    CLI->>Security: Scan System
    Security-->>CLI: Vulnerabilities
    CLI->>GPT4: Architecture Review
    GPT4-->>CLI: Design Analysis
    CLI->>Claude3: Generate Fixes
    Claude3-->>CLI: Fix Implementation
    CLI->>Security: Validate Fixes
    Security-->>CLI: Validation Results
    CLI-->>User: Results & PR
```

## Ethical Guidelines

### Responsible Usage

1. **Consent and Authorization**
   - Only scan authorized systems
   - Respect system boundaries
   - Follow security policies

2. **Data Protection**
   - Handle data responsibly
   - Protect sensitive information
   - Follow privacy guidelines

3. **AI Usage**
   - Review AI suggestions
   - Validate automated fixes
   - Monitor AI behavior

### Best Practices

1. **Scanning**
   - Regular scanning schedule
   - Comprehensive coverage
   - Resource consideration

2. **Analysis**
   - Thorough review
   - Context consideration
   - Impact assessment

3. **Implementation**
   - Careful validation
   - Staged deployment
   - Rollback planning

## Troubleshooting

### Common Issues

1. **Scan Failures**
   ```bash
   # Check scan status
   agentic-security validate --config config.yml
   
   # Review logs
   cat logs/security_scan.log
   ```

2. **AI Integration Issues**
   ```bash
   # Verify API keys
   agentic-security validate --check-api
   
   # Test AI connection
   agentic-security analyze --test-ai
   ```

3. **Pipeline Issues**
   ```bash
   # Check pipeline status
   agentic-security run --dry-run
   
   # Validate configuration
   agentic-security validate --full
   ```

## Configuration

### Advanced Settings

```yaml
security:
  # Custom scan settings
  scan_depth: comprehensive
  custom_rules: path/to/rules
  
  # AI settings
  ai_confidence_threshold: 0.85
  max_suggestions: 5
  
  # Pipeline settings
  parallel_scans: true
  cache_results: true
```

## Further Reading

- [Architecture Guide](../architecture/README.md)
- [Implementation Guide](../implementation/README.md)
- [Future Enhancements](../future/README.md)
