# SPARC CLI Conventions

This document outlines the conventions and usage patterns for the SPARC CLI command-line tool, which integrates with the SPARC framework to provide AI-assisted programming and research capabilities.


SPARC CLI follows SPARC's staged approach to problem-solving:

The basic syntax for sparc is:
```bash
sparc [options] -m "your task message"
```
  - Should be a clear, specific description of the task
  - Example: `sparc -m "Add error handling to the database module"`

  - Useful for understanding code or exploring concepts
  - Example: `sparc -m "Explain the authentication flow" --research-only`

   - Include relevant context in the message
   - Example: `sparc -m "Add input validation to user registration function in auth.py"`

   ```bash
   sparc -m "Add error handling to the database module"
   ```
   ```bash
   sparc -m "Explain the authentication flow" --research-only
   ```
   ```bash
   sparc -m "Refactor user authentication" --hil
   ```
   ```bash
   sparc -m "Optimize database queries" --provider openai --model gpt-4
   ```
   ```bash
   sparc -m "Analyze security vulnerabilities" --expert-provider anthropic --expert-model claude-2
