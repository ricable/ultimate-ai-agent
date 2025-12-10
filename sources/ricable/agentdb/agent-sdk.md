# Agent Skills in the SDK

> Extend Claude with specialized capabilities using Agent Skills in the Claude Agent SDK

## Overview

Agent Skills extend Claude with specialized capabilities that Claude autonomously invokes when relevant. Skills are packaged as `SKILL.md` files containing instructions, descriptions, and optional supporting resources.

For comprehensive information about Skills, including benefits, architecture, and authoring guidelines, see the [Agent Skills overview](/en/docs/agents-and-tools/agent-skills/overview).

## How Skills Work with the SDK

When using the Claude Agent SDK, Skills are:

1. **Defined as filesystem artifacts**: Created as `SKILL.md` files in specific directories (`.claude/skills/`)
2. **Loaded from filesystem**: Skills are loaded from configured filesystem locations. You must specify `settingSources` (TypeScript) or `setting_sources` (Python) to load Skills from the filesystem
3. **Automatically discovered**: Once filesystem settings are loaded, Skill metadata is discovered at startup from user and project directories; full content loaded when triggered
4. **Model-invoked**: Claude autonomously chooses when to use them based on context
5. **Enabled via allowed\_tools**: Add `"Skill"` to your `allowed_tools` to enable Skills

Unlike subagents (which can be defined programmatically), Skills must be created as filesystem artifacts. The SDK does not provide a programmatic API for registering Skills.

<Note>
  **Default behavior**: By default, the SDK does not load any filesystem settings. To use Skills, you must explicitly configure `settingSources: ['user', 'project']` (TypeScript) or `setting_sources=["user", "project"]` (Python) in your options.
</Note>

## Using Skills with the SDK

To use Skills with the SDK, you need to:

1. Include `"Skill"` in your `allowed_tools` configuration
2. Configure `settingSources`/`setting_sources` to load Skills from the filesystem

Once configured, Claude automatically discovers Skills from the specified directories and invokes them when relevant to the user's request.

<CodeGroup>
  ```python Python theme={null}
  import asyncio
  from claude_agent_sdk import query, ClaudeAgentOptions

  async def main():
      options = ClaudeAgentOptions(
          cwd="/path/to/project",  # Project with .claude/skills/
          setting_sources=["user", "project"],  # Load Skills from filesystem
          allowed_tools=["Skill", "Read", "Write", "Bash"]  # Enable Skill tool
      )

      async for message in query(
          prompt="Help me process this PDF document",
          options=options
      ):
          print(message)

  asyncio.run(main())
  ```

  ```typescript TypeScript theme={null}
  import { query } from "@anthropic-ai/claude-agent-sdk";

  for await (const message of query({
    prompt: "Help me process this PDF document",
    options: {
      cwd: "/path/to/project",  // Project with .claude/skills/
      settingSources: ["user", "project"],  // Load Skills from filesystem
      allowedTools: ["Skill", "Read", "Write", "Bash"]  // Enable Skill tool
    }
  })) {
    console.log(message);
  }
  ```
</CodeGroup>

## Skill Locations

Skills are loaded from filesystem directories based on your `settingSources`/`setting_sources` configuration:

* **Project Skills** (`.claude/skills/`): Shared with your team via git - loaded when `setting_sources` includes `"project"`
* **User Skills** (`~/.claude/skills/`): Personal Skills across all projects - loaded when `setting_sources` includes `"user"`
* **Plugin Skills**: Bundled with installed Claude Code plugins

## Creating Skills

Skills are defined as directories containing a `SKILL.md` file with YAML frontmatter and Markdown content. The `description` field determines when Claude invokes your Skill.

**Example directory structure**:

```bash  theme={null}
.claude/skills/processing-pdfs/
└── SKILL.md
```

For complete guidance on creating Skills, including SKILL.md structure, multi-file Skills, and examples, see:

* [Agent Skills in Claude Code](/en/docs/claude-code/skills): Complete guide with examples
* [Agent Skills Best Practices](/en/docs/agents-and-tools/agent-skills/best-practices): Authoring guidelines and naming conventions

## Tool Restrictions

<Note>
  The `allowed-tools` frontmatter field in SKILL.md is only supported when using Claude Code CLI directly. **It does not apply when using Skills through the SDK**.

  When using the SDK, control tool access through the main `allowedTools` option in your query configuration.
</Note>

To restrict tools for Skills in SDK applications, use the `allowedTools` option:

<Note>
  Import statements from the first example are assumed in the following code snippets.
</Note>

<CodeGroup>
  ```python Python theme={null}
  options = ClaudeAgentOptions(
      setting_sources=["user", "project"],  # Load Skills from filesystem
      allowed_tools=["Skill", "Read", "Grep", "Glob"]  # Restricted toolset
  )

  async for message in query(
      prompt="Analyze the codebase structure",
      options=options
  ):
      print(message)
  ```

  ```typescript TypeScript theme={null}
  // Skills can only use Read, Grep, and Glob tools
  for await (const message of query({
    prompt: "Analyze the codebase structure",
    options: {
      settingSources: ["user", "project"],  // Load Skills from filesystem
      allowedTools: ["Skill", "Read", "Grep", "Glob"]  // Restricted toolset
    }
  })) {
    console.log(message);
  }
  ```
</CodeGroup>

## Discovering Available Skills

To see which Skills are available in your SDK application, simply ask Claude:

<CodeGroup>
  ```python Python theme={null}
  options = ClaudeAgentOptions(
      setting_sources=["user", "project"],  # Load Skills from filesystem
      allowed_tools=["Skill"]
  )

  async for message in query(
      prompt="What Skills are available?",
      options=options
  ):
      print(message)
  ```

  ```typescript TypeScript theme={null}
  for await (const message of query({
    prompt: "What Skills are available?",
    options: {
      settingSources: ["user", "project"],  // Load Skills from filesystem
      allowedTools: ["Skill"]
    }
  })) {
    console.log(message);
  }
  ```
</CodeGroup>

Claude will list the available Skills based on your current working directory and installed plugins.

## Testing Skills

Test Skills by asking questions that match their descriptions:

<CodeGroup>
  ```python Python theme={null}
  options = ClaudeAgentOptions(
      cwd="/path/to/project",
      setting_sources=["user", "project"],  # Load Skills from filesystem
      allowed_tools=["Skill", "Read", "Bash"]
  )

  async for message in query(
      prompt="Extract text from invoice.pdf",
      options=options
  ):
      print(message)
  ```

  ```typescript TypeScript theme={null}
  for await (const message of query({
    prompt: "Extract text from invoice.pdf",
    options: {
      cwd: "/path/to/project",
      settingSources: ["user", "project"],  // Load Skills from filesystem
      allowedTools: ["Skill", "Read", "Bash"]
    }
  })) {
    console.log(message);
  }
  ```
</CodeGroup>

Claude automatically invokes the relevant Skill if the description matches your request.

## Troubleshooting

### Skills Not Found

**Check settingSources configuration**: Skills are only loaded when you explicitly configure `settingSources`/`setting_sources`. This is the most common issue:

<CodeGroup>
  ```python Python theme={null}
  # Wrong - Skills won't be loaded
  options = ClaudeAgentOptions(
      allowed_tools=["Skill"]
  )

  # Correct - Skills will be loaded
  options = ClaudeAgentOptions(
      setting_sources=["user", "project"],  # Required to load Skills
      allowed_tools=["Skill"]
  )
  ```

  ```typescript TypeScript theme={null}
  // Wrong - Skills won't be loaded
  const options = {
    allowedTools: ["Skill"]
  };

  // Correct - Skills will be loaded
  const options = {
    settingSources: ["user", "project"],  // Required to load Skills
    allowedTools: ["Skill"]
  };
  ```
</CodeGroup>

For more details on `settingSources`/`setting_sources`, see the [TypeScript SDK reference](/en/api/agent-sdk/typescript#settingsource) or [Python SDK reference](/en/api/agent-sdk/python#settingsource).

**Check working directory**: The SDK loads Skills relative to the `cwd` option. Ensure it points to a directory containing `.claude/skills/`:

<CodeGroup>
  ```python Python theme={null}
  # Ensure your cwd points to the directory containing .claude/skills/
  options = ClaudeAgentOptions(
      cwd="/path/to/project",  # Must contain .claude/skills/
      setting_sources=["user", "project"],  # Required to load Skills
      allowed_tools=["Skill"]
  )
  ```

  ```typescript TypeScript theme={null}
  // Ensure your cwd points to the directory containing .claude/skills/
  const options = {
    cwd: "/path/to/project",  // Must contain .claude/skills/
    settingSources: ["user", "project"],  // Required to load Skills
    allowedTools: ["Skill"]
  };
  ```
</CodeGroup>

See the "Using Skills with the SDK" section above for the complete pattern.

**Verify filesystem location**:

```bash  theme={null}
# Check project Skills
ls .claude/skills/*/SKILL.md

# Check personal Skills
ls ~/.claude/skills/*/SKILL.md
```

### Skill Not Being Used

**Check the Skill tool is enabled**: Confirm `"Skill"` is in your `allowedTools`.

**Check the description**: Ensure it's specific and includes relevant keywords. See [Agent Skills Best Practices](/en/docs/agents-and-tools/agent-skills/best-practices#writing-effective-descriptions) for guidance on writing effective descriptions.

### Additional Troubleshooting

For general Skills troubleshooting (YAML syntax, debugging, etc.), see the [Claude Code Skills troubleshooting section](/en/docs/claude-code/skills#troubleshooting).

## Related Documentation

### Skills Guides

* [Agent Skills in Claude Code](/en/docs/claude-code/skills): Complete Skills guide with creation, examples, and troubleshooting
* [Agent Skills Overview](/en/docs/agents-and-tools/agent-skills/overview): Conceptual overview, benefits, and architecture
* [Agent Skills Best Practices](/en/docs/agents-and-tools/agent-skills/best-practices): Authoring guidelines for effective Skills
* [Agent Skills Cookbook](https://github.com/anthropics/claude-cookbooks/tree/main/skills): Example Skills and templates

### SDK Resources

* [Subagents in the SDK](/en/api/agent-sdk/subagents): Similar filesystem-based agents with programmatic options
* [Slash Commands in the SDK](/en/api/agent-sdk/slash-commands): User-invoked commands
* [SDK Overview](/en/api/agent-sdk/overview): General SDK concepts
* [TypeScript SDK Reference](/en/api/agent-sdk/typescript): Complete API documentation
* [Python SDK Reference](/en/api/agent-sdk/python): Complete API documentation

---
# Subagents in the SDK

> Working with subagents in the Claude Agent SDK

Subagents in the Claude Agent SDK are specialized AIs that are orchestrated by the main agent.
Use subagents for context management and parallelization.

This guide explains how to define and use subagents in the SDK using the `agents` parameter.

## Overview

Subagents can be defined in two ways when using the SDK:

1. **Programmatically** - Using the `agents` parameter in your `query()` options (recommended for SDK applications)
2. **Filesystem-based** - Placing markdown files with YAML frontmatter in designated directories (`.claude/agents/`)

This guide primarily focuses on the programmatic approach using the `agents` parameter, which provides a more integrated development experience for SDK applications.

## Benefits of Using Subagents

### Context Management

Subagents maintain separate context from the main agent, preventing information overload and keeping interactions focused. This isolation ensures that specialized tasks don't pollute the main conversation context with irrelevant details.

**Example**: A `research-assistant` subagent can explore dozens of files and documentation pages without cluttering the main conversation with all the intermediate search results - only returning the relevant findings.

### Parallelization

Multiple subagents can run concurrently, dramatically speeding up complex workflows.

**Example**: During a code review, you can run `style-checker`, `security-scanner`, and `test-coverage` subagents simultaneously, reducing review time from minutes to seconds.

### Specialized Instructions and Knowledge

Each subagent can have tailored system prompts with specific expertise, best practices, and constraints.

**Example**: A `database-migration` subagent can have detailed knowledge about SQL best practices, rollback strategies, and data integrity checks that would be unnecessary noise in the main agent's instructions.

### Tool Restrictions

Subagents can be limited to specific tools, reducing the risk of unintended actions.

**Example**: A `doc-reviewer` subagent might only have access to Read and Grep tools, ensuring it can analyze but never accidentally modify your documentation files.

## Creating Subagents

### Programmatic Definition (Recommended)

Define subagents directly in your code using the `agents` parameter:

```typescript  theme={null}
import { query } from '@anthropic-ai/claude-agent-sdk';

const result = query({
  prompt: "Review the authentication module for security issues",
  options: {
    agents: {
      'code-reviewer': {
        description: 'Expert code review specialist. Use for quality, security, and maintainability reviews.',
        prompt: `You are a code review specialist with expertise in security, performance, and best practices.

When reviewing code:
- Identify security vulnerabilities
- Check for performance issues
- Verify adherence to coding standards
- Suggest specific improvements

Be thorough but concise in your feedback.`,
        tools: ['Read', 'Grep', 'Glob'],
        model: 'sonnet'
      },
      'test-runner': {
        description: 'Runs and analyzes test suites. Use for test execution and coverage analysis.',
        prompt: `You are a test execution specialist. Run tests and provide clear analysis of results.

Focus on:
- Running test commands
- Analyzing test output
- Identifying failing tests
- Suggesting fixes for failures`,
        tools: ['Bash', 'Read', 'Grep'],
      }
    }
  }
});

for await (const message of result) {
  console.log(message);
}
```

### AgentDefinition Configuration

| Field         | Type                                         | Required | Description                                                      |
| :------------ | :------------------------------------------- | :------- | :--------------------------------------------------------------- |
| `description` | `string`                                     | Yes      | Natural language description of when to use this agent           |
| `prompt`      | `string`                                     | Yes      | The agent's system prompt defining its role and behavior         |
| `tools`       | `string[]`                                   | No       | Array of allowed tool names. If omitted, inherits all tools      |
| `model`       | `'sonnet' \| 'opus' \| 'haiku' \| 'inherit'` | No       | Model override for this agent. Defaults to main model if omitted |

### Filesystem-Based Definition (Alternative)

You can also define subagents as markdown files in specific directories:

* **Project-level**: `.claude/agents/*.md` - Available only in the current project
* **User-level**: `~/.claude/agents/*.md` - Available across all projects

Each subagent is a markdown file with YAML frontmatter:

```markdown  theme={null}
---
name: code-reviewer
description: Expert code review specialist. Use for quality, security, and maintainability reviews.
tools: Read, Grep, Glob, Bash
---

Your subagent's system prompt goes here. This defines the subagent's
role, capabilities, and approach to solving problems.
```

**Note:** Programmatically defined agents (via the `agents` parameter) take precedence over filesystem-based agents with the same name.

## How the SDK Uses Subagents

When using the Claude Agent SDK, subagents can be defined programmatically or loaded from the filesystem. Claude will:

1. **Load programmatic agents** from the `agents` parameter in your options
2. **Auto-detect filesystem agents** from `.claude/agents/` directories (if not overridden)
3. **Invoke them automatically** based on task matching and the agent's `description`
4. **Use their specialized prompts** and tool restrictions
5. **Maintain separate context** for each subagent invocation

Programmatically defined agents (via `agents` parameter) take precedence over filesystem-based agents with the same name.

## Example Subagents

For comprehensive examples of subagents including code reviewers, test runners, debuggers, and security auditors, see the [main Subagents guide](/en/docs/claude-code/sub-agents#example-subagents). The guide includes detailed configurations and best practices for creating effective subagents.

## SDK Integration Patterns

### Automatic Invocation

The SDK will automatically invoke appropriate subagents based on the task context. Ensure your agent's `description` field clearly indicates when it should be used:

```typescript  theme={null}
const result = query({
  prompt: "Optimize the database queries in the API layer",
  options: {
    agents: {
      'performance-optimizer': {
        description: 'Use PROACTIVELY when code changes might impact performance. MUST BE USED for optimization tasks.',
        prompt: 'You are a performance optimization specialist...',
        tools: ['Read', 'Edit', 'Bash', 'Grep'],
        model: 'sonnet'
      }
    }
  }
});
```

### Explicit Invocation

Users can request specific subagents in their prompts:

```typescript  theme={null}
const result = query({
  prompt: "Use the code-reviewer agent to check the authentication module",
  options: {
    agents: {
      'code-reviewer': {
        description: 'Expert code review specialist',
        prompt: 'You are a security-focused code reviewer...',
        tools: ['Read', 'Grep', 'Glob']
      }
    }
  }
});
```

### Dynamic Agent Configuration

You can dynamically configure agents based on your application's needs:

```typescript  theme={null}
import { query, type AgentDefinition } from '@anthropic-ai/claude-agent-sdk';

function createSecurityAgent(securityLevel: 'basic' | 'strict'): AgentDefinition {
  return {
    description: 'Security code reviewer',
    prompt: `You are a ${securityLevel === 'strict' ? 'strict' : 'balanced'} security reviewer...`,
    tools: ['Read', 'Grep', 'Glob'],
    model: securityLevel === 'strict' ? 'opus' : 'sonnet'
  };
}

const result = query({
  prompt: "Review this PR for security issues",
  options: {
    agents: {
      'security-reviewer': createSecurityAgent('strict')
    }
  }
});
```

## Tool Restrictions

Subagents can have restricted tool access via the `tools` field:

* **Omit the field** - Agent inherits all available tools (default)
* **Specify tools** - Agent can only use listed tools

Example of a read-only analysis agent:

```typescript  theme={null}
const result = query({
  prompt: "Analyze the architecture of this codebase",
  options: {
    agents: {
      'code-analyzer': {
        description: 'Static code analysis and architecture review',
        prompt: `You are a code architecture analyst. Analyze code structure,
identify patterns, and suggest improvements without making changes.`,
        tools: ['Read', 'Grep', 'Glob']  // No write or execute permissions
      }
    }
  }
});
```

### Common Tool Combinations

**Read-only agents** (analysis, review):

```typescript  theme={null}
tools: ['Read', 'Grep', 'Glob']
```

**Test execution agents**:

```typescript  theme={null}
tools: ['Bash', 'Read', 'Grep']
```

**Code modification agents**:

```typescript  theme={null}
tools: ['Read', 'Edit', 'Write', 'Grep', 'Glob']
```

## Related Documentation

* [Main Subagents Guide](/en/docs/claude-code/sub-agents) - Comprehensive subagent documentation
* [SDK Overview](/en/api/agent-sdk/overview) - Overview of Claude Agent SDK
* [Settings](/en/docs/claude-code/settings) - Configuration file reference
* [Slash Commands](/en/docs/claude-code/slash-commands) - Custom command creation

---
# Slash Commands in the SDK

> Learn how to use slash commands to control Claude Code sessions through the SDK

Slash commands provide a way to control Claude Code sessions with special commands that start with `/`. These commands can be sent through the SDK to perform actions like clearing conversation history, compacting messages, or getting help.

## Discovering Available Slash Commands

The Claude Agent SDK provides information about available slash commands in the system initialization message. Access this information when your session starts:

<CodeGroup>
  ```typescript TypeScript theme={null}
  import { query } from "@anthropic-ai/claude-agent-sdk";

  for await (const message of query({
    prompt: "Hello Claude",
    options: { maxTurns: 1 }
  })) {
    if (message.type === "system" && message.subtype === "init") {
      console.log("Available slash commands:", message.slash_commands);
      // Example output: ["/compact", "/clear", "/help"]
    }
  }
  ```

  ```python Python theme={null}
  import asyncio
  from claude_agent_sdk import query

  async def main():
      async for message in query(
          prompt="Hello Claude",
          options={"max_turns": 1}
      ):
          if message.type == "system" and message.subtype == "init":
              print("Available slash commands:", message.slash_commands)
              # Example output: ["/compact", "/clear", "/help"]

  asyncio.run(main())
  ```
</CodeGroup>

## Sending Slash Commands

Send slash commands by including them in your prompt string, just like regular text:

<CodeGroup>
  ```typescript TypeScript theme={null}
  import { query } from "@anthropic-ai/claude-agent-sdk";

  // Send a slash command
  for await (const message of query({
    prompt: "/compact",
    options: { maxTurns: 1 }
  })) {
    if (message.type === "result") {
      console.log("Command executed:", message.result);
    }
  }
  ```

  ```python Python theme={null}
  import asyncio
  from claude_agent_sdk import query

  async def main():
      # Send a slash command
      async for message in query(
          prompt="/compact",
          options={"max_turns": 1}
      ):
          if message.type == "result":
              print("Command executed:", message.result)

  asyncio.run(main())
  ```
</CodeGroup>

## Common Slash Commands

### `/compact` - Compact Conversation History

The `/compact` command reduces the size of your conversation history by summarizing older messages while preserving important context:

<CodeGroup>
  ```typescript TypeScript theme={null}
  import { query } from "@anthropic-ai/claude-agent-sdk";

  for await (const message of query({
    prompt: "/compact",
    options: { maxTurns: 1 }
  })) {
    if (message.type === "system" && message.subtype === "compact_boundary") {
      console.log("Compaction completed");
      console.log("Pre-compaction tokens:", message.compact_metadata.pre_tokens);
      console.log("Trigger:", message.compact_metadata.trigger);
    }
  }
  ```

  ```python Python theme={null}
  import asyncio
  from claude_agent_sdk import query

  async def main():
      async for message in query(
          prompt="/compact",
          options={"max_turns": 1}
      ):
          if (message.type == "system" and 
              message.subtype == "compact_boundary"):
              print("Compaction completed")
              print("Pre-compaction tokens:", 
                    message.compact_metadata.pre_tokens)
              print("Trigger:", message.compact_metadata.trigger)

  asyncio.run(main())
  ```
</CodeGroup>

### `/clear` - Clear Conversation

The `/clear` command starts a fresh conversation by clearing all previous history:

<CodeGroup>
  ```typescript TypeScript theme={null}
  import { query } from "@anthropic-ai/claude-agent-sdk";

  // Clear conversation and start fresh
  for await (const message of query({
    prompt: "/clear",
    options: { maxTurns: 1 }
  })) {
    if (message.type === "system" && message.subtype === "init") {
      console.log("Conversation cleared, new session started");
      console.log("Session ID:", message.session_id);
    }
  }
  ```

  ```python Python theme={null}
  import asyncio
  from claude_agent_sdk import query

  async def main():
      # Clear conversation and start fresh
      async for message in query(
          prompt="/clear",
          options={"max_turns": 1}
      ):
          if message.type == "system" and message.subtype == "init":
              print("Conversation cleared, new session started")
              print("Session ID:", message.session_id)

  asyncio.run(main())
  ```
</CodeGroup>

## Creating Custom Slash Commands

In addition to using built-in slash commands, you can create your own custom commands that are available through the SDK. Custom commands are defined as markdown files in specific directories, similar to how subagents are configured.

### File Locations

Custom slash commands are stored in designated directories based on their scope:

* **Project commands**: `.claude/commands/` - Available only in the current project
* **Personal commands**: `~/.claude/commands/` - Available across all your projects

### File Format

Each custom command is a markdown file where:

* The filename (without `.md` extension) becomes the command name
* The file content defines what the command does
* Optional YAML frontmatter provides configuration

#### Basic Example

Create `.claude/commands/refactor.md`:

```markdown  theme={null}
Refactor the selected code to improve readability and maintainability.
Focus on clean code principles and best practices.
```

This creates the `/refactor` command that you can use through the SDK.

#### With Frontmatter

Create `.claude/commands/security-check.md`:

```markdown  theme={null}
---
allowed-tools: Read, Grep, Glob
description: Run security vulnerability scan
model: claude-sonnet-4-5-20250929
---

Analyze the codebase for security vulnerabilities including:
- SQL injection risks
- XSS vulnerabilities
- Exposed credentials
- Insecure configurations
```

### Using Custom Commands in the SDK

Once defined in the filesystem, custom commands are automatically available through the SDK:

<CodeGroup>
  ```typescript TypeScript theme={null}
  import { query } from "@anthropic-ai/claude-agent-sdk";

  // Use a custom command
  for await (const message of query({
    prompt: "/refactor src/auth/login.ts",
    options: { maxTurns: 3 }
  })) {
    if (message.type === "assistant") {
      console.log("Refactoring suggestions:", message.message);
    }
  }

  // Custom commands appear in the slash_commands list
  for await (const message of query({
    prompt: "Hello",
    options: { maxTurns: 1 }
  })) {
    if (message.type === "system" && message.subtype === "init") {
      // Will include both built-in and custom commands
      console.log("Available commands:", message.slash_commands);
      // Example: ["/compact", "/clear", "/help", "/refactor", "/security-check"]
    }
  }
  ```

  ```python Python theme={null}
  import asyncio
  from claude_agent_sdk import query

  async def main():
      # Use a custom command
      async for message in query(
          prompt="/refactor src/auth/login.py",
          options={"max_turns": 3}
      ):
          if message.type == "assistant":
              print("Refactoring suggestions:", message.message)
      
      # Custom commands appear in the slash_commands list
      async for message in query(
          prompt="Hello",
          options={"max_turns": 1}
      ):
          if message.type == "system" and message.subtype == "init":
              # Will include both built-in and custom commands
              print("Available commands:", message.slash_commands)
              # Example: ["/compact", "/clear", "/help", "/refactor", "/security-check"]

  asyncio.run(main())
  ```
</CodeGroup>

### Advanced Features

#### Arguments and Placeholders

Custom commands support dynamic arguments using placeholders:

Create `.claude/commands/fix-issue.md`:

```markdown  theme={null}
---
argument-hint: [issue-number] [priority]
description: Fix a GitHub issue
---

Fix issue #$1 with priority $2.
Check the issue description and implement the necessary changes.
```

Use in SDK:

<CodeGroup>
  ```typescript TypeScript theme={null}
  import { query } from "@anthropic-ai/claude-agent-sdk";

  // Pass arguments to custom command
  for await (const message of query({
    prompt: "/fix-issue 123 high",
    options: { maxTurns: 5 }
  })) {
    // Command will process with $1="123" and $2="high"
    if (message.type === "result") {
      console.log("Issue fixed:", message.result);
    }
  }
  ```

  ```python Python theme={null}
  import asyncio
  from claude_agent_sdk import query

  async def main():
      # Pass arguments to custom command
      async for message in query(
          prompt="/fix-issue 123 high",
          options={"max_turns": 5}
      ):
          # Command will process with $1="123" and $2="high"
          if message.type == "result":
              print("Issue fixed:", message.result)

  asyncio.run(main())
  ```
</CodeGroup>

#### Bash Command Execution

Custom commands can execute bash commands and include their output:

Create `.claude/commands/git-commit.md`:

```markdown  theme={null}
---
allowed-tools: Bash(git add:*), Bash(git status:*), Bash(git commit:*)
description: Create a git commit
---

## Context

- Current status: !`git status`
- Current diff: !`git diff HEAD`

## Task

Create a git commit with appropriate message based on the changes.
```

#### File References

Include file contents using the `@` prefix:

Create `.claude/commands/review-config.md`:

```markdown  theme={null}
---
description: Review configuration files
---

Review the following configuration files for issues:
- Package config: @package.json
- TypeScript config: @tsconfig.json
- Environment config: @.env

Check for security issues, outdated dependencies, and misconfigurations.
```

### Organization with Namespacing

Organize commands in subdirectories for better structure:

```bash  theme={null}
.claude/commands/
├── frontend/
│   ├── component.md      # Creates /component (project:frontend)
│   └── style-check.md     # Creates /style-check (project:frontend)
├── backend/
│   ├── api-test.md        # Creates /api-test (project:backend)
│   └── db-migrate.md      # Creates /db-migrate (project:backend)
└── review.md              # Creates /review (project)
```

The subdirectory appears in the command description but doesn't affect the command name itself.

### Practical Examples

#### Code Review Command

Create `.claude/commands/code-review.md`:

```markdown  theme={null}
---
allowed-tools: Read, Grep, Glob, Bash(git diff:*)
description: Comprehensive code review
---

## Changed Files
!`git diff --name-only HEAD~1`

## Detailed Changes
!`git diff HEAD~1`

## Review Checklist

Review the above changes for:
1. Code quality and readability
2. Security vulnerabilities
3. Performance implications
4. Test coverage
5. Documentation completeness

Provide specific, actionable feedback organized by priority.
```

#### Test Runner Command

Create `.claude/commands/test.md`:

```markdown  theme={null}
---
allowed-tools: Bash, Read, Edit
argument-hint: [test-pattern]
description: Run tests with optional pattern
---

Run tests matching pattern: $ARGUMENTS

1. Detect the test framework (Jest, pytest, etc.)
2. Run tests with the provided pattern
3. If tests fail, analyze and fix them
4. Re-run to verify fixes
```

Use these commands through the SDK:

<CodeGroup>
  ```typescript TypeScript theme={null}
  import { query } from "@anthropic-ai/claude-agent-sdk";

  // Run code review
  for await (const message of query({
    prompt: "/code-review",
    options: { maxTurns: 3 }
  })) {
    // Process review feedback
  }

  // Run specific tests
  for await (const message of query({
    prompt: "/test auth",
    options: { maxTurns: 5 }
  })) {
    // Handle test results
  }
  ```

  ```python Python theme={null}
  import asyncio
  from claude_agent_sdk import query

  async def main():
      # Run code review
      async for message in query(
          prompt="/code-review",
          options={"max_turns": 3}
      ):
          # Process review feedback
          pass
      
      # Run specific tests
      async for message in query(
          prompt="/test auth",
          options={"max_turns": 5}
      ):
          # Handle test results
          pass

  asyncio.run(main())
  ```
</CodeGroup>

## See Also

* [Slash Commands](/en/docs/claude-code/slash-commands) - Complete slash command documentation
* [Subagents in the SDK](/en/api/agent-sdk/subagents) - Similar filesystem-based configuration for subagents
* [TypeScript SDK reference](/en/docs/claude-code/typescript-sdk-reference) - Complete API documentation
* [SDK overview](/en/api/agent-sdk/overview) - General SDK concepts
* [CLI reference](/en/docs/claude-code/cli-reference) - Command-line interface

----
# MCP in the SDK

> Extend Claude Code with custom tools using Model Context Protocol servers

## Overview

Model Context Protocol (MCP) servers extend Claude Code with custom tools and capabilities. MCPs can run as external processes, connect via HTTP/SSE, or execute directly within your SDK application.

## Configuration

### Basic Configuration

Configure MCP servers in `.mcp.json` at your project root:

<CodeGroup>
  ```json TypeScript theme={null}
  {
    "mcpServers": {
      "filesystem": {
        "command": "npx",
        "args": ["@modelcontextprotocol/server-filesystem"],
        "env": {
          "ALLOWED_PATHS": "/Users/me/projects"
        }
      }
    }
  }
  ```

  ```json Python theme={null}
  {
    "mcpServers": {
      "filesystem": {
        "command": "python",
        "args": ["-m", "mcp_server_filesystem"],
        "env": {
          "ALLOWED_PATHS": "/Users/me/projects"
        }
      }
    }
  }
  ```
</CodeGroup>

### Using MCP Servers in SDK

<CodeGroup>
  ```typescript TypeScript theme={null}
  import { query } from "@anthropic-ai/claude-agent-sdk";

  for await (const message of query({
    prompt: "List files in my project",
    options: {
      mcpServers: {
        "filesystem": {
          command: "npx",
          args: ["@modelcontextprotocol/server-filesystem"],
          env: {
            ALLOWED_PATHS: "/Users/me/projects"
          }
        }
      },
      allowedTools: ["mcp__filesystem__list_files"]
    }
  })) {
    if (message.type === "result" && message.subtype === "success") {
      console.log(message.result);
    }
  }
  ```

  ```python Python theme={null}
  from claude_agent_sdk import query

  async for message in query(
      prompt="List files in my project",
      options={
          "mcpServers": {
              "filesystem": {
                  "command": "python",
                  "args": ["-m", "mcp_server_filesystem"],
                  "env": {
                      "ALLOWED_PATHS": "/Users/me/projects"
                  }
              }
          },
          "allowedTools": ["mcp__filesystem__list_files"]
      }
  ):
      if message["type"] == "result" and message["subtype"] == "success":
          print(message["result"])
  ```
</CodeGroup>

## Transport Types

### stdio Servers

External processes communicating via stdin/stdout:

<CodeGroup>
  ```typescript TypeScript theme={null}
  // .mcp.json configuration
  {
    "mcpServers": {
      "my-tool": {
        "command": "node",
        "args": ["./my-mcp-server.js"],
        "env": {
          "DEBUG": "${DEBUG:-false}"
        }
      }
    }
  }
  ```

  ```python Python theme={null}
  # .mcp.json configuration
  {
    "mcpServers": {
      "my-tool": {
        "command": "python",
        "args": ["./my_mcp_server.py"],
        "env": {
          "DEBUG": "${DEBUG:-false}"
        }
      }
    }
  }
  ```
</CodeGroup>

### HTTP/SSE Servers

Remote servers with network communication:

<CodeGroup>
  ```typescript TypeScript theme={null}
  // SSE server configuration
  {
    "mcpServers": {
      "remote-api": {
        "type": "sse",
        "url": "https://api.example.com/mcp/sse",
        "headers": {
          "Authorization": "Bearer ${API_TOKEN}"
        }
      }
    }
  }

  // HTTP server configuration
  {
    "mcpServers": {
      "http-service": {
        "type": "http",
        "url": "https://api.example.com/mcp",
        "headers": {
          "X-API-Key": "${API_KEY}"
        }
      }
    }
  }
  ```

  ```python Python theme={null}
  # SSE server configuration
  {
    "mcpServers": {
      "remote-api": {
        "type": "sse",
        "url": "https://api.example.com/mcp/sse",
        "headers": {
          "Authorization": "Bearer ${API_TOKEN}"
        }
      }
    }
  }

  # HTTP server configuration
  {
    "mcpServers": {
      "http-service": {
        "type": "http",
        "url": "https://api.example.com/mcp",
        "headers": {
          "X-API-Key": "${API_KEY}"
        }
      }
    }
  }
  ```
</CodeGroup>

### SDK MCP Servers

In-process servers running within your application. For detailed information on creating custom tools, see the [Custom Tools guide](/en/api/agent-sdk/custom-tools):

## Resource Management

MCP servers can expose resources that Claude can list and read:

<CodeGroup>
  ```typescript TypeScript theme={null}
  import { query } from "@anthropic-ai/claude-agent-sdk";

  // List available resources
  for await (const message of query({
    prompt: "What resources are available from the database server?",
    options: {
      mcpServers: {
        "database": {
          command: "npx",
          args: ["@modelcontextprotocol/server-database"]
        }
      },
      allowedTools: ["mcp__list_resources", "mcp__read_resource"]
    }
  })) {
    if (message.type === "result") console.log(message.result);
  }
  ```

  ```python Python theme={null}
  from claude_agent_sdk import query

  # List available resources
  async for message in query(
      prompt="What resources are available from the database server?",
      options={
          "mcpServers": {
              "database": {
                  "command": "python",
                  "args": ["-m", "mcp_server_database"]
              }
          },
          "allowedTools": ["mcp__list_resources", "mcp__read_resource"]
      }
  ):
      if message["type"] == "result":
          print(message["result"])
  ```
</CodeGroup>

## Authentication

### Environment Variables

<CodeGroup>
  ```typescript TypeScript theme={null}
  // .mcp.json with environment variables
  {
    "mcpServers": {
      "secure-api": {
        "type": "sse",
        "url": "https://api.example.com/mcp",
        "headers": {
          "Authorization": "Bearer ${API_TOKEN}",
          "X-API-Key": "${API_KEY:-default-key}"
        }
      }
    }
  }

  // Set environment variables
  process.env.API_TOKEN = "your-token";
  process.env.API_KEY = "your-key";
  ```

  ```python Python theme={null}
  # .mcp.json with environment variables
  {
    "mcpServers": {
      "secure-api": {
        "type": "sse",
        "url": "https://api.example.com/mcp",
        "headers": {
          "Authorization": "Bearer ${API_TOKEN}",
          "X-API-Key": "${API_KEY:-default-key}"
        }
      }
    }
  }

  # Set environment variables
  import os
  os.environ["API_TOKEN"] = "your-token"
  os.environ["API_KEY"] = "your-key"
  ```
</CodeGroup>

### OAuth2 Authentication

OAuth2 MCP authentication in-client is not currently supported.

## Error Handling

Handle MCP connection failures gracefully:

<CodeGroup>
  ```typescript TypeScript theme={null}
  import { query } from "@anthropic-ai/claude-agent-sdk";

  for await (const message of query({
    prompt: "Process data",
    options: {
      mcpServers: {
        "data-processor": dataServer
      }
    }
  })) {
    if (message.type === "system" && message.subtype === "init") {
      // Check MCP server status
      const failedServers = message.mcp_servers.filter(
        s => s.status !== "connected"
      );
      
      if (failedServers.length > 0) {
        console.warn("Failed to connect:", failedServers);
      }
    }
    
    if (message.type === "result" && message.subtype === "error_during_execution") {
      console.error("Execution failed");
    }
  }
  ```

  ```python Python theme={null}
  from claude_agent_sdk import query

  async for message in query(
      prompt="Process data",
      options={
          "mcpServers": {
              "data-processor": data_server
          }
      }
  ):
      if message["type"] == "system" and message["subtype"] == "init":
          # Check MCP server status
          failed_servers = [
              s for s in message["mcp_servers"]
              if s["status"] != "connected"
          ]
          
          if failed_servers:
              print(f"Failed to connect: {failed_servers}")
      
      if message["type"] == "result" and message["subtype"] == "error_during_execution":
          print("Execution failed")
  ```
</CodeGroup>

## Related Resources

* [Custom Tools Guide](/en/api/agent-sdk/custom-tools) - Detailed guide on creating SDK MCP servers
* [TypeScript SDK Reference](/en/api/agent-sdk/typescript)
* [Python SDK Reference](/en/api/agent-sdk/python)
* [SDK Permissions](/en/api/agent-sdk/sdk-permissions)
* [Common Workflows](/en/docs/claude-code/common-workflows)
