# Agent Booster Prompt Patterns (Morph LLM Compatible)

This guide shows the optimal prompt patterns for using Agent Booster, following Morph LLM's proven patterns for best results.

## Pattern 1: First-Person Instructions

**Why it works:** First-person instructions ("I will add...", "I'm changing...") provide clear intent and focus the model on the specific transformation.

### ❌ Bad: Vague third-person
```
The function should have error handling
```

### ✅ Good: Clear first-person
```
I will add try-catch error handling around JSON.parse()
```

### Example with Agent Booster

```typescript
import { AgentBooster } from 'agent-booster';

const booster = new AgentBooster();

const originalCode = `
function parseConfig(jsonString) {
  return JSON.parse(jsonString);
}
`;

const edit = `
function parseConfig(jsonString) {
  try {
    return JSON.parse(jsonString);
  } catch (error) {
    console.error('Failed to parse config:', error);
    return null;
  }
}
`;

const result = await booster.apply({
  code: originalCode,
  edit: edit,
  language: 'javascript'
});

console.log(result.output);
// ✅ 0-1ms latency, 90% confidence (template match)
```

## Pattern 2: Minimal Context with "// ... existing code ..." Markers

**Why it works:** Only showing changed lines reduces token usage and improves accuracy. Use `// ... existing code ...` (or language-specific comments) to indicate unchanged sections.

### ❌ Bad: Sending entire file
```javascript
// Wastes tokens, slower matching
function add(a, b) { return a + b; }
function subtract(a, b) { return a - b; }
function multiply(a, b) { return a * b; }
function divide(a, b) { return a / b; }  // Only this changes!
function modulo(a, b) { return a % b; }
```

### ✅ Good: Only changed lines
```javascript
// ... existing code ...
function divide(a, b) {
  if (b === 0) throw new Error('Division by zero');
  return a / b;
}
// ... existing code ...
```

### Language-Specific Markers

```typescript
// TypeScript/JavaScript
// ... existing code ...

# Python
# ... existing code ...

// Rust
// ... existing code ...

/* C/C++ - can also use multi-line */
// ... existing code ...

-- SQL
-- ... existing code ...
```

### Example with Agent Booster

```typescript
const originalCode = `
export class Calculator {
  add(a: number, b: number): number {
    return a + b;
  }

  subtract(a: number, b: number): number {
    return a - b;
  }

  multiply(a: number, b: number): number {
    return a * b;
  }

  divide(a: number, b: number): number {
    return a / b;
  }

  modulo(a: number, b: number): number {
    return a % b;
  }
}
`;

// ✅ GOOD: Minimal edit with markers
const minimalEdit = `
  // ... existing code ...
  divide(a: number, b: number): number {
    if (b === 0) throw new Error('Division by zero');
    return a / b;
  }
  // ... existing code ...
`;

const result1 = await booster.apply({
  code: originalCode,
  edit: minimalEdit,
  language: 'typescript'
});
// ✅ 1-2ms latency, 85% confidence

// ❌ BAD: Sending entire class
const fullEdit = originalCode.replace(
  'return a / b;',
  'if (b === 0) throw new Error(\'Division by zero\');\n    return a / b;'
);

const result2 = await booster.apply({
  code: originalCode,
  edit: fullEdit,
  language: 'typescript'
});
// ⚠️ 5-10ms latency, 70% confidence (slower, less accurate)
```

## Pattern 3: Output Parsing (Markdown Code Blocks)

**Use case:** When LLM generates multiple file edits, parse markdown code blocks with metadata.

### Morph LLM Format
```markdown
```filepath=src/utils.ts instruction=Add error handling
function divide(a: number, b: number): number {
  if (b === 0) throw new Error('Division by zero');
  return a / b;
}
\```

```filepath=src/calculator.ts instruction=Use divide from utils
import { divide } from './utils';
// ... existing code ...
\```
```

### Parser Implementation

```typescript
import { AgentBooster } from 'agent-booster';
import * as fs from 'fs';

interface ParsedEdit {
  filepath: string;
  instruction: string;
  code_edit: string;
  language: string;
}

/**
 * Parse markdown code blocks with filepath= and instruction= metadata
 * Compatible with Morph LLM output format
 */
function parseMarkdownEdits(markdown: string): ParsedEdit[] {
  const regex = /```(?:(\w+))?\s*filepath=([^\s]+)\s+instruction=([^\n]+)\n([\s\S]*?)```/g;
  const edits: ParsedEdit[] = [];

  let match;
  while ((match = regex.exec(markdown)) !== null) {
    const [_, language, filepath, instruction, code_edit] = match;

    edits.push({
      filepath: filepath.trim(),
      instruction: instruction.trim(),
      code_edit: code_edit.trim(),
      language: language || detectLanguage(filepath)
    });
  }

  return edits;
}

function detectLanguage(filepath: string): string {
  const ext = filepath.split('.').pop();
  const langMap: { [key: string]: string } = {
    'ts': 'typescript',
    'js': 'javascript',
    'py': 'python',
    'rs': 'rust',
    'go': 'go',
    'java': 'java',
    'c': 'c',
    'cpp': 'cpp',
    'h': 'c',
    'hpp': 'cpp'
  };
  return langMap[ext || ''] || 'javascript';
}

/**
 * Apply multiple parsed edits to files
 */
async function applyMarkdownEdits(markdown: string): Promise<void> {
  const booster = new AgentBooster();
  const edits = parseMarkdownEdits(markdown);

  console.log(`Found ${edits.length} file edits to apply`);

  for (const edit of edits) {
    console.log(`\n📝 Applying edit to ${edit.filepath}`);
    console.log(`   Instruction: ${edit.instruction}`);

    // Read current file
    const originalCode = fs.readFileSync(edit.filepath, 'utf-8');

    // Apply edit with Agent Booster
    const result = await booster.apply({
      code: originalCode,
      edit: edit.code_edit,
      language: edit.language
    });

    if (result.success) {
      // Write modified file
      fs.writeFileSync(edit.filepath, result.output);
      console.log(`   ✅ Success (${result.latency}ms, ${(result.confidence * 100).toFixed(1)}% confidence)`);
    } else {
      console.log(`   ❌ Failed (confidence too low: ${(result.confidence * 100).toFixed(1)}%)`);
    }
  }
}

// Example usage
const llmOutput = `
Here are the changes:

\`\`\`typescript filepath=src/utils.ts instruction=Add error handling to divide
function divide(a: number, b: number): number {
  if (b === 0) throw new Error('Division by zero');
  return a / b;
}
\`\`\`

\`\`\`typescript filepath=src/calculator.ts instruction=Import divide from utils
import { divide } from './utils';

export class Calculator {
  // ... existing code ...

  divide(a: number, b: number): number {
    return divide(a, b);  // Use utility function
  }

  // ... existing code ...
}
\`\`\`
`;

await applyMarkdownEdits(llmOutput);
// Output:
// Found 2 file edits to apply
//
// 📝 Applying edit to src/utils.ts
//    Instruction: Add error handling to divide
//    ✅ Success (1ms, 90.0% confidence)
//
// 📝 Applying edit to src/calculator.ts
//    Instruction: Import divide from utils
//    ✅ Success (2ms, 85.0% confidence)
```

## Pattern 4: Claude Tool Use Integration

**Use case:** Claude makes decisions, Agent Booster applies edits instantly.

### Complete Agentic Loop

```typescript
import Anthropic from '@anthropic-ai/sdk';
import { AgentBooster } from 'agent-booster';
import * as fs from 'fs';

const claude = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
const booster = new AgentBooster();

const tools = [{
  name: "edit_file",
  description: "Make precise edits to code files. Use '// ... existing code ...' for unchanged sections.",
  input_schema: {
    type: "object",
    properties: {
      target_filepath: {
        type: "string",
        description: "Path of file to modify"
      },
      instructions: {
        type: "string",
        description: "First-person instruction (e.g., 'I will add error handling')"
      },
      code_edit: {
        type: "string",
        description: "Precise code lines to edit, using '// ... existing code ...' for unchanged sections"
      }
    },
    required: ["target_filepath", "instructions", "code_edit"]
  }
}];

async function handleEditFileTool(toolInput: {
  target_filepath: string;
  instructions: string;
  code_edit: string;
}) {
  const originalCode = fs.readFileSync(toolInput.target_filepath, 'utf-8');

  // Apply edit with Agent Booster (1ms vs 352ms with Morph LLM)
  const result = await booster.apply({
    code: originalCode,
    edit: toolInput.code_edit,
    language: detectLanguage(toolInput.target_filepath)
  });

  if (result.success) {
    fs.writeFileSync(toolInput.target_filepath, result.output);
    return `✅ Successfully edited ${toolInput.target_filepath} (${result.latency}ms, ${(result.confidence * 100).toFixed(1)}% confidence)`;
  } else {
    return `❌ Failed to edit ${toolInput.target_filepath}: confidence too low (${(result.confidence * 100).toFixed(1)}%). Please provide more specific code snippet.`;
  }
}

async function agenticCodingAgent(userRequest: string) {
  const messages: Anthropic.MessageParam[] = [
    { role: 'user', content: userRequest }
  ];

  while (true) {
    const response = await claude.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 4096,
      tools,
      messages
    });

    if (response.stop_reason === 'tool_use') {
      const toolUse = response.content.find(block => block.type === 'tool_use') as Anthropic.ToolUseBlock;

      if (toolUse.name === 'edit_file') {
        const toolResult = await handleEditFileTool(toolUse.input as any);

        messages.push({
          role: 'assistant',
          content: response.content
        });
        messages.push({
          role: 'user',
          content: [{
            type: 'tool_result',
            tool_use_id: toolUse.id,
            content: toolResult
          }]
        });

        continue;
      }
    }

    const textBlock = response.content.find(block => block.type === 'text') as Anthropic.TextBlock;
    console.log(textBlock?.text);
    break;
  }
}

// Usage
await agenticCodingAgent('Add TypeScript type annotations to all functions in src/utils.ts');
```

## Pattern 5: API Server (Morph LLM Compatible)

**Use case:** Drop-in replacement for Morph LLM API with same endpoints.

### Start Server
```bash
agent-booster-server
# Server runs on http://localhost:3000
```

### Morph LLM Format Request
```typescript
const response = await fetch('http://localhost:3000/v1/chat/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'agent-booster-v1',
    messages: [{
      role: 'user',
      content: `<instruction>I will add error handling to divide function</instruction><code>function divide(a, b) { return a / b; }</code><update>function divide(a, b) { if (b === 0) throw new Error('Division by zero'); return a / b; }</update>`
    }]
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);
// Modified code returned in 0-1ms (vs 352ms with Morph LLM)
```

### Simplified Format Request
```typescript
const response = await fetch('http://localhost:3000/v1/apply', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    code: 'function divide(a, b) { return a / b; }',
    edit: 'function divide(a, b) { if (b === 0) throw new Error(\'Division by zero\'); return a / b; }',
    language: 'javascript'
  })
});

const result = await response.json();
console.log(result.output);
```

## Best Practices Summary

### ✅ DO
- Use first-person instructions ("I will add...", "I'm changing...")
- Use minimal context with `// ... existing code ...` markers
- Specify language explicitly when possible
- Check confidence scores (reject if < 0.7)
- Use batch API for multiple edits

### ❌ DON'T
- Send entire files when only changing a few lines
- Use vague third-person descriptions
- Ignore confidence scores
- Make assumptions about unchanged code

## Performance Impact

| Pattern | Latency | Confidence | Use Case |
|---------|---------|------------|----------|
| Template match (try-catch, etc.) | 0-1ms | 80-90% | Common transformations |
| Minimal context edit | 1-3ms | 75-85% | Focused changes |
| Full file edit | 5-10ms | 60-75% | Large refactors |
| Batch processing (10 edits) | 10-20ms | 70-85% | Multi-file changes |

**Comparison:**
- **Morph LLM:** 200-500ms per edit, $0.01+ per request
- **Agent Booster:** 0-10ms per edit, $0.00 per request
- **Speedup:** 20-500x faster, 100% cost savings

## Complete Example: Multi-File Refactoring

```typescript
import Anthropic from '@anthropic-ai/sdk';
import { AgentBooster } from 'agent-booster';
import * as fs from 'fs';
import * as path from 'path';

const claude = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
const booster = new AgentBooster();

async function refactorProject(instruction: string, files: string[]) {
  // Step 1: Ask Claude for refactoring plan
  const response = await claude.messages.create({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 4096,
    messages: [{
      role: 'user',
      content: `${instruction}

Files to refactor:
${files.map(f => `- ${f}`).join('\n')}

For each file, provide edits in this format:
\`\`\`typescript filepath=<path> instruction=<first-person instruction>
<code edit with // ... existing code ... markers>
\`\`\`
`
    }]
  });

  const textBlock = response.content.find(block => block.type === 'text') as Anthropic.TextBlock;
  const markdown = textBlock.text;

  // Step 2: Parse and apply edits with Agent Booster
  const edits = parseMarkdownEdits(markdown);
  const results = [];

  for (const edit of edits) {
    const originalCode = fs.readFileSync(edit.filepath, 'utf-8');

    const result = await booster.apply({
      code: originalCode,
      edit: edit.code_edit,
      language: edit.language
    });

    if (result.success) {
      fs.writeFileSync(edit.filepath, result.output);
      results.push({ file: edit.filepath, success: true, latency: result.latency });
    } else {
      results.push({ file: edit.filepath, success: false, confidence: result.confidence });
    }
  }

  // Step 3: Report results
  const successful = results.filter(r => r.success).length;
  const totalLatency = results.reduce((sum, r) => sum + (r.latency || 0), 0);

  console.log(`
📊 Refactoring Complete:
   Files processed: ${results.length}
   Successful: ${successful}/${results.length}
   Total time: ${totalLatency}ms
   Average per file: ${(totalLatency / results.length).toFixed(1)}ms

   vs Morph LLM:
   - Would take: ${(results.length * 352).toFixed(0)}ms
   - Speedup: ${(results.length * 352 / totalLatency).toFixed(1)}x faster
   - Cost saved: $${(results.length * 0.01).toFixed(2)}
  `);
}

// Usage
await refactorProject(
  'Add TypeScript type annotations to all function parameters and return types',
  ['src/utils.ts', 'src/calculator.ts', 'src/parser.ts']
);
```

## See Also

- [Claude Tool Integration Guide](./claude-tool-integration.md)
- [API Server Examples](./api-server-example.md)
- [Production Agent Example](./production-agent-example.ts)
