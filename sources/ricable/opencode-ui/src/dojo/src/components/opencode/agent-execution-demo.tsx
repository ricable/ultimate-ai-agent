import React from "react";
import { StreamMessage } from "./stream-message";
/**
 * Demo component showing all the different message types and tools for OpenCode
 * Adapted from Claudia's AgentExecutionDemo for multi-provider support
 */
export const AgentExecutionDemo: React.FC = () => {
  // Sample messages based on OpenCode's multi-provider architecture
  const messages: any[] = [
    // Skip meta message (should not render)
    {
      type: "user",
      isMeta: true,
      message: { content: [] },
      timestamp: "2025-06-29T14:08:53.771Z",
      provider: "anthropic"
    },
    
    // Summary message
    {
      leafUuid: "3c5ecb4f-c1f0-40c2-a357-ab7642ad28b8",
      summary: "OpenCode Multi-Provider Session Configuration and Setup",
      type: "summary" as any,
      provider: "anthropic"
    },
    
    // Assistant with Edit tool (Anthropic provider)
    {
      type: "assistant",
      provider: "anthropic",
      model: "claude-3.5-sonnet-20241022",
      message: {
        content: [{
          type: "tool_use",
          name: "Edit",
          input: {
            file_path: "/Users/user/dev/my-app/src/components/App.tsx",
            new_string: "const handleSubmit = () => { console.log('Form submitted'); };",
            old_string: "const handleSubmit = () => { console.log('Submit'); };"
          }
        }],
        usage: { input_tokens: 150, output_tokens: 75 }
      }
    },
    
    // User with Edit tool result
    {
      type: "user",
      provider: "anthropic",
      message: {
        content: [{
          type: "tool_result",
          content: `The file /Users/user/dev/my-app/src/components/App.tsx has been updated. Here's the result of running \`cat -n\` on a snippet of the edited file:
   15       const App = () => {
   16         const [data, setData] = useState(null);
   17         
   18         const handleSubmit = () => { console.log('Form submitted'); };
   19         
   20         return (
   21           <div className="app">
   22             <h1>My App</h1>
   23           </div>
   24         );
   25       };`
        }]
      }
    },
    
    // Assistant with OpenAI provider and different model
    {
      type: "assistant",
      provider: "openai",
      model: "gpt-4o-2024-08-06",
      message: {
        content: [{
          type: "tool_use",
          name: "bash",
          input: {
            command: "npm test",
            description: "Run the test suite"
          }
        }],
        usage: { input_tokens: 200, output_tokens: 45 }
      }
    },
    
    // Tool result from bash command
    {
      type: "user",
      provider: "openai",
      message: {
        content: [{
          type: "tool_result",
          content: `> my-app@1.0.0 test
> jest

 PASS  src/components/App.test.tsx
  ✓ renders without crashing (23ms)
  ✓ handles form submission (15ms)

Test Suites: 1 passed, 1 total
Tests:       2 passed, 2 total
Snapshots:   0 total
Time:        1.234s

Ran all test suites.`
        }]
      }
    },
    
    // Assistant with Groq provider (faster model)
    {
      type: "assistant",
      provider: "groq",
      model: "llama-3.1-70b-versatile",
      message: {
        content: [{
          type: "tool_use",
          name: "write",
          input: {
            file_path: "/Users/user/dev/my-app/README.md",
            content: `# My App

A modern React application built with TypeScript and Vite.

## Features

- Fast development with Vite
- Type safety with TypeScript
- Modern React patterns
- Comprehensive testing with Jest

## Getting Started

\`\`\`bash
npm install
npm run dev
\`\`\`

## Testing

\`\`\`bash
npm test
\`\`\`

## Building

\`\`\`bash
npm run build
\`\`\`
`
          }
        }],
        usage: { input_tokens: 100, output_tokens: 180 }
      }
    },
    
    // Local provider example (Ollama)
    {
      type: "assistant",
      provider: "ollama",
      model: "llama3.2:3b",
      message: {
        content: [{
          type: "text",
          text: "I've updated the README file with comprehensive documentation. The application now has proper setup instructions, feature descriptions, and usage guidelines. All tests are passing, and the code is well-structured."
        }],
        usage: { input_tokens: 250, output_tokens: 45 }
      }
    }
  ];

  return (
    <div className="max-w-4xl mx-auto p-8 space-y-4">
      <div className="mb-6">
        <h1 className="text-2xl font-bold">OpenCode Multi-Provider Demo</h1>
        <p className="text-muted-foreground mt-2">
          Demonstration of OpenCode&apos;s multi-provider AI system with tool execution across different providers
        </p>
      </div>
      
      {messages.map((message, idx) => (
        <StreamMessage key={idx} message={message} streamMessages={messages} />
      ))}
    </div>
  );
};