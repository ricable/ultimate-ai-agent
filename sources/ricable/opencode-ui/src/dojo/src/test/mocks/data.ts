import type { 
  Provider, 
  Session, 
  Tool, 
  Message, 
  OpenCodeConfig, 
  ProviderMetrics,
  ToolExecution 
} from '@/lib/opencode-client'

export const mockProviders: Provider[] = [
  {
    id: 'anthropic',
    name: 'Anthropic',
    type: 'anthropic',
    models: [
      'claude-3-5-sonnet-20241022',
      'claude-3-5-haiku-20241022',
      'claude-3-opus-20240229'
    ],
    authenticated: true,
    status: 'online',
    cost_per_1k_tokens: 0.003,
    avg_response_time: 850,
    description: 'Claude AI models by Anthropic'
  },
  {
    id: 'openai',
    name: 'OpenAI',
    type: 'openai',
    models: [
      'gpt-4o',
      'gpt-4o-mini',
      'gpt-4-turbo',
      'gpt-3.5-turbo'
    ],
    authenticated: true,
    status: 'online',
    cost_per_1k_tokens: 0.002,
    avg_response_time: 750,
    description: 'GPT models by OpenAI'
  },
  {
    id: 'groq',
    name: 'Groq',
    type: 'groq',
    models: [
      'llama-3.1-70b-versatile',
      'llama-3.1-8b-instant',
      'mixtral-8x7b-32768'
    ],
    authenticated: false,
    status: 'offline',
    cost_per_1k_tokens: 0.001,
    avg_response_time: 300,
    description: 'Fast inference with Groq chips'
  }
]

export const mockSessions: Session[] = [
  {
    id: 'session-1',
    name: 'React Component Development',
    provider: 'anthropic',
    model: 'claude-3-5-sonnet-20241022',
    created_at: Date.now() - 3600000, // 1 hour ago
    updated_at: Date.now() - 1800000, // 30 min ago
    status: 'active',
    message_count: 12,
    total_cost: 0.45,
    project_path: '/tmp/session-1',
    config: {
      project_path: '/tmp/session-1',
      provider: 'anthropic',
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 8000,
      temperature: 0.7,
      enabled_tools: ['file_edit', 'bash', 'browser']
    }
  },
  {
    id: 'session-2',
    name: 'API Integration',
    provider: 'openai',
    model: 'gpt-4o',
    created_at: Date.now() - 7200000, // 2 hours ago
    updated_at: Date.now() - 3600000, // 1 hour ago
    status: 'completed',
    message_count: 8,
    total_cost: 0.23,
    project_path: '/tmp/session-2',
    config: {
      project_path: '/tmp/session-2',
      provider: 'openai',
      model: 'gpt-4o',
      max_tokens: 4000,
      temperature: 0.5,
      enabled_tools: ['file_edit', 'fetch']
    }
  },
  {
    id: 'session-3',
    name: 'Database Schema Design',
    provider: 'anthropic',
    model: 'claude-3-opus-20240229',
    created_at: Date.now() - 86400000, // 1 day ago
    updated_at: Date.now() - 43200000, // 12 hours ago
    status: 'completed',
    message_count: 25,
    total_cost: 1.89,
    project_path: '/tmp/session-3',
    config: {
      project_path: '/tmp/session-3',
      provider: 'anthropic',
      model: 'claude-3-opus-20240229',
      max_tokens: 8000,
      temperature: 0.3,
      enabled_tools: ['file_edit', 'bash', 'database']
    }
  }
]

export const mockMessages: Record<string, Message[]> = {
  'session-1': [
    {
      id: 'msg-1',
      role: 'user',
      type: 'user',
      content: 'Help me create a React component for displaying user profiles.',
      timestamp: Date.now() - 3600000,
      session_id: 'session-1',
      provider: 'anthropic',
      model: 'claude-3-5-sonnet-20241022'
    },
    {
      id: 'msg-2',
      role: 'assistant',
      type: 'assistant',
      content: 'I\'ll help you create a React component for displaying user profiles. Let me start by creating a basic component structure.',
      timestamp: Date.now() - 3500000,
      session_id: 'session-1',
      provider: 'anthropic',
      model: 'claude-3-5-sonnet-20241022',
      tool_calls: [
        {
          name: 'file_edit',
          input: {
            file: 'components/UserProfile.tsx',
            content: 'import React from "react";\n\ninterface UserProfileProps {\n  user: {\n    id: string;\n    name: string;\n    email: string;\n    avatar?: string;\n  };\n}\n\nexport const UserProfile: React.FC<UserProfileProps> = ({ user }) => {\n  return (\n    <div className="user-profile">\n      <h2>{user.name}</h2>\n      <p>{user.email}</p>\n    </div>\n  );\n};'
          }
        }
      ]
    },
    {
      id: 'msg-3',
      role: 'user',
      type: 'user',
      content: 'Great! Can you add styling and make it more visually appealing?',
      timestamp: Date.now() - 3000000,
      session_id: 'session-1',
      provider: 'anthropic',
      model: 'claude-3-5-sonnet-20241022'
    }
  ],
  'session-2': [
    {
      id: 'msg-4',
      role: 'user',
      type: 'user',
      content: 'I need to integrate with a REST API. Can you help me set up the client?',
      timestamp: Date.now() - 7200000,
      session_id: 'session-2',
      provider: 'openai',
      model: 'gpt-4o'
    },
    {
      id: 'msg-5',
      role: 'assistant',
      type: 'assistant',
      content: 'I\'ll help you create a robust API client. Let me set up a basic structure with error handling and TypeScript types.',
      timestamp: Date.now() - 7100000,
      session_id: 'session-2',
      provider: 'openai',
      model: 'gpt-4o'
    }
  ]
}

export const mockTools: Tool[] = [
  {
    id: 'file_edit',
    name: 'File Editor',
    description: 'Create, read, and edit files in the project',
    category: 'file',
    enabled: true,
    config: {
      parameters: {
        type: 'object',
        properties: {
          file: { type: 'string', description: 'File path to edit' },
          content: { type: 'string', description: 'New file content' },
          mode: { type: 'string', enum: ['create', 'update', 'append'] }
        },
        required: ['file', 'content']
      },
      approval_required: false
    }
  },
  {
    id: 'bash',
    name: 'Shell Command',
    description: 'Execute shell commands in the project directory',
    category: 'system',
    enabled: true,
    config: {
      parameters: {
        type: 'object',
        properties: {
          command: { type: 'string', description: 'Shell command to execute' },
          timeout: { type: 'number', description: 'Timeout in seconds' }
        },
        required: ['command']
      },
      approval_required: true
    }
  },
  {
    id: 'browser',
    name: 'Web Browser',
    description: 'Browse the web and fetch content from URLs',
    category: 'custom',
    enabled: true,
    config: {
      parameters: {
        type: 'object',
        properties: {
          url: { type: 'string', description: 'URL to fetch' },
          action: { type: 'string', enum: ['fetch', 'screenshot', 'search'] }
        },
        required: ['url']
      },
      approval_required: false
    }
  },
  {
    id: 'database',
    name: 'Database Query',
    description: 'Execute database queries and manage schemas',
    category: 'custom',
    enabled: true,
    config: {
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'SQL query to execute' },
          database: { type: 'string', description: 'Database name' }
        },
        required: ['query']
      },
      approval_required: true
    }
  }
]

export const mockConfig: OpenCodeConfig = {
  theme: 'opencode',
  model: 'anthropic/claude-3-5-sonnet-20241022',
  autoshare: false,
  autoupdate: true,
  providers: {
    anthropic: {
      apiKey: '***hidden***',
      customEndpoint: 'https://api.anthropic.com'
    },
    openai: {
      apiKey: '***hidden***',
      customEndpoint: 'https://api.openai.com'
    }
  },
  agents: {
    'code-reviewer': {
      model: 'anthropic/claude-3-5-sonnet-20241022',
      maxTokens: 4000,
      systemPrompt: 'You are a senior code reviewer. Analyze code for best practices, security issues, and performance optimizations.'
    }
  },
  mcp: {
    'filesystem': {
      id: 'filesystem',
      name: 'Filesystem Server',
      type: 'stdio',
      command: 'npx',
      args: ['@modelcontextprotocol/server-filesystem', '/tmp'],
      env: {},
      status: 'connected'
    }
  },
  lsp: {
    'typescript': {
      command: 'typescript-language-server',
      args: ['--stdio']
    }
  },
  keybinds: {
    'new_session': 'ctrl+n',
    'save_session': 'ctrl+s',
    'toggle_sidebar': 'ctrl+b'
  },
  shell: {
    path: '/bin/bash',
    args: ['-l']
  }
}

export const mockProviderMetrics: ProviderMetrics[] = [
  {
    provider_id: 'anthropic',
    requests: 125,
    avg_response_time: 850,
    total_cost: 2.45,
    error_rate: 0.02,
    last_24h: {
      requests: 45,
      cost: 0.89,
      avg_response_time: 920
    }
  },
  {
    provider_id: 'openai',
    requests: 89,
    avg_response_time: 750,
    total_cost: 1.89,
    error_rate: 0.01,
    last_24h: {
      requests: 32,
      cost: 0.67,
      avg_response_time: 780
    }
  },
  {
    provider_id: 'groq',
    requests: 12,
    avg_response_time: 300,
    total_cost: 0.15,
    error_rate: 0.08,
    last_24h: {
      requests: 3,
      cost: 0.05,
      avg_response_time: 250
    }
  }
]

export const mockToolExecutions: ToolExecution[] = [
  {
    id: 'exec-1',
    tool_id: 'file_edit',
    session_id: 'session-1',
    status: 'completed',
    params: {
      file: 'components/UserProfile.tsx',
      content: 'React component code...',
      mode: 'create'
    },
    result: 'File created successfully',
    created_at: Date.now() - 1800000,
    completed_at: Date.now() - 1795000
  },
  {
    id: 'exec-2',
    tool_id: 'bash',
    session_id: 'session-1',
    status: 'pending',
    params: {
      command: 'npm install react-icons',
      timeout: 30
    },
    created_at: Date.now() - 60000
  },
  {
    id: 'exec-3',
    tool_id: 'browser',
    session_id: 'session-2',
    status: 'failed',
    params: {
      url: 'https://api.example.com/docs',
      action: 'fetch'
    },
    error: 'Network timeout',
    created_at: Date.now() - 3600000,
    completed_at: Date.now() - 3580000
  }
]

// Session templates for quick session creation
export const mockSessionTemplates = [
  {
    id: 'template-1',
    name: 'React Development',
    description: 'Full-stack React application development',
    provider: 'anthropic',
    model: 'claude-3-5-sonnet-20241022',
    initial_prompt: 'I need help building a React application. Please ask me about the requirements and help me plan the architecture.',
    enabled_tools: ['file_edit', 'bash', 'browser'],
    configuration: {
      max_tokens: 8000,
      temperature: 0.7
    }
  },
  {
    id: 'template-2',
    name: 'Code Review',
    description: 'Code review and optimization',
    provider: 'openai',
    model: 'gpt-4o',
    initial_prompt: 'Please review my code for best practices, potential bugs, and optimization opportunities.',
    enabled_tools: ['file_edit'],
    configuration: {
      max_tokens: 4000,
      temperature: 0.3
    }
  },
  {
    id: 'template-3',
    name: 'API Integration',
    description: 'API development and integration',
    provider: 'anthropic',
    model: 'claude-3-5-sonnet-20241022',
    initial_prompt: 'Help me integrate with external APIs, handle authentication, and manage data flow.',
    enabled_tools: ['file_edit', 'browser', 'fetch'],
    configuration: {
      max_tokens: 6000,
      temperature: 0.5
    }
  }
]