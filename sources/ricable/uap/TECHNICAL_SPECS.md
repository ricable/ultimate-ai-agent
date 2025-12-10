# UAP Technical Specifications

## Component Interface Specifications

### 1. Backend Service Interfaces

#### 1.1 Agent Orchestrator Interface

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from fastapi import WebSocket

class AgentOrchestrator(ABC):
    """Abstract base class for agent orchestration"""
    
    @abstractmethod
    async def initialize_services(self) -> None:
        """Initialize all framework services"""
        pass
    
    @abstractmethod
    def register_connection(self, conn_id: str, websocket: WebSocket) -> None:
        """Register a WebSocket connection"""
        pass
    
    @abstractmethod
    def unregister_connection(self, conn_id: str) -> None:
        """Unregister a WebSocket connection"""
        pass
    
    @abstractmethod
    async def handle_agui_event(self, conn_id: str, event: Dict[str, Any]) -> None:
        """Process AG-UI protocol events"""
        pass
    
    @abstractmethod
    async def handle_http_chat(
        self, 
        agent_id: str, 
        message: str, 
        framework: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle HTTP-based chat requests"""
        pass
    
    @abstractmethod
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        pass
```

#### 1.2 Framework Manager Interface

```python
from enum import Enum
from pydantic import BaseModel
from typing import List, Optional, Union

class MessageType(Enum):
    TEXT = "text"
    IMAGE = "image"
    DOCUMENT = "document"
    TOOL_CALL = "tool_call"

class AgentMessage(BaseModel):
    type: MessageType
    content: str
    metadata: Optional[Dict[str, Any]] = {}
    timestamp: Optional[datetime] = None

class AgentResponse(BaseModel):
    content: str
    agent_id: str
    framework: str
    metadata: Dict[str, Any]
    tools_called: List[str] = []
    confidence_score: Optional[float] = None

class BaseFrameworkManager(ABC):
    """Abstract base class for all framework managers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_initialized = False
        self.health_status = "unknown"
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the framework manager"""
        pass
    
    @abstractmethod
    async def process_message(
        self, 
        message: AgentMessage, 
        context: Dict[str, Any]
    ) -> AgentResponse:
        """Process a message and return response"""
        pass
    
    @abstractmethod
    async def stream_response(
        self, 
        message: AgentMessage, 
        context: Dict[str, Any]
    ) -> AsyncIterator[AgentResponse]:
        """Stream response chunks for real-time updates"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of framework capabilities"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Return framework health and performance metrics"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Gracefully shutdown the framework"""
        pass
```

#### 1.3 CopilotKit Manager Implementation

```python
from copilotkit import CopilotKit
from typing import AsyncIterator

class CopilotKitManager(BaseFrameworkManager):
    """CopilotKit framework implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.copilot_client = None
        self.active_sessions = {}
    
    async def initialize(self) -> bool:
        """Initialize CopilotKit client"""
        try:
            self.copilot_client = CopilotKit(
                api_key=self.config.get("api_key"),
                base_url=self.config.get("base_url", "https://api.copilotkit.ai")
            )
            self.is_initialized = True
            self.health_status = "healthy"
            return True
        except Exception as e:
            self.health_status = f"error: {str(e)}"
            return False
    
    async def process_message(
        self, 
        message: AgentMessage, 
        context: Dict[str, Any]
    ) -> AgentResponse:
        """Process message with CopilotKit"""
        session_id = context.get("session_id", "default")
        
        response = await self.copilot_client.chat.completions.create(
            messages=[{"role": "user", "content": message.content}],
            stream=False,
            context=context
        )
        
        return AgentResponse(
            content=response.choices[0].message.content,
            agent_id=context.get("agent_id", "copilot-agent"),
            framework="copilot",
            metadata={
                "model": response.model,
                "usage": response.usage.dict() if response.usage else {},
                "session_id": session_id
            }
        )
    
    async def stream_response(
        self, 
        message: AgentMessage, 
        context: Dict[str, Any]
    ) -> AsyncIterator[AgentResponse]:
        """Stream CopilotKit response"""
        stream = await self.copilot_client.chat.completions.create(
            messages=[{"role": "user", "content": message.content}],
            stream=True,
            context=context
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield AgentResponse(
                    content=chunk.choices[0].delta.content,
                    agent_id=context.get("agent_id", "copilot-agent"),
                    framework="copilot",
                    metadata={"chunk": True, "timestamp": datetime.utcnow()}
                )
    
    def get_capabilities(self) -> List[str]:
        return [
            "text_generation",
            "conversation",
            "code_assistance", 
            "ui_components",
            "real_time_collaboration"
        ]
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "status": self.health_status,
            "initialized": self.is_initialized,
            "active_sessions": len(self.active_sessions),
            "capabilities": self.get_capabilities(),
            "version": getattr(self.copilot_client, "version", "unknown")
        }
```

### 2. Frontend Component Interfaces

#### 2.1 AG-UI Hook Interface

```typescript
interface AGUIConfig {
  endpoint: string;
  transport: 'websocket' | 'sse';
  reconnect: boolean;
  maxReconnectAttempts?: number;
  reconnectDelay?: number;
  authentication?: {
    token?: string;
    type: 'bearer' | 'query' | 'header';
  };
}

interface AGUIHookReturn {
  messages: AGUIEvent[];
  isConnected: boolean;
  connectionState: 'connecting' | 'connected' | 'disconnected' | 'error';
  sendMessage: (content: string, metadata?: Record<string, any>) => Promise<void>;
  sendEvent: (event: AGUIEvent) => Promise<void>;
  clearMessages: () => void;
  reconnect: () => void;
  disconnect: () => void;
  error?: Error;
}

export function useAGUI(agentId: string, config?: Partial<AGUIConfig>): AGUIHookReturn;
```

#### 2.2 Agent Card Component Interface

```typescript
interface AgentCardProps {
  id: string;
  name: string;
  description: string;
  framework: 'copilot' | 'agno' | 'mastra';
  capabilities?: string[];
  avatar?: string;
  status?: 'online' | 'offline' | 'busy';
  onInteractionStart?: (agentId: string) => void;
  onInteractionEnd?: (agentId: string) => void;
  className?: string;
  maxHeight?: number;
  showStatus?: boolean;
  enableVoice?: boolean;
}

interface AgentCardState {
  isExpanded: boolean;
  messages: AGUIEvent[];
  inputValue: string;
  isTyping: boolean;
  error?: string;
}

export function AgentCard(props: AgentCardProps): JSX.Element;
```

#### 2.3 Message Component Interface

```typescript
interface MessageProps {
  message: AGUIEvent;
  isUser: boolean;
  timestamp?: Date;
  avatar?: string;
  showAvatar?: boolean;
  className?: string;
}

interface MessageContentProps {
  content: string;
  type: 'text' | 'markdown' | 'code' | 'json';
  language?: string;
  maxHeight?: number;
}

export function Message(props: MessageProps): JSX.Element;
export function MessageContent(props: MessageContentProps): JSX.Element;
```

### 3. AG-UI Protocol Specification

#### 3.1 Event Types and Payloads

```typescript
// User-initiated events
interface UserMessageEvent {
  type: 'user_message';
  content: string;
  metadata: {
    framework?: string;
    session_id?: string;
    context?: Record<string, any>;
  };
}

interface UserActionEvent {
  type: 'user_action';
  action: 'like' | 'dislike' | 'copy' | 'regenerate' | 'stop';
  message_id: string;
  metadata?: Record<string, any>;
}

// Agent-initiated events
interface TextMessageContentEvent {
  type: 'text_message_content';
  content: string;
  metadata: {
    agent_id: string;
    framework: string;
    message_id: string;
    is_complete: boolean;
    timestamp: string;
  };
}

interface ToolCallStartEvent {
  type: 'tool_call_start';
  tool_name: string;
  tool_args: Record<string, any>;
  metadata: {
    agent_id: string;
    call_id: string;
    timestamp: string;
  };
}

interface ToolCallEndEvent {
  type: 'tool_call_end';
  tool_name: string;
  result: any;
  success: boolean;
  metadata: {
    agent_id: string;
    call_id: string;
    duration_ms: number;
    timestamp: string;
  };
}

interface StateDeltaEvent {
  type: 'state_delta';
  delta: Record<string, any>;
  metadata: {
    agent_id: string;
    state_version: number;
    timestamp: string;
  };
}

// System events
interface ConnectionEvent {
  type: 'connection_open' | 'connection_close' | 'connection_error';
  metadata: {
    connection_id: string;
    timestamp: string;
    error?: string;
  };
}

type AGUIEvent = 
  | UserMessageEvent 
  | UserActionEvent 
  | TextMessageContentEvent 
  | ToolCallStartEvent 
  | ToolCallEndEvent 
  | StateDeltaEvent 
  | ConnectionEvent;
```

#### 3.2 Protocol State Machine

```typescript
type ConnectionState = 
  | 'disconnected'
  | 'connecting' 
  | 'connected'
  | 'reconnecting'
  | 'error';

interface ProtocolState {
  connectionState: ConnectionState;
  lastMessage?: AGUIEvent;
  messageHistory: AGUIEvent[];
  pendingMessages: AGUIEvent[];
  error?: Error;
  reconnectAttempts: number;
  connectionId?: string;
}

class AGUIProtocol {
  private state: ProtocolState;
  private websocket?: WebSocket;
  private eventHandlers: Map<string, Set<Function>>;
  
  connect(endpoint: string, config: AGUIConfig): Promise<void>;
  disconnect(): void;
  send(event: AGUIEvent): Promise<void>;
  on(eventType: string, handler: Function): void;
  off(eventType: string, handler: Function): void;
  getState(): ProtocolState;
}
```

### 4. Database Schema Specifications

#### 4.1 User and Session Management

```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    avatar_url TEXT,
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Sessions table
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    agent_id VARCHAR(100) NOT NULL,
    framework VARCHAR(50) NOT NULL,
    session_data JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Messages table
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES user_sessions(id) ON DELETE CASCADE,
    message_type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    is_user_message BOOLEAN NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_agent_id ON user_sessions(agent_id);
CREATE INDEX idx_messages_session_id ON messages(session_id);
CREATE INDEX idx_messages_created_at ON messages(created_at);
```

#### 4.2 Agent Configuration and Metrics

```sql
-- Agent configurations
CREATE TABLE agent_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(100) UNIQUE NOT NULL,
    framework VARCHAR(50) NOT NULL,
    config JSONB NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Agent metrics
CREATE TABLE agent_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC NOT NULL,
    metadata JSONB DEFAULT '{}',
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- System events log
CREATE TABLE system_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(100) NOT NULL,
    component VARCHAR(100) NOT NULL,
    event_data JSONB NOT NULL,
    severity VARCHAR(20) DEFAULT 'info',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### 5. Configuration Specifications

#### 5.1 Framework Configuration Schema

```typescript
interface FrameworkConfig {
  name: string;
  type: 'copilot' | 'agno' | 'mastra';
  enabled: boolean;
  config: {
    apiKey?: string;
    baseUrl?: string;
    model?: string;
    temperature?: number;
    maxTokens?: number;
    timeout?: number;
    retries?: number;
    [key: string]: any;
  };
  capabilities: string[];
  routing: {
    priority: number;
    keywords: string[];
    patterns: string[];
  };
}

interface SystemConfig {
  server: {
    host: string;
    port: number;
    cors: {
      origins: string[];
      credentials: boolean;
    };
  };
  websocket: {
    maxConnections: number;
    heartbeatInterval: number;
    messageTimeout: number;
  };
  frameworks: FrameworkConfig[];
  security: {
    jwtSecret: string;
    jwtExpiration: string;
    rateLimiting: {
      requests: number;
      window: string;
    };
  };
  logging: {
    level: 'debug' | 'info' | 'warn' | 'error';
    format: 'json' | 'text';
    outputs: string[];
  };
}
```

#### 5.2 Environment Configuration

```bash
# Backend Configuration
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
BACKEND_WORKERS=4
BACKEND_LOG_LEVEL=info

# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost:5432/uap
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis Configuration (for caching and sessions)
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=20

# Framework API Keys (managed by Teller)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
COHERE_API_KEY=...

# Security Configuration
JWT_SECRET=your-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRATION=86400

# Frontend Configuration
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
VITE_ENVIRONMENT=development
VITE_SENTRY_DSN=...

# Monitoring and Observability
SENTRY_DSN=...
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001
```

### 6. Testing Specifications

#### 6.1 Backend Test Structure

```python
import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

class TestAgentOrchestrator:
    @pytest.fixture
    def orchestrator(self):
        return UAP_AgentOrchestrator()
    
    @pytest.fixture
    def mock_websocket(self):
        websocket = AsyncMock()
        websocket.send_text = AsyncMock()
        websocket.receive_text = AsyncMock()
        return websocket
    
    async def test_framework_routing(self, orchestrator):
        # Test intelligent framework selection
        message = "Analyze this document for key insights"
        framework = await orchestrator._select_framework(message, {})
        assert framework == "agno"
        
        message = "Help me with customer support workflow"
        framework = await orchestrator._select_framework(message, {})
        assert framework == "mastra"
    
    async def test_websocket_lifecycle(self, orchestrator, mock_websocket):
        conn_id = "test-connection-123"
        
        # Test connection registration
        orchestrator.register_connection(conn_id, mock_websocket)
        assert conn_id in orchestrator.active_connections
        
        # Test message handling
        event = {
            "type": "user_message",
            "content": "Hello agent",
            "metadata": {"framework": "copilot"}
        }
        await orchestrator.handle_agui_event(conn_id, event)
        
        # Verify response was sent
        mock_websocket.send_text.assert_called_once()
        
        # Test disconnection
        orchestrator.unregister_connection(conn_id)
        assert conn_id not in orchestrator.active_connections

class TestFrameworkManagers:
    @pytest.mark.asyncio
    async def test_copilot_manager(self):
        config = {"api_key": "test-key", "model": "gpt-4"}
        manager = CopilotKitManager(config)
        
        # Test initialization
        success = await manager.initialize()
        assert success == True
        assert manager.is_initialized == True
        
        # Test message processing
        message = AgentMessage(
            type=MessageType.TEXT,
            content="Hello, how can you help me?"
        )
        response = await manager.process_message(message, {})
        
        assert isinstance(response, AgentResponse)
        assert response.framework == "copilot"
        assert response.content != ""
```

#### 6.2 Frontend Test Structure

```typescript
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { AgentCard } from '@/components/agents/AgentCard';
import { useAGUI } from '@/hooks/useAGUI';

// Mock the useAGUI hook
vi.mock('@/hooks/useAGUI');

describe('AgentCard', () => {
  const mockUseAGUI = useAGUI as vi.MockedFunction<typeof useAGUI>;
  
  beforeEach(() => {
    mockUseAGUI.mockReturnValue({
      messages: [],
      isConnected: true,
      sendMessage: vi.fn(),
      error: undefined
    });
  });

  it('renders agent information correctly', () => {
    render(
      <AgentCard
        id="test-agent"
        name="Test Agent"
        description="A test agent for unit testing"
        framework="copilot"
      />
    );

    expect(screen.getByText('Test Agent')).toBeInTheDocument();
    expect(screen.getByText('A test agent for unit testing')).toBeInTheDocument();
    expect(screen.getByText('copilot')).toBeInTheDocument();
  });

  it('sends messages when user types and presses send', async () => {
    const mockSendMessage = vi.fn();
    mockUseAGUI.mockReturnValue({
      messages: [],
      isConnected: true,
      sendMessage: mockSendMessage,
      error: undefined
    });

    render(
      <AgentCard
        id="test-agent"
        name="Test Agent"
        description="A test agent"
        framework="copilot"
      />
    );

    const input = screen.getByPlaceholderText('Chat with agent...');
    const sendButton = screen.getByText('Send');

    fireEvent.change(input, { target: { value: 'Hello agent' } });
    fireEvent.click(sendButton);

    await waitFor(() => {
      expect(mockSendMessage).toHaveBeenCalledWith('Hello agent', 'copilot');
    });
  });

  it('displays connection status correctly', () => {
    mockUseAGUI.mockReturnValue({
      messages: [],
      isConnected: false,
      sendMessage: vi.fn(),
      error: undefined
    });

    render(
      <AgentCard
        id="test-agent"
        name="Test Agent"
        description="A test agent"
        framework="copilot"
      />
    );

    const statusIndicator = screen.getByRole('status');
    expect(statusIndicator).toHaveClass('bg-red-500');
  });
});

describe('useAGUI Hook', () => {
  it('establishes WebSocket connection and handles messages', async () => {
    // This would require a more complex test setup with WebSocket mocking
    // Implementation depends on testing library choices
  });
});
```

### 7. Deployment and Operations

#### 7.1 Docker Compose for Development

```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: backend/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/uap
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./backend:/app/backend
    command: uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

  frontend:
    build:
      context: .
      dockerfile: frontend/Dockerfile
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app/frontend
      - /app/frontend/node_modules
    command: npm run dev

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=uap
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

#### 7.2 Kubernetes Deployment Manifests

```yaml
# backend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: uap-backend
  labels:
    app: uap-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: uap-backend
  template:
    metadata:
      labels:
        app: uap-backend
    spec:
      containers:
      - name: backend
        image: uap/backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: uap-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: uap-secrets
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: uap-backend-service
spec:
  selector:
    app: uap-backend
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

This technical specification provides the detailed interfaces and implementation patterns needed to build the UAP system according to the architectural design.