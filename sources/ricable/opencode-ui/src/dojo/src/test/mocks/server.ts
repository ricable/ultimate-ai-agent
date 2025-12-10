import { setupServer } from 'msw/node'
import { http, HttpResponse } from 'msw'
import { mockProviders, mockSessions, mockTools, mockConfig, mockMessages, mockProviderMetrics } from './data'

export const handlers = [
  // Health check
  http.get('/api/health', () => {
    return HttpResponse.json({
      status: 'ok',
      version: '2.1.0',
      timestamp: Date.now(),
      uptime: 123456
    })
  }),

  // Test endpoint
  http.get('/api/test', () => {
    return HttpResponse.json({
      success: true,
      message: 'Test endpoint'
    })
  }),

  // Connection test
  http.get('/api/connection/test', () => {
    return HttpResponse.json({
      success: true,
      latency: 150,
      timestamp: Date.now()
    })
  }),

  // Providers
  http.get('/api/providers', () => {
    return HttpResponse.json(mockProviders)
  }),

  http.post('/api/providers/auth', async ({ request }) => {
    const credentials = await request.json() as any as any
    
    return HttpResponse.json({
      success: true,
      provider_id: credentials?.providerId || 'unknown',
      authenticated: true
    })
  }),

  http.post('/api/providers/:providerId/auth', async ({ params, request }) => {
    const { providerId } = params
    const credentials = await request.json() as any
    
    return HttpResponse.json({
      success: true,
      provider_id: providerId,
      authenticated: true
    })
  }),

  http.get('/api/providers/metrics', () => {
    return HttpResponse.json(mockProviderMetrics)
  }),

  http.get('/api/providers/health', () => {
    return HttpResponse.json([
      { provider_id: "anthropic", status: "online", response_time: 850, last_check: Date.now(), uptime: 99.9, region: "us-east-1" },
      { provider_id: "openai", status: "online", response_time: 750, last_check: Date.now(), uptime: 98.5, region: "us-west-2" },
      { provider_id: "groq", status: "degraded", response_time: 1200, last_check: Date.now(), uptime: 95.2, region: "us-central" }
    ])
  }),

  // Sessions
  http.get('/api/sessions', () => {
    return HttpResponse.json(mockSessions)
  }),

  http.post('/api/sessions', async ({ request }) => {
    const config = await request.json() as any
    const newSession = {
      id: `session-${Date.now()}`,
      name: config.name || 'New Session',
      provider: config.provider,
      model: config.model,
      created_at: Date.now(),
      updated_at: Date.now(),
      status: 'active',
      message_count: 0,
      total_cost: 0,
      config,
    }
    
    return HttpResponse.json(newSession)
  }),

  http.get('/api/sessions/:sessionId', ({ params }) => {
    const { sessionId } = params
    const session = mockSessions.find(s => s.id === sessionId)
    
    if (!session) {
      return new HttpResponse(null, { status: 404 })
    }
    
    return HttpResponse.json(session)
  }),

  http.delete('/api/sessions/:sessionId', ({ params }) => {
    const { sessionId } = params
    return HttpResponse.json({ success: true })
  }),

  http.get('/api/sessions/:sessionId/messages', ({ params }) => {
    const { sessionId } = params
    return HttpResponse.json(mockMessages[sessionId as string] || [])
  }),

  http.post('/api/sessions/:sessionId/message', async ({ params, request }) => {
    const { sessionId } = params
    const { content } = await request.json() as any
    
    return HttpResponse.json({
      messageId: `msg-${Date.now()}`,
      sessionId,
      content
    })
  }),

  http.post('/api/sessions/:sessionId/messages', async ({ params, request }) => {
    const { sessionId } = params
    const { content } = await request.json() as any
    
    return HttpResponse.json({
      messageId: `msg-${Date.now()}`,
      sessionId,
      content
    })
  }),

  http.post('/api/sessions/:sessionId/share', ({ params }) => {
    const { sessionId } = params
    return HttpResponse.json({
      url: `https://share.opencode.ai/${sessionId}`,
      expires_at: Date.now() + 86400000,
      password_protected: false,
      view_count: 0
    })
  }),

  // Tools
  http.get('/api/tools', () => {
    return HttpResponse.json(mockTools)
  }),

  http.post('/api/tools/:toolId/execute', async ({ params, request }) => {
    const { toolId } = params
    const input = await request.json() as any
    
    return HttpResponse.json({
      success: true,
      result: `Tool ${toolId} executed successfully`,
      execution_time: 150,
      tool_id: toolId,
      params: input
    })
  }),

  http.get('/api/tools/executions', ({ request }) => {
    const url = new URL(request.url)
    const sessionId = url.searchParams.get('session_id')
    
    return HttpResponse.json([
      {
        id: 'exec-1',
        tool_id: 'file_edit',
        session_id: sessionId || 'session-1',
        status: 'completed',
        params: { file: 'test.txt', content: 'hello' },
        result: 'File updated successfully',
        created_at: Date.now() - 60000,
        completed_at: Date.now() - 30000
      }
    ])
  }),

  http.post('/api/tools/executions/:executionId/approve', ({ params }) => {
    return HttpResponse.json({ success: true })
  }),

  http.post('/api/tools/executions/:executionId/cancel', ({ params }) => {
    return HttpResponse.json({ success: true })
  }),

  // Configuration
  http.get('/api/config', () => {
    return HttpResponse.json(mockConfig)
  }),

  http.post('/api/config', async ({ request }) => {
    const config = await request.json() as any
    return HttpResponse.json({ success: true })
  }),

  http.post('/api/config/validate', async ({ request }) => {
    const config = await request.json() as any
    return HttpResponse.json({
      valid: true,
      errors: [],
      warnings: []
    })
  }),

  // Usage statistics
  http.get('/api/usage/stats', () => {
    return HttpResponse.json({
      total_sessions: 45,
      total_messages: 1234,
      total_cost: 12.45,
      today: {
        sessions: 5,
        messages: 123,
        cost: 2.34
      },
      this_week: {
        sessions: 23,
        messages: 567,
        cost: 6.78
      },
      this_month: {
        sessions: 45,
        messages: 1234,
        cost: 12.45
      }
    })
  }),

  // LSP servers
  http.get('/api/lsp/servers', () => {
    return HttpResponse.json([
      {
        id: 'typescript',
        name: 'TypeScript Language Server',
        command: 'typescript-language-server',
        status: 'running',
        port: 6009
      }
    ])
  }),

  // Diagnostics
  http.get('/api/diagnostics', () => {
    return HttpResponse.json([
      {
        file_path: '/test/file.ts',
        line: 10,
        column: 5,
        severity: 'error',
        message: 'Type error: Expected string, got number'
      }
    ])
  }),

  // Custom commands
  http.get('/api/commands', () => {
    return HttpResponse.json([
      {
        id: 'custom-build',
        name: 'Custom Build',
        command: 'npm run build',
        description: 'Build the project'
      }
    ])
  })
]

export const server = setupServer(...handlers)