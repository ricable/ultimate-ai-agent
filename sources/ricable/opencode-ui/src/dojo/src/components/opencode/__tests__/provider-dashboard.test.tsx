import React from 'react'
import { render, screen, waitFor } from '@/test/utils'
import { ProviderDashboard } from '../provider-dashboard'
import { useSessionStore } from '@/lib/session-store'
import { mockProviders, mockProviderMetrics } from '@/test/mocks/data'

// Mock the session store
vi.mock('@/lib/session-store')

const mockUseSessionStore = useSessionStore as vi.MockedFunction<typeof useSessionStore>

describe('ProviderDashboard', () => {
  const mockStore = {
    providers: mockProviders,
    providerMetrics: mockProviderMetrics,
    providerHealth: [
      { provider_id: "anthropic", status: "online" as const, response_time: 850, last_check: Date.now(), uptime: 99.9, region: "us-east-1" },
      { provider_id: "openai", status: "online" as const, response_time: 750, last_check: Date.now(), uptime: 98.5, region: "us-west-2" }
    ],
    isLoadingProviders: false,
    activeProvider: null,
    actions: {
      loadProviders: vi.fn(),
      loadProviderMetrics: vi.fn(),
      loadProviderHealth: vi.fn(),
      setActiveProvider: vi.fn(),
      authenticateProvider: vi.fn(),
    }
  }

  beforeEach(() => {
    mockUseSessionStore.mockReturnValue(mockStore as any)
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it('renders provider dashboard with providers list', async () => {
    render(<ProviderDashboard />)
    
    expect(screen.getByText('Provider Dashboard')).toBeInTheDocument()
    expect(screen.getByText('Provider Performance')).toBeInTheDocument()
    
    // Check that providers are displayed
    await waitFor(() => {
      expect(screen.getByText('Anthropic')).toBeInTheDocument()
      expect(screen.getByText('OpenAI')).toBeInTheDocument()
    })
  })

  it('displays provider metrics correctly', async () => {
    render(<ProviderDashboard />)
    
    await waitFor(() => {
      // Check for metric displays
      expect(screen.getByText(/125/)).toBeInTheDocument() // Anthropic requests
      expect(screen.getByText(/89/)).toBeInTheDocument() // OpenAI requests
      expect(screen.getByText(/\$2\.45/)).toBeInTheDocument() // Anthropic cost
      expect(screen.getByText(/\$1\.89/)).toBeInTheDocument() // OpenAI cost
    })
  })

  it('shows provider health status', async () => {
    render(<ProviderDashboard />)
    
    await waitFor(() => {
      // Check for health indicators
      const healthIndicators = screen.getAllByText(/online/i)
      expect(healthIndicators.length).toBeGreaterThan(0)
    })
  })

  it('handles provider authentication', async () => {
    const mockAuthenticateProvider = vi.fn().mockResolvedValue(true)
    mockUseSessionStore.mockReturnValue({
      ...mockStore,
      actions: {
        ...mockStore.actions,
        authenticateProvider: mockAuthenticateProvider
      }
    } as any)

    render(<ProviderDashboard />)
    
    // Find and click on a provider that needs authentication
    const groqProvider = screen.getByText('Groq')
    expect(groqProvider).toBeInTheDocument()
    
    // The authentication button should be available for unauthenticated providers
    const authButtons = screen.getAllByText(/authenticate/i)
    expect(authButtons.length).toBeGreaterThan(0)
  })

  it('displays loading state', () => {
    mockUseSessionStore.mockReturnValue({
      ...mockStore,
      isLoadingProviders: true
    } as any)

    render(<ProviderDashboard />)
    
    expect(screen.getByText(/loading/i)).toBeInTheDocument()
  })

  it('shows error state when no providers available', () => {
    mockUseSessionStore.mockReturnValue({
      ...mockStore,
      providers: [],
      isLoadingProviders: false
    } as any)

    render(<ProviderDashboard />)
    
    expect(screen.getByText(/no providers/i)).toBeInTheDocument()
  })

  it('filters providers by status', async () => {
    render(<ProviderDashboard />)
    
    // Check that we can filter by status
    const statusFilter = screen.getByRole('combobox', { name: /filter by status/i })
    expect(statusFilter).toBeInTheDocument()
    
    // Check that online providers are shown by default
    await waitFor(() => {
      expect(screen.getByText('Anthropic')).toBeInTheDocument()
      expect(screen.getByText('OpenAI')).toBeInTheDocument()
    })
  })

  it('handles provider selection', async () => {
    const mockSetActiveProvider = vi.fn()
    mockUseSessionStore.mockReturnValue({
      ...mockStore,
      actions: {
        ...mockStore.actions,
        setActiveProvider: mockSetActiveProvider
      }
    } as any)

    render(<ProviderDashboard />)
    
    // Click on a provider card
    const anthropicCard = screen.getByText('Anthropic').closest('div')
    expect(anthropicCard).toBeInTheDocument()
    
    if (anthropicCard) {
      await waitFor(() => {
        anthropicCard.click()
      })
      
      expect(mockSetActiveProvider).toHaveBeenCalledWith('anthropic')
    }
  })

  it('displays provider costs and usage', async () => {
    render(<ProviderDashboard />)
    
    await waitFor(() => {
      // Check for cost information
      expect(screen.getByText(/\$0\.003/)).toBeInTheDocument() // Anthropic cost per 1k tokens
      expect(screen.getByText(/\$0\.002/)).toBeInTheDocument() // OpenAI cost per 1k tokens
      
      // Check for response times
      expect(screen.getByText(/850ms/)).toBeInTheDocument() // Anthropic response time
      expect(screen.getByText(/750ms/)).toBeInTheDocument() // OpenAI response time
    })
  })

  it('has accessible elements', async () => {
    const { container } = render(<ProviderDashboard />)
    
    // Check for proper ARIA labels and structure
    expect(screen.getByRole('main')).toBeInTheDocument()
    
    // Check that provider cards are accessible
    const providerCards = screen.getAllByRole('article')
    expect(providerCards.length).toBeGreaterThan(0)
    
    // Run accessibility check
    const { checkAccessibility } = await import('@/test/utils')
    const results = await checkAccessibility(container)
    expect(results).toHaveNoViolations()
  })
})