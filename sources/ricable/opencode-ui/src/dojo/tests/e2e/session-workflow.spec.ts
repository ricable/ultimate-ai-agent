import { test, expect } from '@playwright/test'

test.describe('Session Management Workflow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    
    // Wait for the app to load
    await expect(page.getByText('OpenCode Desktop')).toBeVisible()
  })

  test('user can create and interact with a session', async ({ page }) => {
    // 1. Start on projects page
    await expect(page.getByRole('heading', { name: 'Projects' })).toBeVisible()
    
    // 2. Click new session button
    await page.getByRole('button', { name: /new session/i }).click()
    
    // 3. Fill out session creation form
    await page.getByLabel(/session name/i).fill('E2E Test Session')
    
    // Select provider
    await page.getByLabel(/provider/i).click()
    await page.getByText('Anthropic').click()
    
    // Select model
    await page.getByLabel(/model/i).click()
    await page.getByText('claude-3-5-sonnet-20241022').click()
    
    // Create session
    await page.getByRole('button', { name: /create session/i }).click()
    
    // 4. Verify session was created and we're in session view
    await expect(page.getByText('E2E Test Session')).toBeVisible()
    await expect(page.getByPlaceholder(/type your message/i)).toBeVisible()
    
    // 5. Send a message
    const messageInput = page.getByPlaceholder(/type your message/i)
    await messageInput.fill('Hello, can you help me with React components?')
    await page.getByRole('button', { name: /send/i }).click()
    
    // 6. Verify message was sent
    await expect(page.getByText('Hello, can you help me with React components?')).toBeVisible()
    
    // 7. Wait for response (mock should respond)
    await expect(page.getByText(/I'll help you/)).toBeVisible({ timeout: 10000 })
  })

  test('user can share a session', async ({ page }) => {
    // Navigate to an existing session
    await page.getByText('React Component Development').click()
    
    // Click share button
    await page.getByRole('button', { name: /share/i }).click()
    
    // Verify share dialog appears
    await expect(page.getByText(/share session/i)).toBeVisible()
    
    // Copy share link
    await page.getByRole('button', { name: /copy link/i }).click()
    
    // Verify success message
    await expect(page.getByText(/link copied/i)).toBeVisible()
  })

  test('user can delete a session with confirmation', async ({ page }) => {
    // Navigate to session actions
    await page.getByText('React Component Development').hover()
    await page.getByRole('button', { name: /more actions/i }).click()
    
    // Click delete
    await page.getByRole('menuitem', { name: /delete session/i }).click()
    
    // Confirm deletion
    await expect(page.getByText(/are you sure/i)).toBeVisible()
    await page.getByRole('button', { name: /delete/i }).click()
    
    // Verify session is gone
    await expect(page.getByText('React Component Development')).not.toBeVisible()
  })

  test('session list displays correctly', async ({ page }) => {
    // Should show all sessions
    await expect(page.getByText('React Component Development')).toBeVisible()
    await expect(page.getByText('API Integration')).toBeVisible()
    await expect(page.getByText('Database Schema Design')).toBeVisible()
    
    // Should show session metadata
    await expect(page.getByText('anthropic')).toBeVisible()
    await expect(page.getByText('12 messages')).toBeVisible()
    await expect(page.getByText('$0.45')).toBeVisible()
  })

  test('search and filter sessions', async ({ page }) => {
    // Search for specific session
    const searchInput = page.getByPlaceholder(/search sessions/i)
    await searchInput.fill('React')
    
    // Should only show matching sessions
    await expect(page.getByText('React Component Development')).toBeVisible()
    await expect(page.getByText('API Integration')).not.toBeVisible()
    
    // Clear search
    await searchInput.clear()
    
    // Filter by provider
    await page.getByRole('combobox', { name: /filter by provider/i }).click()
    await page.getByText('Anthropic').click()
    
    // Should only show Anthropic sessions
    await expect(page.getByText('React Component Development')).toBeVisible()
    await expect(page.getByText('Database Schema Design')).toBeVisible()
  })
})

test.describe('Provider Management', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    
    // Navigate to providers tab
    await page.getByRole('tab', { name: /providers/i }).click()
  })

  test('displays provider dashboard', async ({ page }) => {
    await expect(page.getByText('Provider Dashboard')).toBeVisible()
    
    // Should show provider cards
    await expect(page.getByText('Anthropic')).toBeVisible()
    await expect(page.getByText('OpenAI')).toBeVisible()
    await expect(page.getByText('Groq')).toBeVisible()
    
    // Should show provider metrics
    await expect(page.getByText(/\$2\.45/)).toBeVisible() // Total cost
    await expect(page.getByText(/850ms/)).toBeVisible() // Response time
  })

  test('user can authenticate a provider', async ({ page }) => {
    // Find unauthenticated provider (Groq)
    await page.getByText('Groq').locator('..').getByRole('button', { name: /authenticate/i }).click()
    
    // Fill in credentials
    await page.getByLabel(/api key/i).fill('test-groq-api-key')
    
    // Submit authentication
    await page.getByRole('button', { name: /save/i }).click()
    
    // Should show success or error message
    await expect(page.getByText(/authentication/i)).toBeVisible()
  })

  test('displays provider health status', async ({ page }) => {
    // Should show health indicators
    await expect(page.getByText(/online/i)).toBeVisible()
    await expect(page.getByText(/response time/i)).toBeVisible()
    await expect(page.getByText(/uptime/i)).toBeVisible()
  })

  test('shows provider cost breakdown', async ({ page }) => {
    // Click on provider for detailed view
    await page.getByText('Anthropic').click()
    
    // Should show detailed cost information
    await expect(page.getByText(/cost per 1k tokens/i)).toBeVisible()
    await expect(page.getByText(/total requests/i)).toBeVisible()
    await expect(page.getByText(/error rate/i)).toBeVisible()
  })
})

test.describe('Tool System', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    
    // Navigate to tools tab
    await page.getByRole('tab', { name: /tools/i }).click()
  })

  test('displays available tools', async ({ page }) => {
    await expect(page.getByText('Available Tools')).toBeVisible()
    
    // Should show tool categories
    await expect(page.getByText('File Operations')).toBeVisible()
    await expect(page.getByText('System')).toBeVisible()
    await expect(page.getByText('Web')).toBeVisible()
    
    // Should show individual tools
    await expect(page.getByText('File Editor')).toBeVisible()
    await expect(page.getByText('Shell Command')).toBeVisible()
    await expect(page.getByText('Web Browser')).toBeVisible()
  })

  test('user can execute a tool', async ({ page }) => {
    // Click on File Editor tool
    await page.getByText('File Editor').click()
    
    // Should show tool execution form
    await expect(page.getByLabel(/file path/i)).toBeVisible()
    await expect(page.getByLabel(/content/i)).toBeVisible()
    
    // Fill in parameters
    await page.getByLabel(/file path/i).fill('test.txt')
    await page.getByLabel(/content/i).fill('Hello World')
    
    // Execute tool
    await page.getByRole('button', { name: /run tool/i }).click()
    
    // Should show execution result
    await expect(page.getByText(/tool executed successfully/i)).toBeVisible()
  })

  test('handles tool approval workflow', async ({ page }) => {
    // Should show pending approvals section
    await expect(page.getByText(/pending approvals/i)).toBeVisible()
    
    // Mock pending approval should be visible
    await expect(page.getByText('npm install')).toBeVisible()
    
    // Click approve button
    await page.getByRole('button', { name: /approve/i }).first().click()
    
    // Should show confirmation or success message
    await expect(page.getByText(/approved/i)).toBeVisible()
  })

  test('displays tool execution history', async ({ page }) => {
    // Should show execution history
    await expect(page.getByText(/execution history/i)).toBeVisible()
    
    // Should show past executions
    await expect(page.getByText(/file created successfully/i)).toBeVisible()
    await expect(page.getByText(/completed/i)).toBeVisible()
  })
})

test.describe('Settings and Configuration', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    
    // Navigate to settings
    await page.getByRole('button', { name: /settings/i }).click()
  })

  test('displays configuration options', async ({ page }) => {
    await expect(page.getByText('Settings')).toBeVisible()
    
    // Should show configuration sections
    await expect(page.getByText(/theme/i)).toBeVisible()
    await expect(page.getByText(/model/i)).toBeVisible()
    await expect(page.getByText(/providers/i)).toBeVisible()
  })

  test('user can change theme', async ({ page }) => {
    // Find theme selector
    await page.getByRole('combobox', { name: /theme/i }).click()
    await page.getByText('Dark').click()
    
    // Should apply dark theme
    await expect(page.locator('html')).toHaveClass(/dark/)
  })

  test('user can update configuration', async ({ page }) => {
    // Change default model
    await page.getByRole('combobox', { name: /default model/i }).click()
    await page.getByText('gpt-4o').click()
    
    // Save settings
    await page.getByRole('button', { name: /save settings/i }).click()
    
    // Should show success message
    await expect(page.getByText(/settings saved/i)).toBeVisible()
  })

  test('validates configuration', async ({ page }) => {
    // Try to set invalid configuration
    await page.getByLabel(/max tokens/i).fill('999999')
    
    // Should show validation error
    await expect(page.getByText(/invalid value/i)).toBeVisible()
  })
})

test.describe('Responsive Design', () => {
  test('works on mobile devices', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 })
    await page.goto('/')
    
    // Should show mobile-optimized layout
    await expect(page.getByRole('button', { name: /menu/i })).toBeVisible()
    
    // Open mobile menu
    await page.getByRole('button', { name: /menu/i }).click()
    
    // Should show navigation options
    await expect(page.getByText('Projects')).toBeVisible()
    await expect(page.getByText('Providers')).toBeVisible()
    await expect(page.getByText('Tools')).toBeVisible()
  })

  test('adapts to tablet size', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 })
    await page.goto('/')
    
    // Should show tablet layout with collapsible sidebar
    await expect(page.getByRole('button', { name: /toggle sidebar/i })).toBeVisible()
  })
})

test.describe('Accessibility', () => {
  test('has proper heading structure', async ({ page }) => {
    await page.goto('/')
    
    // Should have proper heading hierarchy
    await expect(page.getByRole('heading', { level: 1 })).toBeVisible()
    await expect(page.getByRole('heading', { level: 2 })).toBeVisible()
  })

  test('supports keyboard navigation', async ({ page }) => {
    await page.goto('/')
    
    // Tab through main navigation
    await page.keyboard.press('Tab')
    await expect(page.getByRole('tab', { name: /projects/i })).toBeFocused()
    
    await page.keyboard.press('Tab')
    await expect(page.getByRole('tab', { name: /providers/i })).toBeFocused()
    
    // Use arrow keys
    await page.keyboard.press('ArrowLeft')
    await expect(page.getByRole('tab', { name: /projects/i })).toBeFocused()
    
    // Activate with Enter
    await page.keyboard.press('Enter')
    await expect(page.getByText('Projects')).toBeVisible()
  })

  test('has proper ARIA labels', async ({ page }) => {
    await page.goto('/')
    
    // Check for important ARIA labels
    await expect(page.getByRole('main')).toBeVisible()
    await expect(page.getByRole('navigation')).toBeVisible()
    await expect(page.getByLabelText(/search/i)).toBeVisible()
  })
})