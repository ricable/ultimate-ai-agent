/**
 * Agentic Flow Multi-Provider Configuration
 * Supports Claude Code PRO MAX, Google AI Pro, E2B, and OpenRouter
 */

import { AgenticFlowConfig } from 'agentic-flow';

export interface MultiProviderConfig {
  providers: {
    anthropic?: {
      apiKey?: string;
      oauthToken?: string; // Preferred: Claude Code OAuth token
      model: string;
      maxTokens?: number;
      useSubscription?: boolean; // Indicates subscription vs API key
    };
    google?: {
      apiKey?: string;
      oauthClientId?: string; // Preferred: OAuth for AI Pro/Ultra subscription
      oauthClientSecret?: string;
      model: string;
      temperature?: number;
      useSubscription?: boolean; // Indicates subscription vs API key
    };
    openrouter?: {
      apiKey: string;
      model: string;
      siteUrl?: string;
      appName?: string;
    };
  };
  e2b?: {
    apiKey: string;
    timeout?: number;
    template?: string;
  };
  strategy: 'single' | 'consensus' | 'parallel' | 'cascade';
  primaryProvider: 'anthropic' | 'google' | 'openrouter';
  runtime: 'local' | 'devpod' | 'cloud';
}

/**
 * Validate subscription-only mode
 * Throws error if pay-per-token API keys are detected
 *
 * IMPORTANT: This is ENFORCED BY DEFAULT. Set SUBSCRIPTION_ONLY_MODE=false
 * to disable (NOT RECOMMENDED for production).
 */
const enforceSubscriptionOnlyMode = (): void => {
  // Default to subscription-only mode unless explicitly disabled
  const subscriptionOnly = process.env.SUBSCRIPTION_ONLY_MODE !== 'false';
  const rejectApiKeys = process.env.REJECT_API_KEY_AUTH !== 'false';

  if (subscriptionOnly || rejectApiKeys) {
    // Check for blocked API keys
    if (process.env.ANTHROPIC_API_KEY) {
      throw new Error(
        '\n' +
        '╔═══════════════════════════════════════════════════════════════════╗\n' +
        '║  ❌ BLOCKED: ANTHROPIC_API_KEY detected!                          ║\n' +
        '╠═══════════════════════════════════════════════════════════════════╣\n' +
        '║                                                                   ║\n' +
        '║  This project uses SUBSCRIPTION-ONLY mode.                        ║\n' +
        '║  Pay-per-token API keys are NOT allowed.                          ║\n' +
        '║                                                                   ║\n' +
        '║  FIX: Remove ANTHROPIC_API_KEY from config/.env                   ║\n' +
        '║                                                                   ║\n' +
        '║  USE INSTEAD:                                                     ║\n' +
        '║    1. Run: claude login                                           ║\n' +
        '║    2. Run: claude setup-token                                     ║\n' +
        '║    3. Set: CLAUDE_CODE_OAUTH_TOKEN=<your-token>                   ║\n' +
        '║                                                                   ║\n' +
        '║  Or run: npm run auth:setup                                       ║\n' +
        '║                                                                   ║\n' +
        '╚═══════════════════════════════════════════════════════════════════╝\n'
      );
    }

    if (process.env.GOOGLE_AI_API_KEY) {
      throw new Error(
        '\n' +
        '╔═══════════════════════════════════════════════════════════════════╗\n' +
        '║  ❌ BLOCKED: GOOGLE_AI_API_KEY detected!                          ║\n' +
        '╠═══════════════════════════════════════════════════════════════════╣\n' +
        '║                                                                   ║\n' +
        '║  This project uses SUBSCRIPTION-ONLY mode.                        ║\n' +
        '║  Pay-per-token API keys are NOT allowed.                          ║\n' +
        '║                                                                   ║\n' +
        '║  FIX: Remove GOOGLE_AI_API_KEY from config/.env                   ║\n' +
        '║                                                                   ║\n' +
        '║  USE INSTEAD:                                                     ║\n' +
        '║    1. Subscribe: https://gemini.google/subscriptions/             ║\n' +
        '║    2. Setup OAuth in Google Cloud Console                         ║\n' +
        '║    3. Set: GOOGLE_OAUTH_CLIENT_ID=<client-id>                     ║\n' +
        '║    4. Set: GOOGLE_OAUTH_CLIENT_SECRET=<secret>                    ║\n' +
        '║                                                                   ║\n' +
        '║  Or run: npm run auth:setup                                       ║\n' +
        '║                                                                   ║\n' +
        '╚═══════════════════════════════════════════════════════════════════╝\n'
      );
    }

    // Note: OPENROUTER_API_KEY and E2B_API_KEY are ALLOWED
    // They provide additional functionality without replacing subscriptions

    console.log('✅ Subscription-only mode active - Claude/Gemini API keys blocked');
    console.log('   ✓ Anthropic: Using Claude Code OAuth subscription');
    console.log('   ✓ Google: Using Google AI OAuth subscription');
    if (process.env.OPENROUTER_API_KEY) {
      console.log('   ✓ OpenRouter: API key configured (allowed)');
    }
    if (process.env.E2B_API_KEY) {
      console.log('   ✓ E2B: Sandbox API configured (allowed)');
    }
  } else {
    console.warn('⚠️  WARNING: Subscription-only mode is DISABLED');
    console.warn('   Pay-per-token API keys are allowed but NOT recommended.');
  }
};

export const getMultiProviderConfig = (): MultiProviderConfig => {
  // Enforce subscription-only mode first
  enforceSubscriptionOnlyMode();

  const config: MultiProviderConfig = {
    providers: {},
    strategy: (process.env.AGENTIC_FLOW_STRATEGY as any) || 'consensus',
    primaryProvider: (process.env.AGENTIC_FLOW_PRIMARY_PROVIDER as any) || 'anthropic',
    runtime: (process.env.RUNTIME_MODE as any) || 'local'
  };

  // Claude Code Subscription Configuration (SUBSCRIPTION ONLY)
  // OAuth token for container/CI environments, or auto-auth when running in Claude Code CLI
  const anthropicOAuthToken = process.env.CLAUDE_CODE_OAUTH_TOKEN;
  const anthropicModel = process.env.ANTHROPIC_MODEL || 'claude-opus-4-5-20251101'; // Latest: Opus 4.5

  // Always configure Anthropic for subscription mode
  // When running within Claude Code, authentication is automatic
  config.providers.anthropic = {
    oauthToken: anthropicOAuthToken,
    model: anthropicModel,
    maxTokens: 8192,
    useSubscription: true // Always subscription mode
  };

  if (anthropicOAuthToken) {
    console.log('✅ Claude: Using OAuth token (subscription)');
  } else if (process.env.CLAUDECODE === '1') {
    console.log('✅ Claude: Using Claude Code CLI authentication (subscription)');
  } else {
    console.warn('⚠️  Claude: No OAuth token set. Run `claude login` or set CLAUDE_CODE_OAUTH_TOKEN');
  }

  // Google AI Pro/Ultra Subscription Configuration (SUBSCRIPTION ONLY)
  // OAuth credentials required - no API key fallback
  const googleOAuthClientId = process.env.GOOGLE_OAUTH_CLIENT_ID;
  const googleOAuthClientSecret = process.env.GOOGLE_OAUTH_CLIENT_SECRET;
  const googleModel = process.env.GOOGLE_AI_MODEL || 'gemini-3-pro-preview'; // Latest: Gemini 3 Pro

  if (googleOAuthClientId && googleOAuthClientSecret) {
    config.providers.google = {
      oauthClientId: googleOAuthClientId,
      oauthClientSecret: googleOAuthClientSecret,
      model: googleModel,
      temperature: 0.7,
      useSubscription: true // Always subscription mode
    };
    console.log('✅ Google: Using OAuth credentials (AI Pro subscription)');
  } else {
    console.warn(
      '⚠️  Google: OAuth not configured. Set GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_CLIENT_SECRET.\n' +
      '   Follow setup guide in config/.env or docs/AUTH.md'
    );
  }

  // OpenRouter Configuration (ALLOWED - optional for additional models)
  if (process.env.OPENROUTER_API_KEY) {
    config.providers.openrouter = {
      apiKey: process.env.OPENROUTER_API_KEY,
      model: process.env.OPENROUTER_MODEL || 'anthropic/claude-3.5-sonnet',
      siteUrl: 'https://titan-ran.ericsson.com',
      appName: 'TITAN RAN Optimizer'
    };
    console.log('✅ OpenRouter: API configured (additional models available)');
  }

  // E2B Configuration (free tier available)
  if (process.env.E2B_API_KEY && process.env.E2B_API_KEY !== 'e2b_your-key-here') {
    config.e2b = {
      apiKey: process.env.E2B_API_KEY,
      timeout: parseInt(process.env.E2B_TIMEOUT || '300'),
      template: process.env.E2B_TEMPLATE || 'base'
    };
    console.log('✅ E2B: Sandbox API configured');
  }

  return config;
};

/**
 * Agentic Flow Configuration for TITAN
 */
export const agenticFlowConfig: AgenticFlowConfig = {
  // Transport Layer (QUIC with 0-RTT)
  transport: {
    protocol: 'quic',
    enableZeroRTT: true,
    congestionControl: 'bbr',
    maxStreams: 100
  },

  // Memory Layer (AgentDB + HNSW)
  memory: {
    provider: 'agentdb',
    persistence: process.env.AGENTDB_PERSISTENCE === 'true',
    vectorIndex: {
      algorithm: 'hnsw',
      dimensions: 1536,
      metric: 'cosine'
    }
  },

  // Agent Configuration
  agents: {
    maxConcurrent: 10,
    timeout: 30000,
    retryStrategy: {
      maxRetries: 3,
      backoff: 'exponential'
    }
  },

  // Security
  security: {
    quantumResistant: process.env.QUANTUM_SIGNATURES === 'true',
    sandboxIsolation: process.env.SANDBOX_ISOLATION as any || 'strict',
    auditLog: true
  },

  // Platform-specific optimizations
  platform: {
    arch: process.env.PLATFORM_ARCH || 'arm64',
    optimizations: {
      simd: true,
      vectorization: true,
      neonIntrinsics: process.env.PLATFORM_ARCH === 'arm64'
    }
  }
};

export default agenticFlowConfig;
