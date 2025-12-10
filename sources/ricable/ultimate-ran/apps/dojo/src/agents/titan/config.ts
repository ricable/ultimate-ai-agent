/**
 * Titan Council Agent Configuration
 *
 * Defines the Council Members, their avatars, and system configurations
 * for the Neuro-Symbolic RAN Optimization Council.
 */

import {
  CouncilMember,
  CouncilRole,
  CouncilModel,
  CouncilAvatar,
  AgentIntegrationConfig
} from './types';

/**
 * Council Member Avatar Definitions
 */
export const councilAvatars: Map<string, CouncilAvatar> = new Map([
  [
    'analyst-deepseek',
    {
      name: 'The Analyst',
      role: CouncilRole.ANALYST,
      model: CouncilModel.DEEPSEEK_R1,
      color: '#3B82F6', // Blue
      icon: 'chart-line',
      description: 'Logical Analyst - Focuses on Lyapunov chaos detection and mathematical counters'
    }
  ],
  [
    'historian-gemini',
    {
      name: 'The Historian',
      role: CouncilRole.HISTORIAN,
      model: CouncilModel.GEMINI_15_PRO,
      color: '#8B5CF6', // Purple
      icon: 'book-open',
      description: 'Context Historian - Focuses on past episodes and similar scenarios'
    }
  ],
  [
    'strategist-claude',
    {
      name: 'The Strategist',
      role: CouncilRole.STRATEGIST,
      model: CouncilModel.CLAUDE_37_SONNET,
      color: '#10B981', // Green
      icon: 'brain',
      description: 'Strategic Synthesizer - Proposes concrete parameter changes'
    }
  ],
  [
    'chairman',
    {
      name: 'The Chairman',
      role: CouncilRole.CHAIRMAN,
      model: CouncilModel.CLAUDE_37_SONNET,
      color: '#F59E0B', // Amber
      icon: 'gavel',
      description: 'Council Chairman - Synthesizes consensus and calls for votes'
    }
  ]
]);

/**
 * Council Member Definitions
 */
export const councilMembers: Map<string, CouncilMember> = new Map([
  [
    'analyst-deepseek',
    {
      id: 'analyst-deepseek',
      role: CouncilRole.ANALYST,
      model: CouncilModel.DEEPSEEK_R1,
      temperature: 0.1,
      systemPrompt: 'You are the Analyst. Ignore history. Focus ONLY on current telemetry and math. Detect chaos using Lyapunov exponents and statistical analysis.',
      tools: [
        'midstream_analyze_chaos',
        'ruvector_query_topology',
        'calculate_lyapunov',
        'detect_anomalies'
      ],
      avatar: councilAvatars.get('analyst-deepseek')!
    }
  ],
  [
    'historian-gemini',
    {
      id: 'historian-gemini',
      role: CouncilRole.HISTORIAN,
      model: CouncilModel.GEMINI_15_PRO,
      temperature: 0.3,
      systemPrompt: 'You are the Historian. Query agentdb for similar vectors. Warn if this strategy failed before. Provide context from past debates.',
      tools: [
        'agentdb_query_episodes',
        'agentdb_get_reflexion',
        'vector_similarity_search',
        'retrieve_debate_history'
      ],
      avatar: councilAvatars.get('historian-gemini')!
    }
  ],
  [
    'strategist-claude',
    {
      id: 'strategist-claude',
      role: CouncilRole.STRATEGIST,
      model: CouncilModel.CLAUDE_37_SONNET,
      temperature: 0.5,
      systemPrompt: 'You are the Strategist. Synthesize inputs from the Analyst and Historian. Propose concrete RAN parameter changes with clear reasoning.',
      tools: [
        'simulate_gnn_outcome',
        'generate_parameter_set',
        'evaluate_proposal',
        'risk_assessment'
      ],
      avatar: councilAvatars.get('strategist-claude')!
    }
  ],
  [
    'chairman',
    {
      id: 'chairman',
      role: CouncilRole.CHAIRMAN,
      model: CouncilModel.CLAUDE_37_SONNET,
      temperature: 0.2,
      systemPrompt: 'You are the Chairman. Listen to the Council. Synthesize a consensus. Call for a vote if the Council is split. Ensure all voices are heard.',
      tools: [
        'call_vote',
        'synthesize_consensus',
        'request_hitl',
        'validate_decision'
      ],
      avatar: councilAvatars.get('chairman')!
    }
  ]
]);

/**
 * Debate Protocol Configuration
 */
export const debateConfig = {
  // Maximum number of critique rounds
  maxCritiqueRounds: 3,

  // Minimum consensus threshold (percentage)
  consensusThreshold: 0.66, // 2/3 majority

  // Timeout for each debate stage (ms)
  stageTimeout: {
    proposal: 30000,    // 30s
    critique: 20000,    // 20s per round
    synthesis: 15000    // 15s
  },

  // Interference levels that trigger debate
  interferenceThresholds: {
    low: -100,    // dBm
    medium: -95,
    high: -90,
    critical: -85
  },

  // Human-in-the-Loop thresholds
  hitlRequired: {
    lowConsensus: 0.5,      // < 50% consensus requires HITL
    highRisk: 'high',       // High-risk proposals require HITL
    criticalParams: [       // Changes to these always require HITL
      'tx_power_boost',
      'emergency_shutdown',
      'sector_isolation'
    ]
  }
};

/**
 * Visualization Configuration
 */
export const visualConfig = {
  // War Room UI colors (Tailwind-compatible)
  colors: {
    background: 'bg-slate-950',
    surface: 'bg-slate-900/50',
    border: 'border-slate-800',
    text: {
      primary: 'text-slate-100',
      secondary: 'text-slate-400',
      muted: 'text-slate-600'
    },
    status: {
      proposal: 'text-blue-400',
      critique: 'text-amber-400',
      synthesis: 'text-purple-400',
      consensus: 'text-green-400',
      split: 'text-red-400'
    }
  },

  // Glass morphism styling
  glassEffect: 'backdrop-blur-xl bg-white/5 border border-white/10',

  // Animation durations (ms)
  animations: {
    thinkingStep: 300,
    consensusReveal: 500,
    debateTransition: 400
  }
};

/**
 * Export the Titan Council Agent Integration Config
 */
export const titanCouncilConfig: Omit<AgentIntegrationConfig, 'agent'> = {
  name: 'Titan Council',
  icon: 'council',
  description: 'Neuro-Symbolic RAN Optimization Council - Multi-Model Deliberative AI',
  version: '7.0.0',
  tags: ['council', 'neuro-symbolic', 'ran-optimization', 'multi-model']
};
