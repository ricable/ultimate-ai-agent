/**
 * Vitest Global Setup - London School TDD Infrastructure
 * TITAN Neuro-Symbolic RAN Platform
 *
 * This setup file provides mock factories and test utilities following
 * London School TDD (mockist) approach.
 */

import { beforeAll, afterAll, beforeEach, afterEach, vi } from 'vitest';
import type { Mock } from 'vitest';

/**
 * Global test configuration
 */
export const TEST_CONFIG = {
  // Performance thresholds from PRD
  vectorSearchLatencyP95: 10, // ms
  gnnRmseTarget: 2, // dB
  councilConsensusTime: 5000, // ms
  coverageTarget: 80, // %

  // 3GPP compliance ranges
  p0Range: { min: -130, max: -70 }, // dBm
  alphaRange: { min: 0.0, max: 1.0 },

  // Test timeouts
  defaultTimeout: 5000,
  integrationTimeout: 30000,
};

/**
 * Global mocks registry
 */
export const globalMocks = {
  agentdb: null as Mock | null,
  ruvector: null as Mock | null,
  e2bSandbox: null as Mock | null,
  deepseekLLM: null as Mock | null,
  geminiLLM: null as Mock | null,
  claudeLLM: null as Mock | null,
  midstream: null as Mock | null,
  strangeLoops: null as Mock | null,
};

/**
 * Initialize all global mocks before tests
 */
beforeAll(() => {
  // Mock console to reduce noise
  global.console = {
    ...console,
    log: vi.fn(),
    debug: vi.fn(),
    info: vi.fn(),
    warn: console.warn, // Keep warnings
    error: console.error, // Keep errors
  };
});

/**
 * Reset all mocks between tests
 */
beforeEach(() => {
  vi.clearAllMocks();
});

/**
 * Cleanup after all tests
 */
afterAll(() => {
  vi.restoreAllMocks();
});

/**
 * Test helper: Create mock PM counters
 */
export function createMockPMCounters(overrides = {}) {
  return {
    // Uplink counters
    pmUlSinrMean: 10.5,
    pmUlBler: 0.02,
    pmPuschPrbUsage: 65.5,
    pmUlRssi: -95.0,

    // Downlink counters
    pmDlSinrMean: 12.0,
    pmDlBler: 0.01,
    pmPdschPrbUsage: 70.0,

    // Accessibility
    pmRrcConnEstabSucc: 950,
    pmRrcConnEstabAtt: 1000,
    pmCssr: 0.98,
    pmErabEstabSuccQci: { 1: 100, 5: 200, 9: 150 },

    // Retainability
    pmErabRelNormal: 900,
    pmErabRelAbnormal: 10,
    pmCallDropRate: 0.011,

    ...overrides,
  };
}

/**
 * Test helper: Create mock FM alarm
 */
export function createMockFMAlarm(overrides = {}) {
  return {
    alarmId: `alarm-${Date.now()}`,
    alarmType: 'communicationsAlarm',
    probableCause: 'thresholdCrossed',
    specificProblem: 'UL SINR degradation',
    perceivedSeverity: 'MAJOR' as const,
    severity: 'major' as const,
    managedObject: 'NRCELL_001',
    managedObjectInstance: 'SubNetwork=1,MeContext=gNB001,ManagedElement=1,GNBDUFunction=1,NRCellDU=NRCELL_001',
    eventTime: new Date(),
    ackState: 'UNACKNOWLEDGED' as const,
    rootCauseIndicator: false,
    ...overrides,
  };
}

/**
 * Test helper: Create mock cell node for GNN
 */
export function createMockCellNode(overrides = {}) {
  return {
    cellId: 'NRCELL_001',
    features: [10.5, -95.0, 65.5, 12.0], // [SINR, RSRP, PRB usage, CQI]
    p0: -106,
    alpha: 0.8,
    embedding: new Array(768).fill(0).map(() => Math.random()),
    ...overrides,
  };
}

/**
 * Test helper: Create mock interference edge
 */
export function createMockInterferenceEdge(overrides = {}) {
  return {
    fromCell: 'NRCELL_001',
    toCell: 'NRCELL_002',
    features: [500, 0.15, 85] as [number, number, number],
    distance: 500, // meters
    overlapPct: 0.15,
    interferenceCoupling: 85, // dB
    ...overrides,
  };
}

/**
 * Test helper: Create mock learning episode
 */
export function createMockLearningEpisode(overrides = {}) {
  return {
    id: `episode-${Date.now()}`,
    cellId: 'NRCELL_001',
    startTime: Date.now() - 300000,
    endTime: Date.now(),
    pmBefore: createMockPMCounters(),
    pmAfter: createMockPMCounters({ pmUlSinrMean: 12.5 }),
    cmChange: { p0NominalPUSCH: -103 },
    fmAlarms: [],
    outcome: 'SUCCESS' as const,
    reward: 0.87,
    embedding: new Array(768).fill(0).map(() => Math.random()),
    ...overrides,
  };
}

/**
 * Test helper: Create mock HNSW search result
 */
export function createMockSearchResult(overrides = {}) {
  return {
    id: `result-${Date.now()}`,
    score: 0.95,
    item: createMockLearningEpisode(),
    distance: 0.05,
    ...overrides,
  };
}

/**
 * Test helper: Sleep for testing async operations
 */
export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Test helper: Assert performance threshold
 */
export function assertLatency(actualMs: number, thresholdMs: number, operation: string) {
  if (actualMs > thresholdMs) {
    throw new Error(
      `Performance threshold exceeded for ${operation}: ${actualMs}ms > ${thresholdMs}ms`
    );
  }
}

/**
 * Test helper: Generate random 768-dim embedding
 */
export function generateEmbedding(seed?: number): number[] {
  const rng = seed !== undefined ? () => Math.sin(seed++) : Math.random;
  return new Array(768).fill(0).map(() => rng());
}

/**
 * Test helper: Calculate cosine similarity
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) throw new Error('Vectors must have same dimension');

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}
