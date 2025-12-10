/**
 * Type definitions for ReasoningBank Benchmark Suite
 */

export interface BenchmarkConfig {
  iterations: number;
  parallelAgents: number;
  scenarios: string[];
  enableWarmStart: boolean;
  memorySize: number;
  outputFormats: ('json' | 'markdown' | 'csv')[];
  anthropicApiKey?: string;
}

export interface Task {
  id: string;
  name: string;
  description: string;
  domain: string;
  input: string;
  expectedOutput?: string;
  successCriteria: (result: TaskResult) => boolean;
  difficulty: 'easy' | 'medium' | 'hard';
}

export interface TaskResult {
  taskId: string;
  agentType: 'baseline' | 'reasoningbank';
  success: boolean;
  output: string;
  tokens: number;
  latency: number;
  memoriesUsed?: number;
  memoriesCreated?: number;
  confidence?: number;
  error?: string;
  trajectory?: any;
}

export interface BenchmarkScenario {
  name: string;
  description: string;
  domain: string;
  tasks: Task[];
  warmStartMemories?: any[];
  metrics: MetricType[];
}

export type MetricType =
  | 'success_rate'
  | 'token_efficiency'
  | 'latency'
  | 'memory_usage'
  | 'learning_velocity'
  | 'accuracy'
  | 'generalization';

export interface ScenarioResults {
  scenarioName: string;
  baseline: AgentMetrics;
  reasoningbank: AgentMetrics;
  improvement: ImprovementMetrics;
  learningCurve: LearningPoint[];
  timestamp: string;
}

export interface AgentMetrics {
  successRate: number;
  totalTasks: number;
  successfulTasks: number;
  avgTokens: number;
  totalTokens: number;
  avgLatency: number;
  totalLatency: number;
  memoriesUsed?: number;
  memoriesCreated?: number;
  avgConfidence?: number;
  errors: string[];
}

export interface ImprovementMetrics {
  successRateDelta: string;
  successRatePercent: number;
  tokenEfficiency: string;
  tokenSavings: number;
  latencyOverhead: string;
  latencyDelta: number;
  learningVelocity?: number;
}

export interface LearningPoint {
  iteration: number;
  baselineSuccess: number;
  reasoningbankSuccess: number;
  baselineTokens: number;
  reasoningbankTokens: number;
  memoriesAvailable: number;
}

export interface BenchmarkReport {
  summary: {
    totalScenarios: number;
    totalTasks: number;
    overallImprovement: ImprovementMetrics;
    executionTime: number;
  };
  scenarios: ScenarioResults[];
  recommendations: string[];
  generatedAt: string;
}

export interface Agent {
  type: 'baseline' | 'reasoningbank';
  executeTask(task: Task): Promise<TaskResult>;
  reset(): Promise<void>;
  getMemoryStats?(): Promise<{ total: number; avgConfidence: number }>;
}
