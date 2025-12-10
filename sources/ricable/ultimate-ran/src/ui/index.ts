/**
 * TITAN Dashboard - Main Export
 * AG-UI + OpenTUI Dashboard for RAN Optimization
 *
 * @module ui
 * @version 7.0.0-alpha.1
 */

// Core Dashboard
export { TitanDashboard } from './titan-dashboard.js';
export { TitanAPIServer } from './api-server.js';

// Types
export type {
  CellStatus,
  CellKPI,
  InterferenceMatrix,
  FMAlarm,
  PMCounter,
  OptimizationEvent,
  ParameterChange,
  KPIImpact,
  ApprovalRequest,
  SafetyCheck,
  ApprovalResponse,
  DashboardState,
  AGUIEvent,
  GenerativeUIComponent,
  TableColumn,
  ChartDataPoint,
  CommandPaletteAction
} from './types.js';

// React Components (for frontend use)
// Note: These are TSX files and require a React environment
export { InterferenceHeatmap } from './components/InterferenceHeatmap.js';
export { OptimizationTimeline } from './components/OptimizationTimeline.js';
export { ApprovalCard } from './components/ApprovalCard.js';
export { ParameterSlider, ParameterControlPanel } from './components/ParameterSlider.js';

// Re-export AG-UI Server from agui module
export { AGUIServer } from '../agui/server.js';

// AI Integrations (Claude Agent SDK + Google ADK + Swarm Coordinator)
export {
  ClaudeAgentIntegration,
  GoogleADKIntegration,
  AIOrchestrator,
  AISwarmCoordinator,
  runExamples as runAIIntegrationExamples
} from './integrations/index.js';

export type {
  ClaudeAgentConfig,
  GoogleADKConfig,
  AIOrchestrationConfig,
  OptimizationResult,
  SwarmConfig,
  SwarmAgent,
  ConsensusResult,
  SwarmTask
} from './integrations/index.js';
