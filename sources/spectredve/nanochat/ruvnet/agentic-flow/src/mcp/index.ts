/**
 * Medical MCP Server - Main Entry Point
 * Exports all components for both SSE and STDIO transports
 */

// Types
export * from './types';

// Tools
export { MedicalAnalyzeTool } from './tools/medical-analyze';
export { MedicalVerifyTool } from './tools/medical-verify';
export { ProviderNotifyTool } from './tools/provider-notify';
export { ConfidenceScoreTool } from './tools/confidence-score';
export { CitationVerifyTool } from './tools/citation-verify';
export { KnowledgeSearchTool } from './tools/knowledge-search';

// Anti-hallucination features
export { ConfidenceMonitor } from './anti-hallucination/confidence-monitor';
export { CitationValidator } from './anti-hallucination/citation-validator';
export { ProviderWorkflow } from './anti-hallucination/provider-workflow';
export { EmergencyEscalationHandler } from './anti-hallucination/emergency-escalation';

// AgentDB integration
export { AgentDBIntegration } from './agentdb-integration';

// Transports (for programmatic use)
// Note: For CLI usage, run the transport files directly:
// - SSE: node src/mcp/transports/sse.js
// - STDIO: node src/mcp/transports/stdio.js
