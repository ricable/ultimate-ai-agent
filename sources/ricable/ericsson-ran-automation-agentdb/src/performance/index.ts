/**
 * Cognitive RAN Performance Monitoring System
 * Main entry point for comprehensive performance monitoring and bottleneck analysis
 */

export { PerformanceCollector } from './monitoring/PerformanceCollector';
export { BottleneckDetector } from './bottlenecks/BottleneckDetector';
export { RealTimeDashboard } from './dashboard/RealTimeDashboard';
export { CognitiveAnalytics } from './analytics/CognitiveAnalytics';
export { PerformanceReporter } from './reporting/PerformanceReporter';
export { PerformanceOrchestrator } from './orchestration/PerformanceOrchestrator';
export { PerformanceMonitoringSystem } from './PerformanceMonitoringSystem';

export { AgentDBMonitor } from '../integration/agentdb/AgentDBMonitor';

// Export types
export * from '../types/performance';

/**
 * Create and start the complete performance monitoring system
 */
export async function createPerformanceMonitoringSystem(): Promise<PerformanceOrchestrator> {
  const orchestrator = new PerformanceOrchestrator();
  await orchestrator.start();
  return orchestrator;
}

/**
 * Quick start performance monitoring with default configuration
 */
export async function quickStartPerformanceMonitoring(): Promise<{
  orchestrator: PerformanceOrchestrator;
  status: string;
  dashboard: any;
  health: any;
}> {
  console.log('🚀 Quick Starting Cognitive RAN Performance Monitoring...');

  const orchestrator = new PerformanceOrchestrator();
  await orchestrator.start();

  const status = orchestrator.getComponentStatus();
  const health = await orchestrator.getSystemHealth();
  const dashboard = orchestrator.components.dashboard.getDashboard();

  console.log('✅ Performance Monitoring System Started Successfully');
  console.log(`📊 System Health: ${health.overall} (${health.score}/100)`);
  console.log(`🎯 Active Components: ${Object.values(status.components).filter(c => c === 'initialized').length}/${Object.keys(status.components).length}`);

  return {
    orchestrator,
    status,
    dashboard,
    health
  };
}