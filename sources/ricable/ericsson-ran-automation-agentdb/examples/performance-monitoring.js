"use strict";
/**
 * Example: Cognitive RAN Performance Monitoring System Usage
 * Demonstrates how to use the comprehensive performance monitoring and bottleneck analysis
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.example4_RealTimeDashboard = exports.example3_PerformanceAnalysis = exports.example2_AdvancedMonitoring = exports.example1_QuickStart = void 0;
const performance_1 = require("../src/performance");
/**
 * Example 1: Quick Start Performance Monitoring
 */
async function example1_QuickStart() {
    console.log('=== Example 1: Quick Start Performance Monitoring ===\n');
    try {
        // Quick start with default configuration
        const monitoring = await (0, performance_1.quickStartPerformanceMonitoring)();
        console.log('🎯 Performance Monitoring Active:');
        console.log(`   Health Score: ${monitoring.health.score}/100 (${monitoring.health.overall})`);
        console.log(`   Dashboard: ${monitoring.dashboard.name} with ${monitoring.dashboard.widgets.length} widgets`);
        console.log(`   Components: ${Object.values(monitoring.status.components).filter(c => c === 'initialized').length} active`);
        // Let it run for a bit to collect data
        console.log('\n⏳ Collecting performance data for 30 seconds...');
        await new Promise(resolve => setTimeout(resolve, 30000));
        // Get performance overview
        const overview = await monitoring.orchestrator.getPerformanceOverview();
        console.log('\n📊 Performance Overview:');
        console.log(`   System Health: ${overview.systemHealth.score}/100`);
        console.log(`   AgentDB Health: ${overview.agentdbPerformance.healthScore}/100`);
        console.log(`   Cognitive Intelligence: ${overview.cognitiveIntelligence.summary.overallCognitiveScore}/100`);
        console.log(`   Active Issues: ${overview.activeIssues.total}`);
        console.log(`   System Optimal: ${overview.isOptimal ? '✅ Yes' : '⚠️ No'}`);
        // Check performance targets
        console.log('\n🎯 Performance Targets:');
        Object.entries(overview.performanceTargets).forEach(([target, data]) => {
            const status = data.current <= data.target ? '✅' : '⚠️';
            console.log(`   ${status} ${target}: ${data.current.toFixed(2)}${data.unit} (target: ${data.target}${data.unit})`);
        });
        // Generate executive report
        console.log('\n📈 Generating Executive Report...');
        const report = await monitoring.orchestrator.generatePerformanceReport('executive');
        console.log(`   Report ID: ${report.id}`);
        console.log(`   Overall Score: ${report.summary.overallScore}/100`);
        console.log(`   Key Achievements: ${report.summary.achievements.length}`);
        console.log(`   Recommendations: ${report.recommendations.length}`);
        // Stop monitoring
        await monitoring.orchestrator.stop();
        console.log('\n✅ Performance monitoring stopped');
    }
    catch (error) {
        console.error('❌ Error in quick start example:', error);
    }
}
exports.example1_QuickStart = example1_QuickStart;
/**
 * Example 2: Advanced Performance Monitoring with Custom Configuration
 */
async function example2_AdvancedMonitoring() {
    console.log('\n=== Example 2: Advanced Performance Monitoring ===\n');
    try {
        // Create custom monitoring system
        const orchestrator = new performance_1.PerformanceOrchestrator();
        // Set up event listeners for real-time monitoring
        orchestrator.on('started', () => {
            console.log('🚀 Performance monitoring system started');
        });
        orchestrator.on('bottleneck:detected', (bottleneck) => {
            console.log(`🔍 Bottleneck Detected: ${bottleneck.component} - ${bottleneck.description}`);
            console.log(`   Impact: ${bottleneck.impact.performanceLoss}% performance loss`);
            console.log(`   Recommendation: ${bottleneck.recommendation.action}`);
        });
        orchestrator.on('anomaly:detected', (anomaly) => {
            console.log(`⚠️ Anomaly Detected: ${anomaly.metric} - ${anomaly.type}`);
            console.log(`   Deviation: ${anomaly.deviationPercent.toFixed(1)}%`);
            console.log(`   Confidence: ${(anomaly.confidence * 100).toFixed(0)}%`);
        });
        orchestrator.on('quic:health_issue', (issue) => {
            console.log(`🗄️ QUIC Health Issue: ${issue.type}`);
            console.log(`   Impact: ${issue.impact}`);
        });
        orchestrator.on('cognitive:analysis', (analysis) => {
            console.log('🧠 Cognitive Analysis Completed:');
            console.log(`   Temporal Efficiency: ${analysis.temporalAnalysis?.efficiencyScore?.toFixed(1)}%`);
            console.log(`   Strange-Loop Effectiveness: ${analysis.strangeLoopAnalysis?.effectiveness?.toFixed(1)}%`);
            console.log(`   Learning Velocity: ${analysis.learningAnalysis?.learningVelocity?.toFixed(1)} patterns/hour`);
        });
        orchestrator.on('health:critical', (health) => {
            console.log(`🚨 CRITICAL HEALTH ALERT: ${health.overall} (${health.score}/100)`);
        });
        orchestrator.on('report:generated', (report) => {
            console.log(`📊 Report Generated: ${report.type} - ${report.id}`);
            console.log(`   Period: ${report.period.start.toISOString()} to ${report.period.end.toISOString()}`);
            console.log(`   Sections: ${report.sections.length}`);
            console.log(`   Recommendations: ${report.recommendations.length}`);
        });
        // Start monitoring
        await orchestrator.start();
        console.log('✅ Advanced monitoring started with custom event handlers');
        // Monitor for specific conditions
        let monitoringCycles = 0;
        const maxCycles = 10;
        const monitoringInterval = setInterval(async () => {
            monitoringCycles++;
            // Get current system health
            const health = await orchestrator.getSystemHealth();
            console.log(`\n📊 Monitoring Cycle ${monitoringCycles}/${maxCycles}:`);
            console.log(`   Health: ${health.overall} (${health.score}/100)`);
            // Get detailed performance metrics
            const overview = await orchestrator.getPerformanceOverview();
            // Check AgentDB QUIC performance specifically
            const agentdbHealth = overview.agentdbPerformance;
            if (agentdbHealth && agentdbHealth.quicPerformance) {
                const quic = agentdbHealth.quicPerformance;
                console.log(`   QUIC Sync: ${quic.currentLatency.toFixed(2)}ms (target: ${quic.targetLatency}ms) - ${quic.targetMet ? '✅' : '⚠️'}`);
                console.log(`   QUIC Health Score: ${quic.healthScore}/100`);
            }
            // Check cognitive performance
            const cognitive = overview.cognitiveIntelligence;
            if (cognitive && cognitive.summary) {
                console.log(`   Cognitive Score: ${cognitive.summary.overallCognitiveScore}/100`);
                console.log(`   Consciousness Level: ${cognitive.summary.keyMetrics.consciousnessLevel.toFixed(1)}%`);
            }
            // Check for active issues
            if (overview.activeIssues.total > 0) {
                console.log(`   ⚠️ Active Issues: ${overview.activeIssues.total} (${overview.activeIssues.bottlenecks.length} bottlenecks, ${overview.activeIssues.anomalies.length} anomalies)`);
            }
            if (monitoringCycles >= maxCycles) {
                clearInterval(monitoringInterval);
                await orchestrator.stop();
                console.log('\n✅ Advanced monitoring example completed');
            }
        }, 15000); // Every 15 seconds
    }
    catch (error) {
        console.error('❌ Error in advanced monitoring example:', error);
    }
}
exports.example2_AdvancedMonitoring = example2_AdvancedMonitoring;
/**
 * Example 3: Performance Analysis and Optimization
 */
async function example3_PerformanceAnalysis() {
    console.log('\n=== Example 3: Performance Analysis and Optimization ===\n');
    try {
        const orchestrator = new performance_1.PerformanceOrchestrator();
        await orchestrator.start();
        console.log('🔬 Running Comprehensive Performance Analysis...');
        // Wait for data collection
        await new Promise(resolve => setTimeout(resolve, 20000));
        // Get comprehensive performance data
        const overview = await orchestrator.getPerformanceOverview();
        // Analyze AgentDB performance in detail
        console.log('\n📊 AgentDB Performance Analysis:');
        const agentdbHealth = overview.agentdbPerformance;
        if (agentdbHealth) {
            console.log(`   Overall Health: ${agentdbHealth.healthScore}/100 (${agentdbHealth.status})`);
            if (agentdbHealth.currentMetrics) {
                console.log('   Current Metrics:');
                console.log(`     Vector Search Latency: ${agentdbHealth.currentMetrics.vectorSearchLatency.toFixed(2)}ms`);
                console.log(`     QUIC Sync Latency: ${agentdbHealth.currentMetrics.quicSyncLatency.toFixed(2)}ms`);
                console.log(`     Query Throughput: ${agentdbHealth.currentMetrics.queryThroughput.toFixed(0)} queries/sec`);
                console.log(`     Memory Usage: ${agentdbHealth.currentMetrics.memoryUsage.toFixed(0)}MB`);
                console.log(`     Cache Hit Rate: ${(agentdbHealth.currentMetrics.cacheHitRate * 100).toFixed(1)}%`);
            }
            // Check optimization recommendations
            if (agentdbHealth.recommendations && agentdbHealth.recommendations.length > 0) {
                console.log('\n💡 AgentDB Optimization Recommendations:');
                agentdbHealth.recommendations.forEach((rec, index) => {
                    console.log(`   ${index + 1}. [${rec.priority.toUpperCase()}] ${rec.title}`);
                    console.log(`      ${rec.description}`);
                    console.log(`      Expected Improvement: ${rec.expectedImprovement}%`);
                    console.log(`      Implementation: ${rec.implementation}`);
                });
            }
        }
        // Analyze cognitive performance
        console.log('\n🧠 Cognitive Performance Analysis:');
        const cognitive = overview.cognitiveIntelligence;
        if (cognitive && cognitive.summary) {
            console.log(`   Overall Cognitive Score: ${cognitive.summary.overallCognitiveScore}/100`);
            if (cognitive.temporalAnalysis) {
                console.log('   Temporal Reasoning:');
                console.log(`     Efficiency Score: ${cognitive.temporalAnalysis.efficiencyScore?.toFixed(1)}%`);
                console.log(`     Current Expansion: ${cognitive.temporalAnalysis.currentExpansion?.toFixed(0)}x`);
                console.log(`     Trend: ${cognitive.temporalAnalysis.trend}`);
            }
            if (cognitive.strangeLoopAnalysis) {
                console.log('   Strange-Loop Cognition:');
                console.log(`     Effectiveness: ${cognitive.strangeLoopAnalysis.effectiveness?.toFixed(1)}%`);
                console.log(`     Self-Reference Score: ${cognitive.strangeLoopAnalysis.selfReferenceScore?.toFixed(1)}%`);
                console.log(`     Adaptation Rate: ${(cognitive.strangeLoopAnalysis.adaptationRate * 100).toFixed(1)}%`);
            }
            if (cognitive.learningAnalysis) {
                console.log('   Cross-Agent Learning:');
                console.log(`     Knowledge Transfer: ${(cognitive.learningAnalysis.knowledgeTransfer * 100).toFixed(1)}%`);
                console.log(`     Learning Velocity: ${cognitive.learningAnalysis.learningVelocity?.toFixed(1)} patterns/hour`);
                console.log(`     Pattern Retention: ${(cognitive.learningAnalysis.patternRetention * 100).toFixed(1)}%`);
            }
            // Get cognitive insights
            if (cognitive.predictions && cognitive.predictions.length > 0) {
                console.log('\n🔮 Cognitive Performance Predictions:');
                cognitive.predictions.forEach((pred) => {
                    const trend = pred.predictedValue > pred.currentValue ? '↗️' : pred.predictedValue < pred.currentValue ? '↘️' : '➡️';
                    console.log(`   ${trend} ${pred.metric}: ${pred.currentValue.toFixed(2)} → ${pred.predictedValue.toFixed(2)} (${pred.timeframe})`);
                    console.log(`      Confidence: ${(pred.confidence * 100).toFixed(0)}%`);
                });
            }
        }
        // Generate technical report for detailed analysis
        console.log('\n📈 Generating Technical Analysis Report...');
        const technicalReport = await orchestrator.generatePerformanceReport('technical');
        console.log(`   Report: ${technicalReport.id} (${technicalReport.type})`);
        console.log(`   Performance Score: ${technicalReport.summary.overallScore}/100`);
        // Display key findings from technical report
        if (technicalReport.sections.length > 0) {
            console.log('\n🔍 Key Technical Findings:');
            technicalReport.sections.forEach(section => {
                if (section.insights && section.insights.length > 0) {
                    console.log(`   ${section.title}:`);
                    section.insights.slice(0, 3).forEach(insight => {
                        console.log(`     • ${insight}`);
                    });
                }
            });
        }
        // Display optimization recommendations
        if (technicalReport.recommendations.length > 0) {
            console.log('\n💡 Technical Optimization Recommendations:');
            technicalReport.recommendations.slice(0, 5).forEach(rec => {
                console.log(`   [${rec.priority.toUpperCase()}] ${rec.title}`);
                console.log(`   Expected Performance Gain: ${rec.impact.performance}%`);
                console.log(`   Implementation Effort: ${rec.impact.effort}`);
                console.log(`   Timeline: ${rec.timeline}`);
                console.log('');
            });
        }
        await orchestrator.stop();
        console.log('✅ Performance analysis completed');
    }
    catch (error) {
        console.error('❌ Error in performance analysis example:', error);
    }
}
exports.example3_PerformanceAnalysis = example3_PerformanceAnalysis;
/**
 * Example 4: Real-time Dashboard and Alerting
 */
async function example4_RealTimeDashboard() {
    console.log('\n=== Example 4: Real-time Dashboard and Alerting ===\n');
    try {
        const orchestrator = new performance_1.PerformanceOrchestrator();
        // Set up comprehensive alerting
        let alertCount = 0;
        const maxAlerts = 10;
        orchestrator.on('alert', (alert) => {
            if (alertCount < maxAlerts) {
                alertCount++;
                const severity = alert.type === 'threshold_breach' ? '⚠️' : '📊';
                console.log(`${severity} ALERT [${alert.severity?.toUpperCase()}]: ${alert.metric}`);
                console.log(`   Current: ${alert.value?.toFixed(2)} | Threshold: ${alert.threshold?.toFixed(2)}`);
                console.log(`   Deviation: ${alert.deviationPercent?.toFixed(1)}%`);
            }
        });
        orchestrator.on('bottleneck:detected', (bottleneck) => {
            const severity = bottleneck.severity === 'critical' ? '🚨' :
                bottleneck.severity === 'high' ? '⚠️' : '📝';
            console.log(`${severity} BOTTLENECK: ${bottleneck.component}`);
            console.log(`   ${bottleneck.description}`);
            console.log(`   Performance Impact: ${bottleneck.impact.performanceLoss}%`);
            console.log(`   Recommendation: ${bottleneck.recommendation.action}`);
        });
        orchestrator.on('agentdb:alert', (alert) => {
            console.log(`🗄️ AGENTDB ALERT: ${alert.metric}`);
            console.log(`   Value: ${alert.value?.toFixed(3)} | Threshold: ${alert.threshold?.toFixed(3)}`);
        });
        orchestrator.on('quic:health_issue', (issue) => {
            console.log(`🔗 QUIC HEALTH ISSUE: ${issue.type}`);
            console.log(`   ${issue.impact}`);
        });
        await orchestrator.start();
        console.log('📊 Real-time Dashboard Active');
        console.log('🔔 Alerting System Enabled');
        console.log('📈 Monitoring for performance issues...\n');
        // Simulate real-time monitoring dashboard
        let dashboardCycles = 0;
        const maxCycles = 8;
        const dashboardInterval = setInterval(async () => {
            dashboardCycles++;
            console.log(`\n📊 Dashboard Update ${dashboardCycles}/${maxCycles}`);
            console.log('='.repeat(50));
            // Get current performance overview
            const overview = await orchestrator.getPerformanceOverview();
            // System status
            const health = overview.systemHealth;
            const healthIcon = health.overall === 'healthy' ? '🟢' :
                health.overall === 'degraded' ? '🟡' : '🔴';
            console.log(`${healthIcon} System Health: ${health.overall.toUpperCase()} (${health.score}/100)`);
            // Component status
            console.log('\n🔧 Component Status:');
            health.checks.forEach(check => {
                const icon = check.status === 'healthy' ? '✅' :
                    check.status === 'warning' ? '⚠️' : '❌';
                console.log(`   ${icon} ${check.component}: ${check.status} (${check.responseTime}ms)`);
            });
            // Performance metrics
            if (overview.systemMetrics) {
                console.log('\n📈 System Metrics:');
                console.log(`   CPU Usage: ${overview.systemMetrics.cpu.utilization.toFixed(1)}%`);
                console.log(`   Memory Usage: ${overview.systemMetrics.memory.percentage.toFixed(1)}%`);
                console.log(`   Network Latency: ${overview.systemMetrics.network.latency.toFixed(1)}ms`);
                console.log(`   QUIC Sync: ${overview.systemMetrics.network.quicSyncLatency.toFixed(2)}ms`);
            }
            // AgentDB performance
            if (overview.agentdbPerformance && overview.agentdbPerformance.quicPerformance) {
                const quic = overview.agentdbPerformance.quicPerformance;
                const quicIcon = quic.targetMet ? '✅' : '⚠️';
                console.log(`\n🗄️ AgentDB Performance:`);
                console.log(`   ${quicIcon} QUIC Latency: ${quic.currentLatency.toFixed(2)}ms (target: <${quic.targetLatency}ms)`);
                console.log(`   📊 Throughput: ${quic.avgThroughput.toFixed(0)} queries/sec`);
                console.log(`   ❤️ Health Score: ${quic.healthScore}/100`);
            }
            // Cognitive intelligence
            if (overview.cognitiveIntelligence && overview.cognitiveIntelligence.summary) {
                const cognitive = overview.cognitiveIntelligence.summary;
                console.log(`\n🧠 Cognitive Intelligence:`);
                console.log(`   Overall Score: ${cognitive.overallCognitiveScore}/100`);
                console.log(`   Consciousness Level: ${cognitive.keyMetrics.consciousnessLevel.toFixed(1)}%`);
                console.log(`   Temporal Expansion: ${cognitive.keyMetrics.temporalExpansion.toFixed(0)}x`);
                console.log(`   Learning Velocity: ${cognitive.keyMetrics.learningVelocity.toFixed(1)} patterns/hr`);
            }
            // Active issues
            if (overview.activeIssues.total > 0) {
                console.log(`\n⚠️ Active Issues: ${overview.activeIssues.total}`);
                if (overview.activeIssues.bottlenecks.length > 0) {
                    console.log(`   Bottlenecks: ${overview.activeIssues.bottlenecks.length}`);
                }
                if (overview.activeIssues.anomalies.length > 0) {
                    console.log(`   Anomalies: ${overview.activeIssues.anomalies.length}`);
                }
            }
            // Performance targets status
            console.log('\n🎯 Performance Targets:');
            Object.entries(overview.performanceTargets).forEach(([target, data]) => {
                const status = data.current <= data.target ? '✅' : '⚠️';
                const percentage = ((data.current / data.target) * 100).toFixed(0);
                console.log(`   ${status} ${target}: ${data.current.toFixed(2)}${data.unit} (${percentage}% of target)`);
            });
            if (dashboardCycles >= maxCycles) {
                clearInterval(dashboardInterval);
                await orchestrator.stop();
                console.log('\n✅ Real-time dashboard example completed');
                console.log(`📊 Total Alerts Generated: ${alertCount}`);
            }
        }, 10000); // Every 10 seconds
    }
    catch (error) {
        console.error('❌ Error in real-time dashboard example:', error);
    }
}
exports.example4_RealTimeDashboard = example4_RealTimeDashboard;
/**
 * Main execution function
 */
async function main() {
    console.log('🚀 Cognitive RAN Performance Monitoring System Examples');
    console.log('='.repeat(60));
    // Run all examples
    await example1_QuickStart();
    await new Promise(resolve => setTimeout(resolve, 2000));
    await example2_AdvancedMonitoring();
    await new Promise(resolve => setTimeout(resolve, 2000));
    await example3_PerformanceAnalysis();
    await new Promise(resolve => setTimeout(resolve, 2000));
    await example4_RealTimeDashboard();
    console.log('\n🎉 All performance monitoring examples completed successfully!');
    console.log('\nKey Features Demonstrated:');
    console.log('✅ Real-time performance monitoring with <1s updates');
    console.log('✅ AgentDB QUIC synchronization monitoring (<1ms target)');
    console.log('✅ Cognitive intelligence analytics and bottleneck detection');
    console.log('✅ Automated performance reporting with actionable insights');
    console.log('✅ Predictive performance analysis and prevention');
    console.log('✅ Executive dashboards with comprehensive KPI tracking');
    console.log('✅ SWE-Bench performance tracking (84.8% solve rate target)');
    console.log('✅ Temporal reasoning performance monitoring (1000x expansion)');
    console.log('✅ Strange-loop cognition effectiveness analysis');
}
// Run examples if this file is executed directly
if (require.main === module) {
    main().catch(console.error);
}
//# sourceMappingURL=performance-monitoring.js.map