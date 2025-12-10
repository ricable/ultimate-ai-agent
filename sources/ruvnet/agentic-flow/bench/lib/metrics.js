/**
 * Metrics Collection and Analysis for ReasoningBank Benchmarks
 */
export class MetricsCollector {
    baselineResults = [];
    reasoningbankResults = [];
    learningCurve = [];
    addResult(result) {
        if (result.agentType === 'baseline') {
            this.baselineResults.push(result);
        }
        else {
            this.reasoningbankResults.push(result);
        }
    }
    addLearningPoint(point) {
        this.learningCurve.push(point);
    }
    calculateAgentMetrics(results) {
        const successful = results.filter(r => r.success);
        const totalTokens = results.reduce((sum, r) => sum + r.tokens, 0);
        const totalLatency = results.reduce((sum, r) => sum + r.latency, 0);
        const errors = results.filter(r => r.error).map(r => r.error);
        const metrics = {
            successRate: results.length > 0 ? successful.length / results.length : 0,
            totalTasks: results.length,
            successfulTasks: successful.length,
            avgTokens: results.length > 0 ? totalTokens / results.length : 0,
            totalTokens,
            avgLatency: results.length > 0 ? totalLatency / results.length : 0,
            totalLatency,
            errors
        };
        // Add ReasoningBank-specific metrics
        const rbResults = results.filter(r => r.memoriesUsed !== undefined);
        if (rbResults.length > 0) {
            const totalMemoriesUsed = rbResults.reduce((sum, r) => sum + (r.memoriesUsed || 0), 0);
            const totalMemoriesCreated = rbResults.reduce((sum, r) => sum + (r.memoriesCreated || 0), 0);
            const totalConfidence = rbResults.reduce((sum, r) => sum + (r.confidence || 0), 0);
            metrics.memoriesUsed = totalMemoriesUsed;
            metrics.memoriesCreated = totalMemoriesCreated;
            metrics.avgConfidence = rbResults.length > 0 ? totalConfidence / rbResults.length : 0;
        }
        return metrics;
    }
    calculateImprovement(baseline, reasoningbank) {
        const successRateDelta = reasoningbank.successRate - baseline.successRate;
        const successRatePercent = baseline.successRate > 0
            ? (successRateDelta / baseline.successRate) * 100
            : (reasoningbank.successRate > 0 ? 100 : 0);
        const tokenDelta = baseline.avgTokens - reasoningbank.avgTokens;
        const tokenSavings = baseline.avgTokens > 0
            ? (tokenDelta / baseline.avgTokens) * 100
            : 0;
        const latencyDelta = reasoningbank.avgLatency - baseline.avgLatency;
        const latencyOverhead = baseline.avgLatency > 0
            ? (latencyDelta / baseline.avgLatency) * 100
            : 0;
        // Calculate learning velocity (iterations to reach 100% success)
        let learningVelocity;
        if (this.learningCurve.length > 0) {
            const firstSuccess = this.learningCurve.findIndex(p => p.reasoningbankSuccess === 1.0);
            const baselineFirstSuccess = this.learningCurve.findIndex(p => p.baselineSuccess === 1.0);
            if (firstSuccess !== -1 && baselineFirstSuccess !== -1) {
                learningVelocity = baselineFirstSuccess / firstSuccess;
            }
            else if (firstSuccess !== -1) {
                learningVelocity = this.learningCurve.length / firstSuccess;
            }
        }
        return {
            successRateDelta: this.formatPercent(successRateDelta),
            successRatePercent,
            tokenEfficiency: this.formatPercent(tokenSavings / 100),
            tokenSavings,
            latencyOverhead: this.formatPercent(latencyOverhead / 100),
            latencyDelta,
            learningVelocity
        };
    }
    formatPercent(value) {
        const sign = value >= 0 ? '+' : '';
        return `${sign}${(value * 100).toFixed(1)}%`;
    }
    generateScenarioResults(scenarioName) {
        const baseline = this.calculateAgentMetrics(this.baselineResults);
        const reasoningbank = this.calculateAgentMetrics(this.reasoningbankResults);
        const improvement = this.calculateImprovement(baseline, reasoningbank);
        return {
            scenarioName,
            baseline,
            reasoningbank,
            improvement,
            learningCurve: [...this.learningCurve],
            timestamp: new Date().toISOString()
        };
    }
    reset() {
        this.baselineResults = [];
        this.reasoningbankResults = [];
        this.learningCurve = [];
    }
    // Statistical analysis methods
    calculateStandardDeviation(values) {
        if (values.length === 0)
            return 0;
        const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
        const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
        return Math.sqrt(variance);
    }
    calculateConfidenceInterval(values, confidenceLevel = 0.95) {
        if (values.length === 0) {
            return { lower: 0, upper: 0, mean: 0 };
        }
        const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
        const std = this.calculateStandardDeviation(values);
        const z = confidenceLevel === 0.95 ? 1.96 : 2.576; // 95% or 99%
        const margin = z * (std / Math.sqrt(values.length));
        return {
            lower: mean - margin,
            upper: mean + margin,
            mean
        };
    }
    // Performance insights
    generateInsights(results) {
        const insights = [];
        const { baseline, reasoningbank, improvement } = results;
        // Success rate insights
        if (improvement.successRatePercent > 50) {
            insights.push(`🎯 Excellent improvement: ${improvement.successRateDelta} success rate increase`);
        }
        else if (improvement.successRatePercent < 0) {
            insights.push(`⚠️  Warning: Baseline outperformed ReasoningBank (${improvement.successRateDelta})`);
        }
        // Token efficiency insights
        if (improvement.tokenSavings > 30) {
            insights.push(`💰 Significant token savings: ${improvement.tokenEfficiency} reduction`);
        }
        else if (improvement.tokenSavings < 0) {
            insights.push(`⚠️  Token overhead: ${Math.abs(improvement.tokenSavings).toFixed(1)}% increase`);
        }
        // Latency insights
        if (Math.abs(improvement.latencyDelta) < 500) {
            insights.push(`⚡ Minimal latency overhead: ${Math.abs(improvement.latencyDelta).toFixed(0)}ms`);
        }
        else if (improvement.latencyDelta > 1000) {
            insights.push(`🐌 High latency overhead: ${improvement.latencyDelta.toFixed(0)}ms - consider optimization`);
        }
        // Memory insights
        if (reasoningbank.memoriesUsed && reasoningbank.memoriesUsed > 0) {
            const avgMemoriesPerTask = reasoningbank.memoriesUsed / reasoningbank.totalTasks;
            insights.push(`🧠 Memory utilization: ${avgMemoriesPerTask.toFixed(1)} memories per task`);
        }
        // Learning velocity insights
        if (improvement.learningVelocity && improvement.learningVelocity > 2) {
            insights.push(`🚀 Fast learner: ${improvement.learningVelocity.toFixed(1)}x faster than baseline`);
        }
        return insights;
    }
}
// Export singleton instance
export const metrics = new MetricsCollector();
