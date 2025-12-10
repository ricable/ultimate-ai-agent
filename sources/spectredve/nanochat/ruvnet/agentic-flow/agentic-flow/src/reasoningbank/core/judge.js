/**
 * LLM-as-Judge for trajectory evaluation
 * Algorithm 2 from ReasoningBank paper
 */
import { readFileSync } from 'fs';
import { join } from 'path';
import { loadConfig } from '../utils/config.js';
/**
 * Judge a task trajectory using LLM evaluation
 */
export async function judgeTrajectory(trajectory, query, options = {}) {
    const config = loadConfig();
    const startTime = Date.now();
    console.log(`[INFO] Judging trajectory for query: ${query.substring(0, 100)}...`);
    // Load judge prompt template
    const promptPath = join(process.cwd(), 'src', 'reasoningbank', 'prompts', 'judge.json');
    const promptTemplate = JSON.parse(readFileSync(promptPath, 'utf-8'));
    // Format trajectory for judgment
    const trajectoryText = formatTrajectory(trajectory);
    // Check if we have Anthropic API key
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
        console.warn('[WARN] ANTHROPIC_API_KEY not set, using heuristic judgment');
        return heuristicJudge(trajectory, query);
    }
    try {
        // Call Anthropic API with judge prompt
        const prompt = promptTemplate.template
            .replace('{{task_query}}', query)
            .replace('{{trajectory}}', trajectoryText);
        const response = await fetch('https://api.anthropic.com/v1/messages', {
            method: 'POST',
            headers: {
                'x-api-key': apiKey,
                'anthropic-version': '2023-06-01',
                'content-type': 'application/json'
            },
            body: JSON.stringify({
                model: config.judge.model,
                max_tokens: config.judge.max_tokens,
                temperature: config.judge.temperature,
                system: promptTemplate.system,
                messages: [{ role: 'user', content: prompt }]
            })
        });
        if (!response.ok) {
            throw new Error(`Anthropic API error: ${response.status}`);
        }
        const result = await response.json();
        const content = result.content[0].text;
        // Parse JSON response
        const verdict = parseVerdict(content);
        const duration = Date.now() - startTime;
        console.log(`[INFO] Judgment complete: ${verdict.label} (${verdict.confidence}) in ${duration}ms`);
        db.logMetric('rb.judge.latency_ms', duration);
        db.logMetric('rb.judge.success_rate', verdict.label === 'Success' ? 1 : 0);
        return verdict;
    }
    catch (error) {
        console.error('[ERROR] Judge failed:', error);
        console.warn('[WARN] Falling back to heuristic judgment');
        return heuristicJudge(trajectory, query);
    }
}
/**
 * Format trajectory for LLM consumption
 */
function formatTrajectory(trajectory) {
    const steps = trajectory.steps || [];
    let formatted = '';
    for (let i = 0; i < steps.length; i++) {
        formatted += `Step ${i + 1}: ${JSON.stringify(steps[i], null, 2)}\n\n`;
    }
    return formatted || 'No steps recorded';
}
/**
 * Parse verdict from LLM response
 */
function parseVerdict(content) {
    try {
        // Try to extract JSON from response
        const jsonMatch = content.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
            const parsed = JSON.parse(jsonMatch[0]);
            return {
                label: parsed.verdict?.label || parsed.label || 'Failure',
                confidence: parsed.verdict?.confidence || parsed.confidence || 0.5,
                reasons: parsed.verdict?.reasons || parsed.reasons || []
            };
        }
    }
    catch (error) {
        console.warn('[WARN] Failed to parse verdict JSON, using text analysis');
    }
    // Fallback: text-based detection
    const lower = content.toLowerCase();
    const isSuccess = lower.includes('success') && !lower.includes('failure');
    return {
        label: isSuccess ? 'Success' : 'Failure',
        confidence: 0.6,
        reasons: ['Parsed from text (JSON parse failed)']
    };
}
/**
 * Heuristic judgment when LLM is unavailable
 * Simple rule-based classification
 */
function heuristicJudge(trajectory, query) {
    const steps = trajectory.steps || [];
    // Check for error indicators
    const hasErrors = steps.some(step => JSON.stringify(step).toLowerCase().includes('error'));
    // Check for completion indicators
    const hasCompletion = steps.some(step => JSON.stringify(step).toLowerCase().includes('complete'));
    // Simple heuristic: success if no errors and has completion markers
    const isSuccess = !hasErrors && hasCompletion && steps.length > 0;
    return {
        label: isSuccess ? 'Success' : 'Failure',
        confidence: 0.5, // Low confidence for heuristic
        reasons: [
            `Heuristic judgment (no API key)`,
            `Steps: ${steps.length}`,
            `Errors: ${hasErrors}`,
            `Completion markers: ${hasCompletion}`
        ]
    };
}
// Import db late to avoid circular dependency
import * as db from '../db/queries.js';
