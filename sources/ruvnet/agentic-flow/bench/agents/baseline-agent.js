/**
 * Baseline Agent (No Memory)
 *
 * Executes tasks without any learning or memory capabilities.
 * Used as the control group in benchmarks.
 */
import Anthropic from '@anthropic-ai/sdk';
export class BaselineAgent {
    type = 'baseline';
    client;
    model;
    constructor(apiKey, model = 'claude-sonnet-4-5-20250929') {
        this.client = new Anthropic({
            apiKey: apiKey || process.env.ANTHROPIC_API_KEY
        });
        this.model = model;
    }
    async executeTask(task) {
        const startTime = Date.now();
        try {
            // Execute task with Claude (no memory injection)
            const response = await this.client.messages.create({
                model: this.model,
                max_tokens: 4096,
                messages: [
                    {
                        role: 'user',
                        content: `${task.description}\n\nInput:\n${task.input}`
                    }
                ]
            });
            const latency = Date.now() - startTime;
            const output = response.content[0].type === 'text' ? response.content[0].text : '';
            const tokens = response.usage.input_tokens + response.usage.output_tokens;
            // Evaluate success
            const success = task.successCriteria({
                taskId: task.id,
                agentType: 'baseline',
                success: false, // Will be updated
                output,
                tokens,
                latency
            });
            return {
                taskId: task.id,
                agentType: 'baseline',
                success,
                output,
                tokens,
                latency
            };
        }
        catch (error) {
            const latency = Date.now() - startTime;
            return {
                taskId: task.id,
                agentType: 'baseline',
                success: false,
                output: '',
                tokens: 0,
                latency,
                error: error instanceof Error ? error.message : String(error)
            };
        }
    }
    async reset() {
        // Baseline has no state to reset
    }
}
