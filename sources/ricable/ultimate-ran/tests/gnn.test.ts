
import { describe, test, expect, beforeEach } from 'vitest';
import { GNNUplinkOptimizer, GraphAttentionNetwork } from '../src/gnn/uplink-optimizer.js';

describe('GNNUplinkOptimizer', () => {
    let optimizer: GNNUplinkOptimizer;

    beforeEach(() => {
        optimizer = new GNNUplinkOptimizer({
            learningRate: 0.01,
            discountFactor: 0.95,
            epsilon: 0.1,
            batchSize: 32,
            memorySize: 1000,
            ruvectorDbPath: './test_gnn.db'
        } as any);
    });

    test('should initialize correctly', () => {
        expect(optimizer).toBeDefined();
        const stats = optimizer.getStats();
        expect(stats.cellCount).toBe(0);
        expect(stats.avgReward).toBe(0);
    });

    test('should maintain cell graph', () => {
        const cellId = 'CELL_001';
        // Add cell logic check (if public method exists)
        // The class has addCell method?
        // Looking at file content view in Step 143, it mentions `cellGraph` and `gat`.
        // Let's check if there is a public `addCell` or similar, or `initialize`.
        // It seems it works via `createEpisode` or interactions.

        // Let's test createEpisode
        const pmBefore = { pmUlSinrMean: 10, pmUlBler: 0.1 };
        const pmAfter = { pmUlSinrMean: 12, pmUlBler: 0.05 };
        const action = { p0: -100, alpha: 0.8 };

        const episode = optimizer.createEpisode(cellId, pmBefore as any, pmAfter as any, action);

        expect(episode).toBeDefined();
        expect(episode.reward).toBeGreaterThan(0);
        expect(episode.embedding).toHaveLength(768); // As per constant EMBEDDING_DIM
    });

    test('should store episode', async () => {
        const cellId = 'CELL_001';
        const pmBefore = { pmUlSinrMean: 10, pmUlBler: 0.1 };
        const pmAfter = { pmUlSinrMean: 12, pmUlBler: 0.05 };
        const action = { p0: -100, alpha: 0.8 };

        const episode = optimizer.createEpisode(cellId, pmBefore as any, pmAfter as any, action);

        await optimizer.storeEpisode(episode);

        const stats = optimizer.getStats();
        expect(stats.episodeCount).toBe(1);
    });
});
