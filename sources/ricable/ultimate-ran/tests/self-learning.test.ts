
import { describe, test, expect, beforeEach, vi } from 'vitest';
import { SelfLearningAgent, SpatialLearner } from '../src/learning/self-learner.js';

describe('SelfLearningAgent', () => {
    let agent: SelfLearningAgent;

    beforeEach(() => {
        agent = new SelfLearningAgent({
            learningRate: 0.1,
            discountFactor: 0.9,
            epsilon: 0.1
        } as any);
    });

    test('should initialize correctly', () => {
        expect(agent).toBeDefined();
        // Check internal state if possible, or public getters
    });

    test('should process data point via midstream', async () => {
        const spy = vi.spyOn(agent, 'emit');

        // Trigger private processPMData via midstream event
        // Accessing private midstream via any cast
        (agent as any).midstream.emit('pm', {
            cellId: 'CELL_001',
            dataType: 'PM',
            timestamp: Date.now(),
            metrics: {
                pmUlSinrMean: -15, // Low SINR -> anomaly
                pmUlBler: 0.05
            }
        });

        expect(spy).toHaveBeenCalledWith('anomaly', expect.any(Object));
    });

    test('should record episode and calculate reward', () => {
        const episode = {
            id: 'ep-1',
            cellId: 'CELL_001',
            startTime: Date.now(),
            endTime: Date.now() + 1000,
            pmBefore: { pmUlSinrMean: 10, pmCssr: 0.95 },
            pmAfter: { pmUlSinrMean: 15, pmCssr: 0.98 }, // Improved significantly
            cmChange: { electricalTilt: 2 },
            fmAlarms: [],
            // Reward outcome undefined to trigger calculation
        };

        agent.recordEpisode(episode as any);

        // Access private episodes
        const episodes = (agent as any).episodes;
        expect(episodes.length).toBe(1);
        expect(episodes[0].reward).toBeGreaterThan(0);
        expect(episodes[0].outcome).toBe('SUCCESS');
    });
});

describe('SpatialLearner', () => {
    let learner: SpatialLearner;

    beforeEach(() => {
        learner = new SpatialLearner();
    });

    test('should initialize', () => {
        expect(learner).toBeDefined();
    });
});
