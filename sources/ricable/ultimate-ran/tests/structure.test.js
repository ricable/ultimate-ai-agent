/**
 * TITAN Platform Structure Tests
 * Checks for existence of files and key exports/methods.
 */

import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { readFileSync, existsSync } from 'fs';
import { describe, it, expect } from 'vitest';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const ROOT = join(__dirname, '..');

describe('Track A: Council Engine (Backend)', () => {
    it('Council Orchestrator file exists', () => {
        const path = join(ROOT, 'src/council/orchestrator.ts');
        expect(existsSync(path)).toBe(true);
    });

    it('Council Orchestrator exports councilDefinitions', () => {
        // Note: Assuming file exists from previous test
        const content = readFileSync(join(ROOT, 'src/council/orchestrator.ts'), 'utf-8');
        expect(content).toContain('councilDefinitions');
        expect(content).toContain('analyst-deepseek');
        expect(content).toContain('historian-gemini');
        expect(content).toContain('strategist-claude');
    });

    it('Council Orchestrator has fan_out_to_council method', () => {
        const content = readFileSync(join(ROOT, 'src/council/orchestrator.ts'), 'utf-8');
        const hasMethod = content.includes('fan_out_to_council') || content.includes('fanOutToCouncil');
        expect(hasMethod).toBe(true);
    });

    it('Debate Protocol file exists', () => {
        const path = join(ROOT, 'src/council/debate-protocol.ts');
        expect(existsSync(path)).toBe(true);
    });

    it('Multi-Model Router file exists', () => {
        const path = join(ROOT, 'src/council/router.ts');
        expect(existsSync(path)).toBe(true);
    });

    it('Router has capability-based routing', () => {
        const content = readFileSync(join(ROOT, 'src/council/router.ts'), 'utf-8');
        const hasMethod = content.includes('route_by_capability') || content.includes('routeByCapability');
        expect(hasMethod).toBe(true);
    });

    it('Chairman Agent file exists', () => {
        const path = join(ROOT, 'src/council/chairman.ts');
        expect(existsSync(path)).toBe(true);
    });
});

describe('Track B: AG-UI Dojo (Frontend)', () => {
    it('Dojo agent directory exists', () => {
        const path = join(ROOT, 'apps/dojo/src/agents/titan');
        expect(existsSync(path)).toBe(true);
    });

    it('TitanCouncilAgent implemented', () => {
        const path = join(ROOT, 'apps/dojo/src/agents/titan/index.ts');
        expect(existsSync(path)).toBe(true);
        const content = readFileSync(path, 'utf-8');
        expect(content).toContain('TitanCouncilAgent');
    });

    it('UI Components directory exists', () => {
        const path = join(ROOT, 'apps/dojo/src/components/titan');
        expect(existsSync(path)).toBe(true);
    });
});

describe('Track C: Consensus Memory (Data)', () => {
    it('Memory schema file exists', () => {
        const path = join(ROOT, 'src/memory/schema.ts');
        expect(existsSync(path)).toBe(true);
    });

    it('Schema has debate_episodes', () => {
        const content = readFileSync(join(ROOT, 'src/memory/schema.ts'), 'utf-8');
        const hasDebate = content.includes('debate') || content.includes('Debate');
        expect(hasDebate).toBe(true);
    });

    it('Vector indexing file exists', () => {
        const path = join(ROOT, 'src/memory/vector-index.ts');
        expect(existsSync(path)).toBe(true);
    });
});

describe('Track D: SPARC Governance', () => {
    it('Safety hooks file exists', () => {
        const path = join(ROOT, 'src/hooks/safety.ts');
        expect(existsSync(path)).toBe(true);
    });

    it('SPARC Enforcer file exists', () => {
        const path = join(ROOT, 'src/governance/sparc-enforcer.ts');
        expect(existsSync(path)).toBe(true);
    });

    it('5-gate validation implemented', () => {
        const content = readFileSync(join(ROOT, 'src/governance/sparc-enforcer.ts'), 'utf-8');
        expect(content.toLowerCase()).toContain('specification');
        expect(content.toLowerCase()).toContain('pseudocode');
        expect(content.toLowerCase()).toContain('architecture');
        expect(content.toLowerCase()).toContain('refinement');
        expect(content.toLowerCase()).toContain('completion');
    });
});

describe('Configuration Files', () => {
    it('Package.json exists', () => {
        const path = join(ROOT, 'package.json');
        expect(existsSync(path)).toBe(true);
    });

    it('Package.json has required scripts', () => {
        const pkg = JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf-8'));
        expect(pkg.scripts.start).toBeDefined();
        expect(pkg.scripts.build).toBeDefined();
    });
});

describe('Core Infrastructure', () => {
    it('Main index.js exists', () => {
        const path = join(ROOT, 'src/index.js');
        expect(existsSync(path)).toBe(true);
    });

    it('RACS orchestrator exists', () => {
        const path = join(ROOT, 'src/racs/orchestrator.js');
        expect(existsSync(path)).toBe(true);
    });

    it('AgentDB client exists', () => {
        const path = join(ROOT, 'src/cognitive/agentdb-client.js');
        expect(existsSync(path)).toBe(true);
    });
});
