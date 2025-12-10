/**
 * TITAN RAN Automation - Integration Tests
 * End-to-end testing for cognitive mesh components
 */

import { TitanOrchestrator } from '../src/racs/orchestrator.js';
import { AgentDBClient } from '../src/cognitive/agentdb-client.js';
import { RuvectorEngine } from '../src/cognitive/ruvector-engine.js';
import { SPARCValidator } from '../src/sparc/validator.js';
import { AGUIServer } from '../src/agui/server.js';
import { describe, it, expect, beforeEach } from 'vitest';

describe('Integration Tests', () => {

  describe('Suite 1: AgentDB Client', () => {
    it('AgentDB initialization', async () => {
      const agentDB = new AgentDBClient({
        path: ':memory:',
        backend: 'ruvector',
        dimension: 768
      });

      await agentDB.initialize();
      expect(agentDB.initialized).toBe(true);
    });

    it('Store optimization episode', async () => {
      const agentDB = new AgentDBClient({ path: ':memory:', dimension: 768 });

      const episode = {
        symptom: 'Low SINR in sector A',
        context: { cluster: 'downtown', cells: ['cell-1', 'cell-2'] },
        actionSequence: ['analyze', 'optimize', 'deploy'],
        outcome: 'SINR improved by 2.1dB',
        critique: 'Success - pattern applicable to similar scenarios'
      };

      const result = await agentDB.storeEpisode(episode);
      expect(result.stored).toBe(true);
      expect(result.id).toBeDefined();
    });

    it('Vector embedding', async () => {
      const agentDB = new AgentDBClient({ path: ':memory:', dimension: 768 });

      const text = 'Optimize parameters for maximum throughput';
      const vector = await agentDB.embed(text);

      expect(Array.isArray(vector)).toBe(true);
      expect(vector.length).toBe(768);
    });

    it('Store reflexion', async () => {
      const agentDB = new AgentDBClient({ path: ':memory:', dimension: 768 });

      const reflexion = {
        action: 'Increased alpha to 1.0',
        result: 'Cell became unstable',
        critique: 'Alpha should not exceed 0.9 for dense urban',
        doNotRepeat: true
      };

      const result = await agentDB.storeReflexion(reflexion);
      expect(result.stored).toBe(true);
    });
  });

  describe('Suite 2: Ruvector Engine', () => {
    it('Ruvector initialization', async () => {
      const ruvector = new RuvectorEngine({
        path: ':memory:',
        dimension: 768,
        metric: 'cosine'
      });

      await ruvector.initialize();
      expect(ruvector.initialized).toBe(true);
    });

    it('Create hypergraph', async () => {
      const ruvector = new RuvectorEngine({ path: ':memory:', dimension: 768 });
      await ruvector.initialize();

      const cells = [
        { id: 'cell-1', position: { lat: 40.7, lon: -74.0 }, features: [0.1, 0.2, 0.3] },
        { id: 'cell-2', position: { lat: 40.8, lon: -74.1 }, features: [0.2, 0.3, 0.4] },
        { id: 'cell-3', position: { lat: 40.9, lon: -74.2 }, features: [0.3, 0.4, 0.5] }
      ];

      const hypergraph = ruvector.createHypergraph(cells);

      expect(hypergraph.nodes.length).toBe(3);
      expect(Array.isArray(hypergraph.hyperedges)).toBe(true);
    });

    it('Calculate attention weights', async () => {
      const ruvector = new RuvectorEngine({ path: ':memory:', dimension: 768 });

      const targetCell = { id: 'cell-1', position: { lat: 40.7, lon: -74.0 } };
      const neighbors = [
        { id: 'cell-2', position: { lat: 40.8, lon: -74.1 } },
        { id: 'cell-3', position: { lat: 40.9, lon: -74.2 } }
      ];

      const weights = ruvector.calculateAttentionWeights(targetCell, neighbors);

      expect(weights.length).toBe(2);
      expect(weights[0].weight).toBeDefined();
      expect(weights[0].hasLOS).toBeDefined();
    });
  });

  describe('Suite 3: SPARC Validator', () => {
    it('SPARC validator initialization', async () => {
      const validator = new SPARCValidator({});
      expect(validator.config).toBeDefined();
    });

    it('Valid artifact passes all gates', async () => {
      const validator = new SPARCValidator({});

      const artifact = {
        id: 'test-artifact-valid',
        specification: {
          objective_function: 'maximize(SINR)',
          safety_constraints: ['CSSR >= 0.995']
        },
        pseudocode: 'for each cell -> optimize parameters',
        architecture: {
          stack: 'ruvnet'
        },
        refinement: {
          tests: ['test1', 'test2'],
          memoryUsage: 50
        },
        completion: {
          compliant: true
        },
        lyapunovExponent: -0.05
      };

      const result = await validator.validateArtifact(artifact);
      expect(result.passed).toBe(true);
    });

    it('Invalid specification fails', async () => {
      const validator = new SPARCValidator({});

      const artifact = {
        id: 'test-artifact-invalid',
        specification: {
          // Missing objective_function
          safety_constraints: ['CSSR >= 0.995']
        },
        pseudocode: 'test',
        architecture: {},
        refinement: { tests: [] },
        completion: {}
      };

      const result = await validator.validateArtifact(artifact);
      expect(result.passed).toBe(false);
    });
  });

  describe('Suite 4: AG-UI Server', () => {
    it('AG-UI server initialization', async () => {
      const server = new AGUIServer({ port: 4000 });
      expect(server.port).toBe(4000);
    });

    it('AG-UI server start', async () => {
      const server = new AGUIServer({ port: 4001 });
      await server.start();
      expect(server.running).toBe(true);
    });

    it('Render interference heatmap', async () => {
      const server = new AGUIServer({ port: 4003 });

      const cells = ['cell-1', 'cell-2'];
      const interferenceMatrix = [[0, 5], [5, 0]];

      server.renderInterferenceHeatmap(cells, interferenceMatrix, 10);
      expect(true).toBe(true); // Just ensuring no throw
    });
  });

  describe('Suite 5: Titan Orchestrator', () => {
    it('Orchestrator initialization', async () => {
      const agentDB = new AgentDBClient({ path: ':memory:', dimension: 768 });
      const ruvector = new RuvectorEngine({ path: ':memory:', dimension: 768 });
      const sparcValidator = new SPARCValidator({});
      const aguiServer = new AGUIServer({ port: 4004 });

      const orchestrator = new TitanOrchestrator({
        config: { swarm: { agents: [] } },
        agentDB,
        ruvector,
        sparcValidator,
        aguiServer
      });

      expect(orchestrator).toBeDefined();
    });

    it('Spawn single agent', async () => {
      const agentDB = new AgentDBClient({ path: ':memory:', dimension: 768 });
      const ruvector = new RuvectorEngine({ path: ':memory:', dimension: 768 });
      const sparcValidator = new SPARCValidator({});
      const aguiServer = new AGUIServer({ port: 4005 });

      const orchestrator = new TitanOrchestrator({
        config: { swarm: { agents: [] } },
        agentDB,
        ruvector,
        sparcValidator,
        aguiServer
      });

      const agent = await orchestrator.spawnAgent('architect', 'test context');

      expect(agent.id).toBeDefined();
      expect(agent.type).toBe('architect');
      expect(agent.status).toBe('initialized');
    });

    it('Determine squad for optimization intent', async () => {
      const agentDB = new AgentDBClient({ path: ':memory:', dimension: 768 });
      const ruvector = new RuvectorEngine({ path: ':memory:', dimension: 768 });
      const sparcValidator = new SPARCValidator({});
      const aguiServer = new AGUIServer({ port: 4006 });

      const orchestrator = new TitanOrchestrator({
        config: { swarm: { agents: [] } },
        agentDB,
        ruvector,
        sparcValidator,
        aguiServer
      });

      const squad = orchestrator.determineSquad('Optimize P0/alpha for downtown cluster', []);

      expect(squad).toContain('architect');
      expect(squad).toContain('artisan');
      expect(squad).toContain('guardian');
    });

    it('Execute RIV pattern', async () => {
      const agentDB = new AgentDBClient({ path: ':memory:', dimension: 768 });
      const ruvector = new RuvectorEngine({ path: ':memory:', dimension: 768 });
      const sparcValidator = new SPARCValidator({});
      const aguiServer = new AGUIServer({ port: 4007 });

      const orchestrator = new TitanOrchestrator({
        config: { swarm: { agents: [] } },
        agentDB,
        ruvector,
        sparcValidator,
        aguiServer
      });

      const result = await orchestrator.executeRIV('Deploy configuration to cluster');

      expect(result.missionPlan).toBeDefined();
      expect(result.workers).toBeDefined();
      expect(result.sentinel).toBeDefined();
    });
  });

  describe('Suite 6: End-to-End Integration', () => {
    it('Full cognitive mesh initialization', async () => {
      const agentDB = new AgentDBClient({ path: ':memory:', dimension: 768 });
      const ruvector = new RuvectorEngine({ path: ':memory:', dimension: 768 });
      const sparcValidator = new SPARCValidator({});
      const aguiServer = new AGUIServer({ port: 4008 });

      await agentDB.initialize();
      await ruvector.initialize();
      await aguiServer.start();

      const orchestrator = new TitanOrchestrator({
        config: {
          codename: 'Neuro-Symbolic Titan',
          swarm: { agents: ['architect', 'artisan', 'guardian'] }
        },
        agentDB,
        ruvector,
        sparcValidator,
        aguiServer
      });

      expect(agentDB.initialized).toBe(true);
      expect(ruvector.initialized).toBe(true);
      expect(aguiServer.running).toBe(true);
      expect(orchestrator).toBeDefined();
    });

    it('Intent routing and squad spawning', async () => {
      const agentDB = new AgentDBClient({ path: ':memory:', dimension: 768 });
      const ruvector = new RuvectorEngine({ path: ':memory:', dimension: 768 });
      const sparcValidator = new SPARCValidator({});
      const aguiServer = new AGUIServer({ port: 4009 });

      const orchestrator = new TitanOrchestrator({
        config: { swarm: { agents: [] } },
        agentDB,
        ruvector,
        sparcValidator,
        aguiServer
      });

      const squad = await orchestrator.routeIntent('Optimize SINR for urban cluster');

      expect(Array.isArray(squad)).toBe(true);
      expect(squad.length).toBeGreaterThan(0);
    });
  });

});
