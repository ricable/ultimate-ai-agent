/**
 * Swarm Intelligence Example
 * Demonstrates multi-agent collaboration using ruv-swarm
 */

import 'dotenv/config';
import { SwarmOrchestrator } from 'ruv-swarm';
import { AgenticFlow } from 'agentic-flow';

/**
 * Example: Software Development Swarm
 * Multiple specialized agents collaborate on a project
 */
async function softwareDevelopmentSwarm() {
  console.log('\n' + '='.repeat(60));
  console.log('Software Development Swarm Example');
  console.log('='.repeat(60));

  // Define specialized agents
  const swarm = new SwarmOrchestrator({
    topology: 'hierarchical', // Queen-drone pattern

    // Queen agent: Project manager
    queen: new AgenticFlow({
      name: 'ProjectManager',
      role: 'Break down tasks and coordinate team',
      provider: 'local',
      baseURL: process.env.GAIANET_ENDPOINT,
      model: process.env.GAIANET_MODEL
    }),

    // Drone agents: Specialized workers
    drones: [
      new AgenticFlow({
        name: 'BackendDeveloper',
        role: 'API and database development',
        specialization: 'Node.js, Express, PostgreSQL',
        provider: 'local',
        baseURL: process.env.GAIANET_ENDPOINT,
        model: process.env.GAIANET_MODEL
      }),

      new AgenticFlow({
        name: 'FrontendDeveloper',
        role: 'UI/UX implementation',
        specialization: 'React, TypeScript, CSS',
        provider: 'local',
        baseURL: process.env.GAIANET_ENDPOINT,
        model: process.env.GAIANET_MODEL
      }),

      new AgenticFlow({
        name: 'QAEngineer',
        role: 'Testing and quality assurance',
        specialization: 'Jest, Cypress, test automation',
        provider: 'local',
        baseURL: process.env.GAIANET_ENDPOINT,
        model: process.env.GAIANET_MODEL
      }),

      new AgenticFlow({
        name: 'DevOpsEngineer',
        role: 'Deployment and infrastructure',
        specialization: 'Docker, CI/CD, monitoring',
        provider: 'local',
        baseURL: process.env.GAIANET_ENDPOINT,
        model: process.env.GAIANET_MODEL
      })
    ]
  });

  // Project specification
  const project = {
    name: 'Task Management API',
    requirements: [
      'RESTful API for task CRUD operations',
      'User authentication with JWT',
      'SQLite database',
      'Input validation',
      'Unit and integration tests',
      'Docker containerization'
    ]
  };

  console.log('\n📋 Project:', project.name);
  console.log('Requirements:', project.requirements.join(', '));

  // Execute swarm
  const result = await swarm.execute({
    task: `Build a ${project.name} with the following requirements: ${project.requirements.join(', ')}`,

    // Swarm behavior configuration
    collaboration: {
      enabled: true,
      mode: 'sequential', // Agents work in sequence (can be 'parallel' or 'hybrid')
      communication: 'shared-context' // Agents share context via AgentDB
    },

    // Quality gates
    validation: {
      codeReview: true,
      testing: true,
      deployment: true
    }
  });

  console.log('\n✅ Swarm Execution Complete!');
  console.log('\n📦 Deliverables:');
  console.log('- Backend API:', result.artifacts.backend?.files || 'N/A');
  console.log('- Frontend:', result.artifacts.frontend?.files || 'N/A');
  console.log('- Tests:', result.artifacts.tests?.coverage || 'N/A');
  console.log('- Deployment:', result.artifacts.deployment?.dockerfile ? 'Ready' : 'Pending');

  return result;
}

/**
 * Example: Mesh Network for Parallel Processing
 * All agents communicate peer-to-peer
 */
async function meshNetworkExample() {
  console.log('\n' + '='.repeat(60));
  console.log('Mesh Network Example: Parallel Data Processing');
  console.log('='.repeat(60));

  const swarm = new SwarmOrchestrator({
    topology: 'mesh', // All agents can communicate with each other

    agents: Array.from({ length: 4 }, (_, i) =>
      new AgenticFlow({
        name: `DataProcessor${i + 1}`,
        role: 'Process data partition',
        provider: 'local',
        baseURL: process.env.GAIANET_ENDPOINT,
        model: process.env.GAIANET_MODEL
      })
    )
  });

  // Simulate large dataset
  const dataset = Array.from({ length: 10000 }, (_, i) => ({
    id: i,
    value: Math.random() * 1000,
    category: ['A', 'B', 'C', 'D'][Math.floor(Math.random() * 4)]
  }));

  const result = await swarm.execute({
    task: 'Process dataset: calculate statistics per category',
    data: dataset,

    // Parallel execution
    execution: {
      mode: 'parallel',
      partitioning: 'auto', // Auto-partition data across agents
      aggregation: 'reduce' // Combine results via reduce operation
    }
  });

  console.log('\n📊 Processing Results:');
  console.log('Records processed:', result.recordsProcessed);
  console.log('Execution time:', result.executionTime, 'ms');
  console.log('Speedup:', result.speedup, 'x');
  console.log('Statistics:', result.statistics);

  return result;
}

/**
 * Example: Star Network with Central Coordinator
 * One central agent coordinates multiple workers
 */
async function starNetworkExample() {
  console.log('\n' + '='.repeat(60));
  console.log('Star Network Example: Code Review');
  console.log('='.repeat(60));

  const swarm = new SwarmOrchestrator({
    topology: 'star',

    // Central coordinator
    hub: new AgenticFlow({
      name: 'LeadReviewer',
      role: 'Coordinate code review and synthesize feedback',
      provider: 'local',
      baseURL: process.env.GAIANET_ENDPOINT,
      model: process.env.GAIANET_MODEL
    }),

    // Specialist reviewers
    spokes: [
      new AgenticFlow({
        name: 'SecurityReviewer',
        role: 'Security and vulnerability analysis',
        specialization: 'OWASP, security best practices',
        provider: 'local',
        baseURL: process.env.GAIANET_ENDPOINT,
        model: process.env.GAIANET_MODEL
      }),

      new AgenticFlow({
        name: 'PerformanceReviewer',
        role: 'Performance optimization',
        specialization: 'Algorithmic complexity, profiling',
        provider: 'local',
        baseURL: process.env.GAIANET_ENDPOINT,
        model: process.env.GAIANET_MODEL
      }),

      new AgenticFlow({
        name: 'StyleReviewer',
        role: 'Code style and best practices',
        specialization: 'Clean code, design patterns',
        provider: 'local',
        baseURL: process.env.GAIANET_ENDPOINT,
        model: process.env.GAIANET_MODEL
      })
    ]
  });

  const codeToReview = `
    function getUserData(userId) {
      const data = db.query('SELECT * FROM users WHERE id = ' + userId);
      return data[0];
    }
  `;

  const result = await swarm.execute({
    task: 'Review this code for issues',
    code: codeToReview,

    // Review configuration
    review: {
      depth: 'comprehensive',
      autoFix: true, // Agents propose fixes
      consensus: true // Require agreement from all reviewers
    }
  });

  console.log('\n🔍 Review Results:');
  console.log('\nSecurity Issues:', result.issues.security?.length || 0);
  console.log('Performance Issues:', result.issues.performance?.length || 0);
  console.log('Style Issues:', result.issues.style?.length || 0);

  console.log('\n💡 Recommended Fixes:');
  console.log(result.fixes);

  return result;
}

/**
 * Main execution
 */
async function main() {
  console.log('🐝 Sovereign Agentic Stack - Swarm Intelligence Examples');
  console.log('Using local GaiaNet node:', process.env.GAIANET_ENDPOINT);

  try {
    await softwareDevelopmentSwarm();
    await meshNetworkExample();
    await starNetworkExample();

    console.log('\n' + '='.repeat(60));
    console.log('✅ All swarm examples completed!');
    console.log('='.repeat(60));

    console.log('\n📈 Benefits of Swarm Architecture:');
    console.log('  • Parallel processing (faster execution)');
    console.log('  • Specialized expertise (better quality)');
    console.log('  • Fault tolerance (redundancy)');
    console.log('  • Scalability (add more agents as needed)');
    console.log('  • ALL running on your local GaiaNet node (zero API costs!)');

  } catch (error) {
    console.error('\n❌ Error:', error.message);
    process.exit(1);
  }
}

// Run if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}

export {
  softwareDevelopmentSwarm,
  meshNetworkExample,
  starNetworkExample
};
