/**
 * QDAG Orchestrator - Quantum Directed Acyclic Graph
 * DAG-based workflow orchestration for complex agent pipelines
 */

import 'dotenv/config';
import { QDAGOrchestrator } from '@ruv/qdag';
import { AgentDB } from 'agentdb';
import { RuvLLM } from 'ruvllm';

/**
 * Initialize QDAG with AgentDB and RuvLLM
 */
const agentDB = new AgentDB({
  adapter: 'sqlite',
  database: process.env.AGENT_DB_PATH || './agent-db/qdag.db',
  distributed: {
    enabled: true,
    redis: {
      host: process.env.REDIS_HOST || 'localhost',
      port: parseInt(process.env.REDIS_PORT) || 6379
    }
  }
});

const ruvLLM = new RuvLLM({
  providers: [
    {
      name: 'local-gaianet',
      type: 'openai-compatible',
      baseURL: process.env.GAIANET_ENDPOINT || 'http://localhost:8080/v1',
      model: process.env.GAIANET_MODEL || 'Qwen2.5-Coder-32B-Instruct',
      priority: 1
    }
  ]
});

const qdag = new QDAGOrchestrator({
  database: agentDB,
  llm: ruvLLM,

  // DAG execution configuration
  execution: {
    mode: 'distributed', // Execute across cluster
    parallelism: 'max', // Max parallel execution of independent nodes

    // Checkpointing for long-running workflows
    checkpointing: {
      enabled: true,
      interval: 60000, // Every minute
      storage: './checkpoints'
    }
  },

  // Workflow versioning
  versioning: {
    enabled: true,
    storage: agentDB
  }
});

/**
 * Example 1: Software Development Pipeline
 */
async function softwareDevelopmentPipeline() {
  console.log('\n' + '='.repeat(60));
  console.log('QDAG Example 1: Software Development Pipeline');
  console.log('='.repeat(60));

  // Define DAG workflow
  const workflow = await qdag.createWorkflow({
    name: 'fullstack-app-development',
    description: 'End-to-end application development pipeline',

    // Define nodes (tasks)
    nodes: [
      {
        id: 'requirements',
        type: 'agent',
        agent: 'business-analyst',
        task: 'Analyze requirements and create specification',
        inputs: {
          userStory: 'Build a task management application'
        },
        outputs: ['specification']
      },

      {
        id: 'architecture',
        type: 'agent',
        agent: 'architect',
        task: 'Design system architecture',
        inputs: {
          specification: 'requirements.specification'
        },
        outputs: ['architecture-diagram', 'tech-stack']
      },

      {
        id: 'database-design',
        type: 'agent',
        agent: 'database-engineer',
        task: 'Design database schema',
        inputs: {
          specification: 'requirements.specification',
          architecture: 'architecture.architecture-diagram'
        },
        outputs: ['schema', 'migrations']
      },

      {
        id: 'backend-api',
        type: 'agent',
        agent: 'backend-developer',
        task: 'Implement REST API',
        inputs: {
          architecture: 'architecture.tech-stack',
          schema: 'database-design.schema'
        },
        outputs: ['api-code', 'api-tests']
      },

      {
        id: 'frontend-ui',
        type: 'agent',
        agent: 'frontend-developer',
        task: 'Build user interface',
        inputs: {
          specification: 'requirements.specification',
          api: 'backend-api.api-code'
        },
        outputs: ['ui-code', 'ui-tests']
      },

      {
        id: 'integration-tests',
        type: 'agent',
        agent: 'qa-engineer',
        task: 'Write integration tests',
        inputs: {
          backend: 'backend-api.api-code',
          frontend: 'frontend-ui.ui-code'
        },
        outputs: ['integration-tests']
      },

      {
        id: 'deployment',
        type: 'agent',
        agent: 'devops-engineer',
        task: 'Create deployment configuration',
        inputs: {
          backend: 'backend-api.api-code',
          frontend: 'frontend-ui.ui-code',
          tests: 'integration-tests.integration-tests'
        },
        outputs: ['dockerfile', 'kubernetes-config', 'ci-cd-pipeline']
      },

      {
        id: 'documentation',
        type: 'agent',
        agent: 'technical-writer',
        task: 'Generate documentation',
        inputs: {
          specification: 'requirements.specification',
          api: 'backend-api.api-code',
          deployment: 'deployment.dockerfile'
        },
        outputs: ['readme', 'api-docs', 'deployment-guide']
      }
    ],

    // Define edges (dependencies)
    edges: [
      // Requirements → Architecture, Database Design
      { from: 'requirements', to: 'architecture' },
      { from: 'requirements', to: 'database-design' },

      // Architecture → Database Design, Backend, Frontend
      { from: 'architecture', to: 'database-design' },
      { from: 'architecture', to: 'backend-api' },

      // Database → Backend
      { from: 'database-design', to: 'backend-api' },

      // Backend → Frontend (needs API contract)
      { from: 'backend-api', to: 'frontend-ui' },

      // Backend + Frontend → Integration Tests
      { from: 'backend-api', to: 'integration-tests' },
      { from: 'frontend-ui', to: 'integration-tests' },

      // All code → Deployment
      { from: 'backend-api', to: 'deployment' },
      { from: 'frontend-ui', to: 'deployment' },
      { from: 'integration-tests', to: 'deployment' },

      // Everything → Documentation
      { from: 'requirements', to: 'documentation' },
      { from: 'backend-api', to: 'documentation' },
      { from: 'deployment', to: 'documentation' }
    ]
  });

  console.log('\n📊 Workflow DAG:');
  console.log('Nodes:', workflow.nodes.length);
  console.log('Edges:', workflow.edges.length);
  console.log('Parallelizable:', workflow.analysis.maxParallelism);

  // Visualize execution plan
  console.log('\n🔄 Execution Plan:');
  workflow.executionPlan.forEach((stage, i) => {
    console.log(`\nStage ${i + 1} (parallel):`);
    stage.forEach(node => console.log(`  - ${node.id}: ${node.task}`));
  });

  // Execute workflow
  const result = await qdag.execute(workflow, {
    // Monitor progress
    onNodeStart: (node) => {
      console.log(`\n▶️  Starting: ${node.id}`);
    },
    onNodeComplete: (node, output) => {
      console.log(`✅ Completed: ${node.id} (${output.executionTime}ms)`);
    },
    onNodeError: (node, error) => {
      console.log(`❌ Failed: ${node.id} - ${error.message}`);
    }
  });

  console.log('\n✅ Pipeline Complete!');
  console.log('Total Execution Time:', result.totalExecutionTime, 'ms');
  console.log('Parallel Efficiency:', result.parallelEfficiency, '%');
  console.log('Artifacts Generated:', result.artifacts.length);

  return result;
}

/**
 * Example 2: Data Science Pipeline
 */
async function dataSciencePipeline() {
  console.log('\n' + '='.repeat(60));
  console.log('QDAG Example 2: Data Science Pipeline');
  console.log('='.repeat(60));

  const workflow = await qdag.createWorkflow({
    name: 'ml-model-training',
    description: 'Complete ML pipeline from data to deployment',

    nodes: [
      {
        id: 'data-collection',
        type: 'agent',
        task: 'Collect and aggregate data from sources',
        outputs: ['raw-data']
      },
      {
        id: 'data-cleaning',
        type: 'agent',
        task: 'Clean and validate data',
        inputs: { data: 'data-collection.raw-data' },
        outputs: ['clean-data', 'data-quality-report']
      },
      {
        id: 'feature-engineering',
        type: 'agent',
        task: 'Engineer features for ML model',
        inputs: { data: 'data-cleaning.clean-data' },
        outputs: ['features', 'feature-importance']
      },
      {
        id: 'train-test-split',
        type: 'agent',
        task: 'Split data into train/test sets',
        inputs: { data: 'feature-engineering.features' },
        outputs: ['train-set', 'test-set']
      },
      {
        id: 'model-training-1',
        type: 'agent',
        task: 'Train Random Forest model',
        inputs: { data: 'train-test-split.train-set' },
        outputs: ['rf-model', 'rf-metrics']
      },
      {
        id: 'model-training-2',
        type: 'agent',
        task: 'Train XGBoost model',
        inputs: { data: 'train-test-split.train-set' },
        outputs: ['xgb-model', 'xgb-metrics']
      },
      {
        id: 'model-training-3',
        type: 'agent',
        task: 'Train Neural Network',
        inputs: { data: 'train-test-split.train-set' },
        outputs: ['nn-model', 'nn-metrics']
      },
      {
        id: 'model-evaluation',
        type: 'agent',
        task: 'Evaluate all models on test set',
        inputs: {
          testData: 'train-test-split.test-set',
          rfModel: 'model-training-1.rf-model',
          xgbModel: 'model-training-2.xgb-model',
          nnModel: 'model-training-3.nn-model'
        },
        outputs: ['evaluation-report', 'best-model']
      },
      {
        id: 'model-deployment',
        type: 'agent',
        task: 'Deploy best model as API',
        inputs: { model: 'model-evaluation.best-model' },
        outputs: ['api-endpoint', 'deployment-config']
      }
    ],

    edges: [
      { from: 'data-collection', to: 'data-cleaning' },
      { from: 'data-cleaning', to: 'feature-engineering' },
      { from: 'feature-engineering', to: 'train-test-split' },
      { from: 'train-test-split', to: 'model-training-1' },
      { from: 'train-test-split', to: 'model-training-2' },
      { from: 'train-test-split', to: 'model-training-3' },
      { from: 'model-training-1', to: 'model-evaluation' },
      { from: 'model-training-2', to: 'model-evaluation' },
      { from: 'model-training-3', to: 'model-evaluation' },
      { from: 'train-test-split', to: 'model-evaluation' },
      { from: 'model-evaluation', to: 'model-deployment' }
    ]
  });

  console.log('\n🔬 ML Pipeline:');
  console.log('Training 3 models in parallel');
  console.log('Automatic best model selection');

  const result = await qdag.execute(workflow);

  console.log('\n✅ Pipeline Complete!');
  console.log('Best Model:', result.outputs['model-evaluation'].bestModel);
  console.log('Accuracy:', result.outputs['model-evaluation'].accuracy);
  console.log('API Endpoint:', result.outputs['model-deployment'].apiEndpoint);

  return result;
}

/**
 * Example 3: Conditional Branching
 */
async function conditionalWorkflow() {
  console.log('\n' + '='.repeat(60));
  console.log('QDAG Example 3: Conditional Branching');
  console.log('='.repeat(60));

  const workflow = await qdag.createWorkflow({
    name: 'adaptive-development',
    description: 'Workflow that adapts based on agent outputs',

    nodes: [
      {
        id: 'analyze-requirements',
        type: 'agent',
        task: 'Analyze project complexity',
        outputs: ['complexity-level']
      },
      {
        id: 'simple-implementation',
        type: 'agent',
        task: 'Build simple version',
        condition: 'analyze-requirements.complexity-level === "simple"',
        outputs: ['simple-code']
      },
      {
        id: 'complex-implementation',
        type: 'agent',
        task: 'Build complex version with architecture',
        condition: 'analyze-requirements.complexity-level === "complex"',
        outputs: ['complex-code', 'architecture']
      },
      {
        id: 'testing',
        type: 'agent',
        task: 'Run tests',
        inputs: {
          code: 'simple-implementation.simple-code || complex-implementation.complex-code'
        },
        outputs: ['test-results']
      },
      {
        id: 'deploy',
        type: 'agent',
        task: 'Deploy to production',
        condition: 'testing.test-results.passed === true',
        outputs: ['deployment-url']
      }
    ],

    edges: [
      { from: 'analyze-requirements', to: 'simple-implementation' },
      { from: 'analyze-requirements', to: 'complex-implementation' },
      { from: 'simple-implementation', to: 'testing' },
      { from: 'complex-implementation', to: 'testing' },
      { from: 'testing', to: 'deploy' }
    ]
  });

  const result = await qdag.execute(workflow);

  console.log('\n✅ Adaptive Workflow Complete!');
  console.log('Path Taken:', result.executedPath);
  console.log('Deployment:', result.outputs['deploy']?.deploymentUrl || 'Skipped (tests failed)');

  return result;
}

/**
 * Main execution
 */
async function main() {
  console.log('🚀 QDAG Orchestrator - DAG-based Agent Workflows');

  try {
    await agentDB.initialize();
    await ruvLLM.initialize();

    await softwareDevelopmentPipeline();
    await dataSciencePipeline();
    await conditionalWorkflow();

    console.log('\n' + '='.repeat(60));
    console.log('✅ All QDAG examples completed successfully!');
    console.log('='.repeat(60));

  } catch (error) {
    console.error('\n❌ Error:', error.message);
    process.exit(1);
  } finally {
    await agentDB.close();
  }
}

// Run if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}

export {
  qdag,
  agentDB,
  ruvLLM,
  softwareDevelopmentPipeline,
  dataSciencePipeline,
  conditionalWorkflow
};
