/**
 * DAA Workflow Examples
 *
 * Demonstrates complex workflow orchestration patterns
 */

import { DAA } from 'daa-sdk';

async function runWorkflowExamples() {
  console.log('🔄 DAA Workflow Examples\n');

  const daa = new DAA({
    orchestrator: {
      enableMRAP: true,
      workflowEngine: true,
    },
  });

  await daa.init();

  // Example 1: Sequential Processing Workflow
  console.log('1️⃣  Sequential Processing Workflow');
  console.log('----------------------------------\n');

  const dataProcessingWorkflow = {
    id: 'data-processing',
    name: 'Data Processing Pipeline',
    type: 'sequential',
    steps: [
      {
        id: 'fetch',
        name: 'Fetch Data',
        type: 'http',
        config: {
          url: 'https://api.example.com/data',
          method: 'GET',
          headers: { 'Authorization': 'Bearer token' },
        },
      },
      {
        id: 'validate',
        name: 'Validate Schema',
        type: 'validation',
        config: {
          schema: {
            type: 'object',
            required: ['id', 'name', 'value'],
          },
        },
      },
      {
        id: 'transform',
        name: 'Transform Data',
        type: 'function',
        config: {
          fn: 'transformData',
          mapping: {
            id: 'entityId',
            name: 'displayName',
            value: 'amount',
          },
        },
      },
      {
        id: 'store',
        name: 'Store in Database',
        type: 'database',
        config: {
          table: 'processed_data',
          operation: 'upsert',
        },
      },
    ],
  };

  console.log('Workflow:', dataProcessingWorkflow.name);
  console.log('Type:', dataProcessingWorkflow.type);
  console.log('Steps:');
  dataProcessingWorkflow.steps.forEach((step, i) => {
    console.log(`  ${i + 1}. ${step.name} (${step.type})`);
  });
  console.log();

  // Example 2: Parallel Execution Workflow
  console.log('2️⃣  Parallel Execution Workflow');
  console.log('-------------------------------\n');

  const parallelAnalysisWorkflow = {
    id: 'parallel-analysis',
    name: 'Parallel Data Analysis',
    type: 'parallel',
    branches: [
      {
        id: 'sentiment-analysis',
        name: 'Sentiment Analysis',
        steps: [
          { id: 'load-nlp-model', type: 'ml', config: { model: 'sentiment' } },
          { id: 'analyze-text', type: 'function', config: { fn: 'analyzeSentiment' } },
          { id: 'save-sentiment', type: 'database', config: { table: 'sentiments' } },
        ],
      },
      {
        id: 'entity-extraction',
        name: 'Entity Extraction',
        steps: [
          { id: 'load-ner-model', type: 'ml', config: { model: 'ner' } },
          { id: 'extract-entities', type: 'function', config: { fn: 'extractEntities' } },
          { id: 'save-entities', type: 'database', config: { table: 'entities' } },
        ],
      },
      {
        id: 'topic-modeling',
        name: 'Topic Modeling',
        steps: [
          { id: 'load-lda-model', type: 'ml', config: { model: 'lda' } },
          { id: 'identify-topics', type: 'function', config: { fn: 'modelTopics' } },
          { id: 'save-topics', type: 'database', config: { table: 'topics' } },
        ],
      },
    ],
    merge: {
      id: 'aggregate-results',
      type: 'function',
      config: { fn: 'aggregateAnalysis' },
    },
  };

  console.log('Workflow:', parallelAnalysisWorkflow.name);
  console.log('Type:', parallelAnalysisWorkflow.type);
  console.log('Parallel branches:', parallelAnalysisWorkflow.branches.length);
  parallelAnalysisWorkflow.branches.forEach((branch, i) => {
    console.log(`  Branch ${i + 1}: ${branch.name} (${branch.steps.length} steps)`);
  });
  console.log('Merge strategy:', parallelAnalysisWorkflow.merge.config.fn);
  console.log();

  // Example 3: Conditional Workflow
  console.log('3️⃣  Conditional Workflow');
  console.log('-----------------------\n');

  const conditionalWorkflow = {
    id: 'fraud-detection',
    name: 'Fraud Detection Workflow',
    type: 'conditional',
    entry: 'check-transaction',
    steps: {
      'check-transaction': {
        type: 'function',
        config: { fn: 'calculateRiskScore' },
        next: {
          'score < 30': 'approve-transaction',
          '30 <= score < 70': 'manual-review',
          'score >= 70': 'block-transaction',
        },
      },
      'approve-transaction': {
        type: 'function',
        config: { fn: 'approveTransaction' },
        next: 'notify-user',
      },
      'manual-review': {
        type: 'human-task',
        config: { queue: 'fraud-review', sla: '2h' },
        next: {
          approved: 'approve-transaction',
          rejected: 'block-transaction',
        },
      },
      'block-transaction': {
        type: 'function',
        config: { fn: 'blockTransaction' },
        next: 'notify-user',
      },
      'notify-user': {
        type: 'notification',
        config: { channel: 'email', template: 'transaction-status' },
        next: null, // End of workflow
      },
    },
  };

  console.log('Workflow:', conditionalWorkflow.name);
  console.log('Type:', conditionalWorkflow.type);
  console.log('Decision tree:');
  console.log('  Risk < 30%: ✅ Auto-approve');
  console.log('  Risk 30-70%: 👤 Manual review (2h SLA)');
  console.log('  Risk > 70%: 🚫 Auto-block');
  console.log();

  // Example 4: Event-Driven Workflow
  console.log('4️⃣  Event-Driven Workflow');
  console.log('------------------------\n');

  const eventDrivenWorkflow = {
    id: 'order-fulfillment',
    name: 'Order Fulfillment Workflow',
    type: 'event-driven',
    triggers: [
      { event: 'order.created', action: 'start-workflow' },
      { event: 'payment.confirmed', action: 'process-order' },
      { event: 'shipment.dispatched', action: 'notify-customer' },
      { event: 'order.cancelled', action: 'refund-payment' },
    ],
    states: {
      pending: {
        on: {
          'payment.confirmed': 'processing',
          'order.cancelled': 'cancelled',
        },
      },
      processing: {
        on: {
          'inventory.allocated': 'ready-to-ship',
          'inventory.unavailable': 'backorder',
        },
      },
      'ready-to-ship': {
        on: {
          'shipment.dispatched': 'shipped',
        },
      },
      shipped: {
        on: {
          'shipment.delivered': 'completed',
        },
      },
      backorder: {
        on: {
          'inventory.restocked': 'processing',
          'backorder.timeout': 'cancelled',
        },
      },
      cancelled: {
        terminal: true,
      },
      completed: {
        terminal: true,
      },
    },
  };

  console.log('Workflow:', eventDrivenWorkflow.name);
  console.log('Type:', eventDrivenWorkflow.type);
  console.log('Triggers:');
  eventDrivenWorkflow.triggers.forEach((trigger) => {
    console.log(`  📡 ${trigger.event} → ${trigger.action}`);
  });
  console.log('States:', Object.keys(eventDrivenWorkflow.states).length);
  console.log();

  // Example 5: Compensation (Saga) Pattern
  console.log('5️⃣  Compensation Pattern (Saga)');
  console.log('-------------------------------\n');

  const sagaWorkflow = {
    id: 'distributed-transaction',
    name: 'Distributed Transaction Saga',
    type: 'saga',
    steps: [
      {
        id: 'reserve-inventory',
        action: 'reserveInventory',
        compensation: 'releaseInventory',
      },
      {
        id: 'charge-payment',
        action: 'chargePayment',
        compensation: 'refundPayment',
      },
      {
        id: 'create-shipment',
        action: 'createShipment',
        compensation: 'cancelShipment',
      },
      {
        id: 'send-confirmation',
        action: 'sendConfirmationEmail',
        compensation: 'sendCancellationEmail',
      },
    ],
    onFailure: 'rollback', // Execute compensations in reverse order
  };

  console.log('Workflow:', sagaWorkflow.name);
  console.log('Type:', sagaWorkflow.type);
  console.log('Forward steps:');
  sagaWorkflow.steps.forEach((step, i) => {
    console.log(`  ${i + 1}. ${step.action}`);
  });
  console.log('Compensation strategy:', sagaWorkflow.onFailure);
  console.log('On failure, compensations execute in reverse:');
  [...sagaWorkflow.steps].reverse().forEach((step, i) => {
    console.log(`  ${i + 1}. ${step.compensation}`);
  });
  console.log();

  console.log('🎉 Workflow examples completed!\n');
}

// Run examples
runWorkflowExamples().catch((error) => {
  console.error('❌ Error:', error);
  process.exit(1);
});
