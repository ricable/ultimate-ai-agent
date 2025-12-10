# Full-Stack DAA Agent Template

Complete DAA ecosystem with orchestrator, QuDAG networking, and workflow engine.

## Features

- **MRAP Orchestrator**: Monitor-Reason-Act-Plan autonomous loop
- **Workflow Engine**: Complex workflow orchestration with multiple patterns
- **QuDAG Network**: Quantum-resistant P2P networking and token exchange
- **Rules Engine**: Business logic and decision-making
- **Token Economy**: rUv token management and dynamic fees
- **Event Bus**: Pub/sub messaging for decoupled components

## Quick Start

### Installation

```bash
npm install
```

### Build

```bash
npm run build
```

### Run Examples

```bash
# Run main full-stack demo
npm start

# Run orchestrator examples
npm run test:orchestrator

# Run QuDAG network examples
npm run test:qudag

# Run workflow examples
npm run test:workflows
```

## Project Structure

```
full-stack/
├── src/
│   ├── index.ts          # Main full-stack demo
│   ├── orchestrator.ts   # MRAP and rules examples
│   ├── qudag.ts         # Network and token examples
│   └── workflows.ts     # Workflow patterns
├── package.json         # Dependencies and scripts
├── tsconfig.json        # TypeScript configuration
└── README.md           # This file
```

## Architecture

### MRAP Orchestrator

The Monitor-Reason-Act-Plan (MRAP) loop enables continuous self-improvement:

```typescript
const daa = new DAA({
  orchestrator: {
    enableMRAP: true,
    workflowEngine: true,
    eventBusSize: 1000,
  },
});

await daa.init();
await daa.orchestrator.start();
```

**MRAP Phases:**
1. **Monitor**: Observe system state and environment
2. **Reason**: Analyze observations and make decisions
3. **Act**: Execute chosen actions
4. **Plan**: Set goals and adapt strategies

### Workflow Engine

Supports multiple workflow patterns:

**Sequential Workflows:**
```typescript
const workflow = {
  type: 'sequential',
  steps: [
    { id: 'fetch', type: 'http' },
    { id: 'validate', type: 'validation' },
    { id: 'transform', type: 'function' },
    { id: 'store', type: 'database' },
  ],
};
```

**Parallel Workflows:**
```typescript
const workflow = {
  type: 'parallel',
  branches: [
    { name: 'sentiment-analysis', steps: [...] },
    { name: 'entity-extraction', steps: [...] },
    { name: 'topic-modeling', steps: [...] },
  ],
  merge: { type: 'function', fn: 'aggregateResults' },
};
```

**Conditional Workflows:**
```typescript
const workflow = {
  type: 'conditional',
  steps: {
    'check-risk': {
      next: {
        'low': 'approve',
        'medium': 'review',
        'high': 'block',
      },
    },
  },
};
```

**Event-Driven Workflows:**
```typescript
const workflow = {
  type: 'event-driven',
  triggers: [
    { event: 'order.created', action: 'start' },
    { event: 'payment.confirmed', action: 'process' },
  ],
  states: {
    pending: { on: { 'payment.confirmed': 'processing' } },
    processing: { on: { 'inventory.allocated': 'shipping' } },
  },
};
```

**Saga Pattern (Compensation):**
```typescript
const workflow = {
  type: 'saga',
  steps: [
    { action: 'reserveInventory', compensation: 'releaseInventory' },
    { action: 'chargePayment', compensation: 'refundPayment' },
    { action: 'createShipment', compensation: 'cancelShipment' },
  ],
  onFailure: 'rollback',
};
```

### Rules Engine

Define business logic as declarative rules:

```typescript
const rules = [
  {
    id: 'resource-allocation',
    condition: 'agent.cpuUsage > 80',
    action: 'spawn-additional-agent',
    priority: 'high',
  },
  {
    id: 'cost-optimization',
    condition: 'task.estimatedCost > agent.balance',
    action: 'reject-task',
    priority: 'critical',
  },
];

const result = await daa.rules.evaluate(context);
```

### Token Economy

Manage rUv tokens with dynamic fees:

```typescript
// Get balance
const balance = await daa.economy.getBalance('agent-001');

// Transfer tokens
await daa.economy.transfer('agent-001', 'agent-002', 100);

// Calculate dynamic fee
const fee = await daa.economy.calculateFee(operation);
```

### QuDAG Network

Quantum-resistant P2P networking:

```typescript
// Secure communication
const mlkem = daa.crypto.mlkem();
const keypair = mlkem.generateKeypair();
const { ciphertext, sharedSecret } = mlkem.encapsulate(keypair.publicKey);

// Token transactions
const tx = await daa.exchange.createTransaction('alice', 'bob', 250);
const signedTx = await daa.exchange.signTransaction(tx, privateKey);
const isValid = daa.exchange.verifyTransaction(signedTx);
await daa.exchange.submitTransaction(signedTx);
```

## Use Cases

### 1. Distributed ML Training Pipeline

```typescript
const mlWorkflow = {
  id: 'ml-training',
  steps: [
    { id: 'ingest', type: 'function', config: { parallel: true, shards: 10 } },
    { id: 'preprocess', type: 'function', config: { validation: true } },
    { id: 'train', type: 'ml', config: { distributed: true, nodes: 5 } },
    { id: 'evaluate', type: 'function', config: { metrics: ['accuracy', 'loss'] } },
    { id: 'deploy', type: 'deployment', config: { strategy: 'blue-green' } },
  ],
};
```

### 2. Fraud Detection System

```typescript
const fraudWorkflow = {
  id: 'fraud-detection',
  type: 'conditional',
  steps: {
    'check-transaction': {
      next: {
        'score < 30': 'approve',
        '30 <= score < 70': 'manual-review',
        'score >= 70': 'block',
      },
    },
  },
};
```

### 3. Order Fulfillment

```typescript
const orderWorkflow = {
  id: 'order-fulfillment',
  type: 'saga',
  steps: [
    { action: 'reserveInventory', compensation: 'releaseInventory' },
    { action: 'chargePayment', compensation: 'refundPayment' },
    { action: 'createShipment', compensation: 'cancelShipment' },
  ],
};
```

## Performance

- **Native Bindings**: Full performance with NAPI-rs
- **Parallel Execution**: Multiple workflows run concurrently
- **Event-Driven**: Non-blocking async operations
- **Zero-Copy**: Optimized memory operations

## Security

- **Quantum-Resistant**: ML-KEM-768 and ML-DSA algorithms
- **Encrypted Communication**: All network traffic encrypted
- **Signature Verification**: All transactions digitally signed
- **Access Control**: Role-based permissions

## Next Steps

1. Customize workflows for your use case
2. Add business rules to the rules engine
3. Deploy to production with load balancing
4. Integrate with external systems
5. Explore the [ML Training Template](../ml-training)

## Documentation

- [DAA SDK Docs](https://github.com/ruvnet/daa)
- [MRAP Orchestrator](https://github.com/ruvnet/daa/tree/main/crates/orchestrator)
- [QuDAG Network](https://github.com/ruvnet/daa/tree/main/crates/qudag)
- [Workflow Patterns](https://github.com/ruvnet/daa/tree/main/crates/orchestrator/workflows)

## License

MIT
