# DAA Orchestrator - Node.js Bindings

High-performance Node.js bindings for the DAA (Decentralized Autonomous Agents) Orchestrator, built with NAPI-rs.

## Features

- **MRAP Loop**: Monitor, Reason, Act, Plan autonomous loop for continuous operation
- **Workflow Engine**: Create and execute complex multi-step workflows
- **Rules Engine**: Define and evaluate business rules and policies
- **Economy Manager**: Token management, accounts, and trading operations
- **High Performance**: Native Rust performance with zero-copy data transfer
- **Type Safe**: Auto-generated TypeScript definitions
- **Cross-Platform**: Support for Linux, macOS, and Windows

## Installation

```bash
npm install @daa/orchestrator
```

## Quick Start

### Initialize the Library

```javascript
const { initialize, Orchestrator } = require('@daa/orchestrator');

// Initialize with logging
initialize('info');

// Create and start the orchestrator
const orchestrator = new Orchestrator({
  enabled: true,
  loopIntervalMs: 1000,
  maxTasksPerIteration: 10,
  enableLearning: true
});

await orchestrator.start();
```

### MRAP Loop (Monitor, Reason, Act, Plan)

The orchestrator provides an autonomous loop that continuously monitors system state, reasons about decisions, acts on them, and plans future actions.

```javascript
// Start the MRAP loop
await orchestrator.start();

// Monitor system state
const state = await orchestrator.monitor();
console.log('System state:', state.state);
console.log('Uptime:', state.uptimeSeconds, 'seconds');
console.log('Is healthy:', state.isHealthy);

// Health check
const isHealthy = await orchestrator.healthCheck();
if (!isHealthy) {
  console.error('System is unhealthy!');
}

// Get current configuration
const config = orchestrator.getConfig();
console.log('Loop interval:', config.loopIntervalMs, 'ms');

// Get statistics
const stats = await orchestrator.getStatistics();
console.log('Total iterations:', stats.totalIterations);
console.log('Active tasks:', stats.activeTasks);

// Stop the loop
await orchestrator.stop();

// Restart the loop
await orchestrator.restart();
```

### Workflow Engine

Create and execute complex multi-step workflows with parallel or sequential execution.

```javascript
const { WorkflowEngine } = require('@daa/orchestrator');

const workflowEngine = new WorkflowEngine({
  maxExecutionTimeMs: 3600000,  // 1 hour
  maxSteps: 100,
  parallelExecution: true
});

await workflowEngine.start();

// Define a workflow
const workflow = {
  id: 'data-pipeline-1',
  name: 'Data Processing Pipeline',
  steps: [
    {
      id: 'step-1',
      stepType: 'fetch_data',
      parameters: JSON.stringify({
        source: 'api',
        endpoint: 'https://api.example.com/data'
      })
    },
    {
      id: 'step-2',
      stepType: 'transform',
      parameters: JSON.stringify({
        operation: 'normalize',
        fields: ['price', 'volume']
      })
    },
    {
      id: 'step-3',
      stepType: 'validate',
      parameters: JSON.stringify({
        rules: ['non_negative', 'within_bounds']
      })
    },
    {
      id: 'step-4',
      stepType: 'store',
      parameters: JSON.stringify({
        destination: 'database',
        table: 'processed_data'
      })
    }
  ]
};

// Validate the workflow
const isValid = await workflowEngine.validateWorkflow(workflow);
console.log('Workflow valid:', isValid);

// Execute the workflow
const result = await workflowEngine.executeWorkflow(workflow);
console.log('Workflow status:', result.status);
result.results.forEach(stepResult => {
  console.log(`Step ${stepResult.stepId}: ${stepResult.status}`);
  console.log('Output:', JSON.parse(stepResult.output));
});

// Get active workflow count
const activeCount = await workflowEngine.getActiveCount();
console.log('Active workflows:', activeCount);
```

### Rules Engine

Define and evaluate business rules for policy enforcement and decision automation.

```javascript
const { RulesEngine } = require('@daa/orchestrator');

const rulesEngine = new RulesEngine();

// Define a rule
const balanceRule = {
  id: 'balance-check-1',
  name: 'Minimum Balance Check',
  description: 'Ensure account has minimum balance before transactions',
  conditions: JSON.stringify([
    {
      GreaterThan: {
        field: 'balance',
        value: 100
      }
    }
  ]),
  actions: JSON.stringify([
    {
      Log: {
        level: 'Info',
        message: 'Balance check passed'
      }
    }
  ]),
  priority: 10,
  enabled: true
};

// Validate the rule
const isValid = await rulesEngine.validateRule(balanceRule);
console.log('Rule valid:', isValid);

// Add the rule
await rulesEngine.addRule(balanceRule);

// Evaluate rules with context
const result = await rulesEngine.evaluate({
  data: JSON.stringify({
    balance: 500,
    transaction_amount: 100,
    user_id: 'user-123',
    transaction_type: 'transfer'
  })
});

console.log('Evaluation result:', result.resultType);
if (result.resultType === 'deny') {
  console.log('Denial reason:', result.message);
} else if (result.resultType === 'modified') {
  console.log('Modifications:', JSON.parse(result.modifications));
}

// Evaluate a specific rule
const specificResult = await rulesEngine.evaluateRule('balance-check-1', {
  data: JSON.stringify({ balance: 50 })
});

// Get rule count
const ruleCount = await rulesEngine.getRuleCount();
console.log('Total rules:', ruleCount);
```

### Economy Manager

Manage tokens, accounts, and trading operations.

```javascript
const { EconomyManager } = require('@daa/orchestrator');

const economyManager = new EconomyManager();

// Create an account
const account = await economyManager.createAccount('agent-123');
console.log('Account ID:', account.id);
console.log('Agent ID:', account.agentId);
console.log('Status:', account.status);

// Set initial balance (for testing)
await economyManager.setBalance(account.id, 'rUv', 1000.0);
await economyManager.setBalance(account.id, 'USD', 5000.0);

// Get account information
const accountInfo = await economyManager.getAccount(account.id);
console.log('Account info:', accountInfo);

// Get balance for a specific token
const ruvBalance = await economyManager.getBalance(account.id, 'rUv');
console.log('rUv balance:', ruvBalance.amount);

// Get all balances
const allBalances = await economyManager.getAllBalances(account.id);
allBalances.forEach(balance => {
  console.log(`${balance.token}: ${balance.amount}`);
});

// Transfer tokens between accounts
const account2 = await economyManager.createAccount('agent-456');
const transferResult = await economyManager.transfer({
  fromAccount: account.id,
  toAccount: account2.id,
  token: 'rUv',
  amount: 100.0,
  memo: 'Payment for services'
});

console.log('Transfer ID:', transferResult.transactionId);
console.log('Status:', transferResult.status);

// Create a trading order
const order = await economyManager.createOrder({
  id: 'order-1',
  symbol: 'rUv/USD',
  orderType: 'limit',
  side: 'buy',
  quantity: 10.0,
  price: 50.0,
  status: 'pending'
});

console.log('Order created:', order.id);

// Get account count
const accountCount = await economyManager.getAccountCount();
console.log('Total accounts:', accountCount);
```

### Complete Example

Here's a complete example that uses all components together:

```javascript
const {
  initialize,
  Orchestrator,
  WorkflowEngine,
  RulesEngine,
  EconomyManager
} = require('@daa/orchestrator');

async function main() {
  // Initialize the library
  initialize('info');

  // Create economy manager and set up accounts
  const economyManager = new EconomyManager();
  const account = await economyManager.createAccount('agent-001');
  await economyManager.setBalance(account.id, 'rUv', 10000.0);

  // Create rules engine and add rules
  const rulesEngine = new RulesEngine();
  await rulesEngine.addRule({
    id: 'min-balance',
    name: 'Minimum Balance Rule',
    conditions: JSON.stringify([
      { GreaterThan: { field: 'balance', value: 100 } }
    ]),
    actions: JSON.stringify([
      { Log: { level: 'Info', message: 'Balance sufficient' } }
    ]),
    enabled: true
  });

  // Create workflow engine and define workflow
  const workflowEngine = new WorkflowEngine({
    parallelExecution: true,
    maxSteps: 50
  });
  await workflowEngine.start();

  const workflow = {
    id: 'autonomous-trading',
    name: 'Autonomous Trading Workflow',
    steps: [
      {
        id: 'check-balance',
        stepType: 'validate_balance',
        parameters: JSON.stringify({ accountId: account.id })
      },
      {
        id: 'analyze-market',
        stepType: 'market_analysis',
        parameters: JSON.stringify({ symbol: 'rUv/USD' })
      },
      {
        id: 'execute-trade',
        stepType: 'place_order',
        parameters: JSON.stringify({
          accountId: account.id,
          orderType: 'market',
          side: 'buy'
        })
      }
    ]
  };

  // Start the orchestrator with MRAP loop
  const orchestrator = new Orchestrator({
    enabled: true,
    loopIntervalMs: 5000,
    enableLearning: true
  });
  await orchestrator.start();

  // Monitor system state
  setInterval(async () => {
    const state = await orchestrator.monitor();
    const stats = await orchestrator.getStatistics();

    console.log('=== System Status ===');
    console.log('State:', state.state);
    console.log('Uptime:', Math.floor(state.uptimeSeconds), 'seconds');
    console.log('Healthy:', state.isHealthy);
    console.log('Active tasks:', stats.activeTasks);
    console.log('====================');
  }, 10000);

  // Execute workflow periodically
  setInterval(async () => {
    const result = await workflowEngine.executeWorkflow(workflow);
    console.log('Workflow executed:', result.status);
  }, 30000);

  // Graceful shutdown
  process.on('SIGINT', async () => {
    console.log('Shutting down...');
    await orchestrator.stop();
    process.exit(0);
  });
}

main().catch(console.error);
```

## API Reference

### Orchestrator

#### Constructor
- `new Orchestrator(config: OrchestratorConfig)`

#### Methods
- `start(): Promise<void>` - Start the MRAP loop
- `stop(): Promise<void>` - Stop the MRAP loop
- `restart(): Promise<void>` - Restart the MRAP loop
- `monitor(): Promise<SystemState>` - Get current system state
- `healthCheck(): Promise<boolean>` - Perform health check
- `getConfig(): OrchestratorConfig` - Get current configuration
- `getStatistics(): Promise<SystemStatistics>` - Get system statistics

### WorkflowEngine

#### Constructor
- `new WorkflowEngine(config?: WorkflowConfig)`

#### Methods
- `start(): Promise<void>` - Start the workflow engine
- `createWorkflow(workflow: Workflow): Promise<string>` - Create a workflow
- `executeWorkflow(workflow: Workflow): Promise<WorkflowResult>` - Execute a workflow
- `validateWorkflow(workflow: Workflow): Promise<boolean>` - Validate a workflow
- `getActiveCount(): Promise<number>` - Get active workflow count

### RulesEngine

#### Constructor
- `new RulesEngine()`

#### Methods
- `addRule(rule: Rule): Promise<void>` - Add a rule
- `evaluate(context: ExecutionContext): Promise<RuleResult>` - Evaluate all rules
- `evaluateRule(ruleId: string, context: ExecutionContext): Promise<RuleResult>` - Evaluate specific rule
- `validateRule(rule: Rule): Promise<boolean>` - Validate a rule
- `getRuleCount(): Promise<number>` - Get rule count

### EconomyManager

#### Constructor
- `new EconomyManager()`

#### Methods
- `createAccount(agentId: string): Promise<Account>` - Create an account
- `getAccount(accountId: string): Promise<Account>` - Get account info
- `getBalance(accountId: string, token: string): Promise<Balance>` - Get token balance
- `getAllBalances(accountId: string): Promise<Balance[]>` - Get all balances
- `transfer(transfer: TransferRequest): Promise<TransferResult>` - Transfer tokens
- `createOrder(order: TradeOrder): Promise<TradeOrder>` - Create a trading order
- `setBalance(accountId: string, token: string, amount: number): Promise<void>` - Set balance (testing)
- `getAccountCount(): Promise<number>` - Get account count

## TypeScript Support

This package includes auto-generated TypeScript definitions:

```typescript
import {
  Orchestrator,
  WorkflowEngine,
  RulesEngine,
  EconomyManager,
  OrchestratorConfig,
  SystemState
} from '@daa/orchestrator';

const config: OrchestratorConfig = {
  enabled: true,
  loopIntervalMs: 1000,
  enableLearning: true
};

const orchestrator = new Orchestrator(config);
```

## Performance

Built with Rust and NAPI-rs for maximum performance:

- **Zero-copy**: Direct memory access between Rust and Node.js
- **Async by default**: All operations are non-blocking
- **Minimal overhead**: Native Rust performance with minimal binding overhead
- **Efficient serialization**: Optimized JSON serialization for complex types

## Platform Support

Pre-built binaries are available for:

- Linux x64 (glibc and musl)
- Linux ARM64 (glibc and musl)
- macOS x64
- macOS ARM64 (Apple Silicon)
- Windows x64
- Windows ARM64

## Building from Source

```bash
# Install dependencies
npm install

# Build the native module
npm run build

# Build in debug mode
npm run build:debug

# Run tests
npm test
```

## Development

```bash
# Format code
npm run format

# Lint code
npm run lint

# Run benchmarks
npm run bench
```

## License

MIT OR Apache-2.0

## Contributing

Contributions are welcome! Please see the main DAA repository for contribution guidelines.

## Links

- [Main Repository](https://github.com/ruvnet/daa)
- [Documentation](https://github.com/ruvnet/daa/tree/main/docs)
- [Issues](https://github.com/ruvnet/daa/issues)
