# QuDAG Exchange Interface Specifications

## Overview

The Interface Agent has created three interface crates that provide access to the QuDAG Exchange system. All interfaces are ready and waiting for the Core Implementation Agent to complete the actual exchange logic.

## CLI Interface Specification

### Binary: `qudag-exchange`

**Commands:**
- `create-account --name <name> [--initial-balance <amount>]`
- `balance [--account <id>]`
- `transfer --to <recipient> --amount <amount> [--memo <text>]`
- `resource-status [--detailed]`
- `consensus-info [--peers]`
- `config show|init|set <key> <value>`

**Output Formats:**
- Text (default) - Human-readable colored output
- JSON (`--output json`) - Machine-readable format

**Features:**
- Secure password prompts for vault access
- Configuration file support
- Colored terminal output
- Table formatting for lists
- Transaction confirmation prompts

## API Interface Specification

### Server: `qudag-exchange-server`

**Base URL:** `http://localhost:8585/api/v1`

**Endpoints:**

```
POST   /accounts                 Create account
GET    /accounts/:id/balance     Get balance
POST   /transactions             Submit transaction  
GET    /transactions/:id         Get transaction status
GET    /resources/status         Resource usage
GET    /resources/costs          Operation costs
GET    /consensus/info           Consensus status
GET    /consensus/peers          Peer list
GET    /health                   Health check
```

**Authentication:**
- JWT tokens in Authorization header
- `Bearer <token>` format

**Request/Response Format:**
- JSON with proper content-type headers
- Error responses include error code and message

## WASM Interface Specification

### Package: `qudag-exchange-wasm`

**Classes:**
- `QuDAGExchange` - Main exchange interface
- `WasmAccount` - Account information
- `WasmTransaction` - Transaction data

**Methods:**
- `new QuDAGExchange()` - Create instance
- `create_account(name: string): Promise<WasmAccount>`
- `get_balance(account_id: string): Promise<number>`
- `transfer(from: string, to: string, amount: number): Promise<WasmTransaction>`
- `get_resource_costs(): object`

**Storage:**
- Uses browser localStorage for persistence
- Node.js uses memory storage

**Build Targets:**
- `pkg-web/` - ES modules for browsers
- `pkg-node/` - CommonJS for Node.js  
- `pkg-bundler/` - For webpack/Rollup

## Integration Requirements

All interfaces expect the core to provide:

1. **Account Management**
   - Create accounts with quantum-resistant keys
   - Query balances from ledger
   - Validate account existence

2. **Transaction Processing**
   - Submit signed transactions
   - Query transaction status
   - Validate signatures with qudag-crypto

3. **Resource Metering**
   - Track resource usage
   - Deduct rUv for operations
   - Provide usage statistics

4. **Consensus Integration**
   - Submit transactions to DAG
   - Query confirmation status
   - Get network information

## Error Handling

All interfaces handle errors consistently:
- Account not found
- Insufficient balance
- Invalid transaction
- Network errors
- Authentication failures

Errors are properly typed and include descriptive messages.