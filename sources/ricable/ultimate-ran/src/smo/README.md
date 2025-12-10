# SMO Integration - PM/FM Data Pipeline

Complete PM (Performance Management) and FM (Fault Management) data pipeline for Ericsson SMO integration with Titan RAN.

## Overview

This module provides real-time collection, streaming, and analysis of PM counters and FM alarms from Ericsson ENM/OSS, with advanced features including:

- **PM Counter Collection** (3GPP TS 28.552)
- **FM Alarm Handling** (3GPP TS 28.532)
- **Real-time Data Streaming** via midstream processor
- **Alarm Correlation** for root cause analysis
- **Self-Healing Agent Integration**
- **PM-FM Cross-Correlation**
- **agentdb Storage** for GNN training

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      SMO Manager                              │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────┐              ┌──────────────────┐      │
│  │  PM Collector   │              │   FM Handler     │      │
│  │                 │              │                  │      │
│  │ • 10-min ROP    │              │ • Real-time      │      │
│  │ • 3GPP TS 28.552│◄────┐   ┌───►│ • SSE Streaming  │      │
│  │ • KPI Calc      │     │   │    │ • Correlation    │      │
│  │ • Anomaly Det   │     │   │    │ • Self-Healing   │      │
│  └─────────────────┘     │   │    └──────────────────┘      │
│           │              │   │              │                │
│           │         ┌────▼───▼────┐         │                │
│           └────────►│  Midstream  │◄────────┘                │
│                     │  Processor  │                          │
│                     └─────────────┘                          │
│                            │                                 │
│                     ┌──────▼──────┐                          │
│                     │   agentdb   │                          │
│                     │ (GNN Training)                         │
│                     └─────────────┘                          │
└──────────────────────────────────────────────────────────────┘
```

## Components

### 1. PM Collector (`pm-collector.ts`)

Collects Performance Management counters from Ericsson ENM.

#### PM Counters (3GPP TS 28.552)

**Uplink:**
- `pmUlSinrMean`: Mean UL SINR (dB)
- `pmUlBler`: UL Block Error Rate (0-1)
- `pmPuschPrbUsage`: PUSCH PRB utilization (%)
- `pmUlRssi`: UL RSSI (dBm)

**Downlink:**
- `pmDlSinrMean`: Mean DL SINR (dB)
- `pmDlBler`: DL Block Error Rate (0-1)
- `pmPdschPrbUsage`: PDSCH PRB utilization (%)

**Accessibility KPIs:**
- `pmRrcConnEstabSucc`: RRC Connection Success count
- `pmRrcConnEstabAtt`: RRC Connection Attempts
- `pmCssr`: Call Setup Success Rate (calculated)

**Retainability KPIs:**
- `pmErabRelNormal`: Normal E-RAB releases
- `pmErabRelAbnormal`: Abnormal E-RAB releases (drops)
- `pmCallDropRate`: Call drop rate (calculated)

#### Features

- Configurable ROP (Result Output Period) - default 10 minutes
- Real-time streaming via midstream processor
- Automatic KPI calculation (CSSR, Drop Rate, etc.)
- Anomaly detection with configurable thresholds
- agentdb storage for historical analysis and GNN training
- Support for cell/sector/site aggregation

### 2. FM Handler (`fm-handler.ts`)

Handles Fault Management alarms from Ericsson ENM.

#### Alarm Structure (3GPP TS 28.532)

```typescript
interface FMAlarm {
  alarmId: string;
  severity: 'critical' | 'major' | 'minor' | 'warning' | 'cleared';
  probableCause: string;
  specificProblem: string;
  managedObject: string;  // Cell DN
  eventTime: Date;
  // ... additional fields
}
```

#### Features

- Real-time alarm polling from ENM
- **Server-Sent Events (SSE)** for live alarm streaming
- **Alarm Correlation** for root cause analysis
  - CASCADE: Severity-based cascading alarms
  - COMMON_CAUSE: Same probable cause
  - DUPLICATE: Identical alarms
  - TEMPORAL: Time-based correlation
- **Self-Healing Integration**
  - AUTO_RECOVERY: Automatic recovery actions
  - PARAMETER_TUNE: Parameter optimization
  - CELL_RESTART: Cell restart (escalation)
  - ESCALATE: Human intervention required
- Alarm acknowledgment and clearing
- Historical alarm storage

### 3. SMO Manager (`index.ts`)

Unified manager that integrates PM and FM pipelines with cross-correlation.

#### PM-FM Cross-Correlation

The SMO Manager correlates PM anomalies with FM alarms to identify root causes:

```typescript
// Example: Low SINR anomaly + "Signal Quality" alarm
// → Correlation: "Interference or coverage issue"

// Example: High drop rate + "Power Problem" alarm
// → Correlation: "RRU Power Issue causing RF degradation"
```

## Usage

### Basic Example

```typescript
import { SMOManager } from './smo';

const smo = new SMOManager({
  pm: {
    ropInterval: 600000,  // 10 minutes
    cells: ['CELL-001', 'CELL-002', 'CELL-003'],
    enableStreaming: true,
    storageEnabled: true
  },
  fm: {
    pollingInterval: 30000,  // 30 seconds
    enableSSE: true,
    enableAutoHealing: true
  },
  enableCrossCorrelation: true
});

// Event listeners
smo.on('pm_anomaly', (anomaly) => {
  console.log(`PM Anomaly: ${anomaly.type} in ${anomaly.cellId}`);
});

smo.on('fm_alarm', (alarm) => {
  console.log(`FM Alarm: [${alarm.severity}] ${alarm.specificProblem}`);
});

smo.on('pmfm_correlation', (correlation) => {
  console.log(`Correlation: ${correlation.likelyCause}`);
});

// Start
await smo.start();
```

### PM Collector Standalone

```typescript
import { PMCollector } from './smo/pm-collector';

const pmCollector = new PMCollector({
  ropInterval: 600000,
  cells: ['CELL-001', 'CELL-002'],
  counters: [
    'pmUlSinrMean',
    'pmDlSinrMean',
    'pmCssr',
    'pmCallDropRate'
  ]
});

pmCollector.on('anomaly_detected', (anomaly) => {
  // Handle anomaly
});

await pmCollector.start();
```

### FM Handler Standalone

```typescript
import { FMHandler } from './smo/fm-handler';

const fmHandler = new FMHandler({
  pollingInterval: 30000,
  enableSSE: true,
  enableAutoHealing: true
});

fmHandler.on('correlation_detected', (correlation) => {
  console.log(`Root Cause: ${correlation.rootCause.specificProblem}`);
  console.log(`Symptoms: ${correlation.symptoms.length} alarms`);
});

await fmHandler.start();
```

## Events

### PM Collector Events

- `started`: PM collector started
- `stopped`: PM collector stopped
- `collection_complete`: PM collection completed
- `pm_received`: Individual PM data point received
- `anomaly_detected`: PM anomaly detected
- `batch_flushed`: Midstream batch flushed
- `pm_stored`: PM data stored to agentdb
- `cells_added`: Cells added to collection
- `cells_removed`: Cells removed from collection

### FM Handler Events

- `started`: FM handler started
- `stopped`: FM handler stopped
- `poll_complete`: Alarm poll completed
- `alarm_processed`: Alarm processed
- `alarm_cleared`: Alarm cleared
- `alarm_acknowledged`: Alarm acknowledged
- `correlation_detected`: Alarm correlation detected
- `self_healing_triggered`: Self-healing action triggered
- `self_healing_completed`: Self-healing action completed
- `alarm_event`: SSE alarm event

### SMO Manager Events

- `started`: SMO manager started
- `stopped`: SMO manager stopped
- `pm_anomaly`: PM anomaly detected
- `fm_alarm`: FM alarm received
- `pmfm_correlation`: PM-FM correlation detected
- `alarm_correlation`: Alarm correlation detected
- `self_healing`: Self-healing action
- `pm_collection_complete`: PM collection complete
- `fm_poll_complete`: FM poll complete
- `healing_validated`: Healing action validated with PM

## Configuration

### PMCollectorConfig

```typescript
interface PMCollectorConfig {
  ropInterval: number;          // ROP interval in ms (default: 600000)
  enmEndpoint?: string;         // ENM API endpoint
  cells: string[];              // Cell DNs to collect from
  counters: string[];           // PM counters to collect
  enableStreaming: boolean;     // Enable midstream (default: true)
  storageEnabled: boolean;      // Enable agentdb storage (default: true)
  aggregationLevel: 'cell' | 'sector' | 'site';
}
```

### FMHandlerConfig

```typescript
interface FMHandlerConfig {
  enmEndpoint?: string;         // ENM API endpoint
  pollingInterval: number;      // Alarm poll interval in ms
  enableSSE: boolean;           // Enable Server-Sent Events (default: true)
  ssePort: number;              // SSE server port (default: 3001)
  correlationWindow: number;    // Correlation window in ms (default: 300000)
  enableAutoHealing: boolean;   // Enable self-healing (default: true)
  severityFilter?: AlarmSeverity[];  // Filter by severity
  storageEnabled: boolean;      // Enable agentdb storage (default: true)
}
```

### SMOManagerConfig

```typescript
interface SMOManagerConfig {
  pm?: Partial<PMCollectorConfig>;
  fm?: Partial<FMHandlerConfig>;
  enableCrossCorrelation?: boolean;  // PM-FM correlation (default: true)
  autoTuneThresholds?: boolean;      // Auto-adjust thresholds (default: true)
}
```

## Integration with Titan RAN

### agentdb Storage

PM and FM data are stored in agentdb for:
- GNN (Graph Neural Network) training
- Historical trend analysis
- Spatial learning with ruvector
- Episode memory for reinforcement learning

### Midstream Integration

The midstream processor provides:
- Buffered real-time streaming
- Automatic batching and flushing
- Event emission for downstream consumers
- Flow entropy calculation for anomaly detection

### Self-Healing Integration

Self-healing actions integrate with:
- Council agents for decision-making
- SPARC governance for safety validation
- PM validation to verify healing effectiveness

## Running Examples

```bash
# Compile TypeScript
npm run build

# Run basic SMO manager example
node dist/smo/example.js 1

# Run PM collector standalone
node dist/smo/example.js 2

# Run FM handler standalone
node dist/smo/example.js 3

# Run all examples
node dist/smo/example.js all
```

## Testing

```bash
# Run integration tests
npm test

# Run with coverage
npm test -- --coverage
```

## Performance Targets

- **PM Collection**: < 1s per ROP
- **FM Polling**: < 500ms per poll
- **Alarm Correlation**: < 100ms
- **Midstream Flush**: < 50ms
- **agentdb Write**: < 10ms

## 3GPP Standards Compliance

- **3GPP TS 28.552**: Performance Management (PM)
- **3GPP TS 28.532**: Fault Management (FM)
- **3GPP TS 28.554**: KPI definitions
- **3GPP TS 32.435**: PM file format
- **ITU-T X.733**: Alarm severity levels

## Future Enhancements

- [ ] Real ENM API integration (currently mocked)
- [ ] WebSocket support for push-based alarms
- [ ] Machine learning-based correlation
- [ ] Predictive alarm analysis
- [ ] Multi-vendor support (Nokia, Huawei)
- [ ] Grafana dashboard integration
- [ ] Alarm pattern recognition
- [ ] Automated threshold tuning

## License

Proprietary - Ericsson Autonomous Networks Division
