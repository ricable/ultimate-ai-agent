# TITAN RAN Dashboard

AG-UI + OpenTUI Integration for Real-time RAN Optimization

## Overview

The TITAN Dashboard provides a comprehensive **Glass Box** operational experience for the autonomous RAN optimization system. It combines:

- **AG-UI Protocol**: Generative UI components for rich visualizations
- **OpenTUI**: Terminal-based interface for operators
- **SSE Streaming**: Real-time KPI and event updates
- **HITL Approval**: Human-in-the-loop safety controls

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TITAN Dashboard                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  AG-UI       │  │  OpenTUI     │  │  SSE Stream  │     │
│  │  Protocol    │  │  Terminal    │  │  API         │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              TitanDashboard (Core)                   │  │
│  │  - State Management                                  │  │
│  │  - Visualization Rendering                          │  │
│  │  - Parameter Controls                               │  │
│  │  - Approval Queue                                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. InterferenceHeatmap.tsx

D3.js/visx-based heatmap visualization for cell-to-cell interference.

**Features:**
- NxN interference matrix visualization
- Color-coded by interference level
- Interactive hover tooltips
- Click to select cells
- Customizable threshold

**Usage:**
```tsx
<InterferenceHeatmap
  cells={cellStatus}
  interferenceMatrix={matrix}
  threshold={-90}
  onCellClick={(cellId) => console.log('Selected:', cellId)}
/>
```

### 2. OptimizationTimeline.tsx

Timeline visualization of GNN decisions and council debates.

**Features:**
- Chronological event display
- Event type filtering
- Status filtering
- Expandable event details
- Parameter change tables
- KPI impact display

**Usage:**
```tsx
<OptimizationTimeline
  events={optimizationEvents}
  maxEvents={100}
  onEventClick={(event) => console.log('Event:', event)}
/>
```

### 3. ApprovalCard.tsx

HITL approval interface for critical RAN operations.

**Features:**
- Risk level indicators
- Safety check validation
- Parameter change preview
- Signature verification
- Expiration countdown
- Predicted impact display

**Usage:**
```tsx
<ApprovalCard
  approval={approvalRequest}
  onApprove={(id, signature) => handleApprove(id, signature)}
  onReject={(id, signature) => handleReject(id, signature)}
/>
```

### 4. ParameterSlider.tsx

Interactive controls for P0 (transmit power) and Alpha (antenna tilt) adjustments.

**Features:**
- Real-time value updates
- 3GPP bounds enforcement
- AI recommendations
- Quick presets (Min/Mid/Max)
- Cell KPI display
- Change delta tracking

**Usage:**
```tsx
<ParameterSlider
  cell={cellStatus}
  parameter={{
    name: 'power_dbm',
    label: 'Transmit Power (dBm)',
    min: -130,
    max: 46,
    step: 0.1,
    unit: 'dBm'
  }}
  onChange={(cellId, param, value) => handleChange(cellId, param, value)}
  onCommit={(cellId, param, value) => handleCommit(cellId, param, value)}
/>
```

## API Server

### SSE Streaming Endpoints

Real-time server-sent events for live updates:

- **GET /api/stream/kpi** - KPI counters (10s interval)
- **GET /api/stream/events** - Optimization events
- **GET /api/stream/alarms** - FM alarms
- **GET /api/stream/approvals** - HITL approvals

**Example:**
```javascript
const eventSource = new EventSource('http://localhost:8080/api/stream/kpi');

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('KPI Update:', data);
};
```

### REST Endpoints

- **GET /api/cells** - List all cells
- **GET /api/cells/:cellId** - Get cell details
- **POST /api/cells/:cellId/parameters** - Update cell parameters
- **GET /api/alarms** - List all alarms
- **POST /api/alarms/:alarmId/clear** - Clear alarm
- **GET /api/optimization/events** - List optimization events
- **POST /api/optimization/events** - Create optimization event
- **GET /api/approvals** - List approval requests
- **POST /api/approvals/:approvalId/approve** - Approve request
- **POST /api/approvals/:approvalId/reject** - Reject request
- **GET /api/interference/matrix** - Get interference matrix
- **GET /health** - Health check

## OpenTUI Terminal Interface

Command-line interface for operators:

### Commands

- **show_cells** - Display cell status table
- **show_alarms** - Display FM alarms
- **show_kpi_chart** - Display KPI trends
- **approve_pending** - Review and approve pending optimizations

### Example Output

```
=== CELL STATUS TABLE ===

CELL_ID       | PCI  | POWER  | TILT | STATUS     | RSRP    | SINR   | PRB%
--------------------------------------------------------------------------------
Cell-001      |  100 |   5.3  |  7.2 | active     |  -87.2  |  12.3  |   45
Cell-002      |  101 |   4.8  |  6.5 | degraded   |  -92.1  |   8.7  |   67
Cell-003      |  102 |   6.1  |  8.0 | active     |  -85.3  |  14.2  |   52
```

## Quick Start

### 1. Run the Demo

```bash
npm run ui:demo
```

This starts the full dashboard with mock data.

### 2. Programmatic Usage

```typescript
import { TitanDashboard, TitanAPIServer } from './src/ui/index.js';

// Initialize dashboard
const dashboard = new TitanDashboard({
  port: 8080,
  aguiPort: 3000,
  sseUpdateInterval: 10000,
  enableOpenTUI: true,
  enableAGUI: true
});

// Start dashboard
await dashboard.start();

// Update cell status
dashboard.updateCellStatus({
  cell_id: 'Cell-001',
  pci: 100,
  power_dbm: 5.0,
  tilt_deg: 7.0,
  // ... other fields
});

// Render visualizations
dashboard.renderInterferenceHeatmap(cells, matrix, -90);
dashboard.renderOptimizationTimeline(events);
dashboard.renderParameterControls(['Cell-001', 'Cell-002']);
dashboard.renderApprovalQueue();

// Start API server
const apiServer = new TitanAPIServer(dashboard, 8080);
await apiServer.start();
```

## Integration with TITAN Council

The dashboard integrates with the TITAN Council orchestrator:

```typescript
import { CouncilOrchestrator } from '../council/orchestrator.js';
import { TitanDashboard } from './titan-dashboard.js';

const dashboard = new TitanDashboard();
const council = new CouncilOrchestrator();

// Forward council events to dashboard
council.on('proposal_created', (proposal) => {
  dashboard.addOptimizationEvent({
    id: proposal.id,
    event_type: 'council_debate',
    reasoning: proposal.content,
    // ...
  });
});

council.on('consensus_reached', (decision) => {
  dashboard.createApprovalRequest({
    action: decision.action,
    riskLevel: assessRisk(decision),
    // ...
  });
});
```

## Configuration

### Dashboard Config

```typescript
interface TitanDashboardConfig {
  port?: number;              // API server port (default: 8080)
  aguiPort?: number;          // AG-UI server port (default: 3000)
  sseUpdateInterval?: number; // SSE update interval ms (default: 10000)
  enableOpenTUI?: boolean;    // Enable terminal UI (default: true)
  enableAGUI?: boolean;       // Enable AG-UI protocol (default: true)
}
```

### Parameter Definitions

```typescript
const parameters: ParameterDefinition[] = [
  {
    name: 'power_dbm',
    label: 'Transmit Power (dBm)',
    min: -130,
    max: 46,
    step: 0.1,
    unit: 'dBm',
    description: 'Cell transmit power (3GPP TS 38.104)'
  },
  {
    name: 'tilt_deg',
    label: 'Antenna Tilt (degrees)',
    min: 0,
    max: 15,
    step: 0.5,
    unit: '°',
    description: 'Vertical antenna tilt angle'
  },
  {
    name: 'azimuth_deg',
    label: 'Azimuth (degrees)',
    min: 0,
    max: 360,
    step: 1,
    unit: '°',
    description: 'Horizontal antenna azimuth'
  }
];
```

## Safety & Compliance

### 3GPP Bounds Checking

All parameter changes are validated against 3GPP specifications:

- **Power**: -130 to 46 dBm (TS 38.104)
- **Tilt**: 0 to 15 degrees (typical operator range)
- **Azimuth**: 0 to 360 degrees

### HITL Approval Workflow

1. **Risk Assessment**: Changes are classified as low/medium/high/critical risk
2. **Safety Checks**: Automated validation against constraints
3. **Human Approval**: Critical changes require operator signature
4. **Expiration**: Approvals expire after 1 hour
5. **Audit Trail**: All approvals are logged with signature

## Testing

```bash
# Run dashboard demo
npm run ui:demo

# Run with specific port
PORT=9000 npm run ui:demo

# Test SSE streams
curl http://localhost:8080/api/stream/kpi

# Test REST API
curl http://localhost:8080/api/cells
curl http://localhost:8080/health
```

## Performance

- **SSE Update Rate**: 10 seconds (configurable)
- **Max Timeline Events**: 1000 (auto-pruned)
- **Concurrent SSE Clients**: Unlimited (limited by system resources)
- **Dashboard Latency**: <50ms (typical)

## Browser Support

The React components require a modern browser with:

- ES2020 support
- Canvas API
- Server-Sent Events (SSE)
- Fetch API

Tested on:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## License

PROPRIETARY - Ericsson Autonomous Networks Division

## Version

7.0.0-alpha.1 (Neuro-Symbolic Titan)
