# MCP Connection Recovery Design

## Overview
This document outlines the comprehensive auto-recovery mechanisms for MCP connection stability (Issue #65).

## Current Architecture Analysis

### Existing Components
1. **MCPClient** - Basic connection state management
   - Simple `connected` boolean flag
   - No retry logic or recovery mechanisms
   - Throws errors on connection failures

2. **StdioTransport** - Basic I/O handling
   - No reconnection capabilities
   - Stdin close detection but no recovery
   - No heartbeat or health monitoring

3. **LifecycleManager** - Server-side lifecycle
   - Has health check infrastructure
   - Basic auto-restart capability
   - No client-side recovery coordination

### Identified Gaps
- No connection-level heartbeat mechanism
- No automatic reconnection on failure
- No graceful degradation to CLI
- No state preservation during disconnects
- No exponential backoff for retries
- No connection event notifications

## Proposed Recovery Architecture

### 1. Connection Health Monitor
A dedicated component for continuous connection health monitoring.

```typescript
interface ConnectionHealthMonitor {
  // Configuration
  heartbeatInterval: number;       // Default: 5000ms
  heartbeatTimeout: number;        // Default: 10000ms
  maxMissedHeartbeats: number;     // Default: 3
  
  // Methods
  start(): Promise<void>;
  stop(): Promise<void>;
  checkHealth(): Promise<HealthStatus>;
  onHealthChange(handler: (status: HealthStatus) => void): void;
}

interface HealthStatus {
  healthy: boolean;
  lastHeartbeat: Date;
  missedHeartbeats: number;
  latency: number;
  error?: string;
}
```

### 2. Heartbeat Mechanism
Lightweight ping/pong protocol for connection liveness.

```typescript
interface HeartbeatProtocol {
  // Client sends ping
  ping(): Promise<void>;
  
  // Server responds with pong
  pong(pingId: string): Promise<void>;
  
  // Track round-trip time
  measureLatency(): number;
}
```

Implementation approach:
- Use MCP notification channel for heartbeats
- Include timestamp in ping for latency measurement
- Track missed heartbeats for failure detection
- Exponentially increase interval on failures

### 3. Automatic Reconnection Logic
Smart reconnection with exponential backoff.

```typescript
interface ReconnectionManager {
  // Configuration
  maxRetries: number;              // Default: 10
  initialDelay: number;            // Default: 1000ms
  maxDelay: number;                // Default: 30000ms
  backoffMultiplier: number;       // Default: 2
  
  // Methods
  attemptReconnection(): Promise<boolean>;
  reset(): void;
  onReconnect(handler: () => void): void;
  onFailure(handler: (error: Error) => void): void;
}
```

Reconnection strategy:
1. Detect connection failure via heartbeat timeout
2. Start exponential backoff retry sequence
3. Preserve pending requests during reconnection
4. Restore state after successful reconnection
5. Emit events for UI feedback

### 4. Graceful CLI Fallback
Seamless transition when MCP connection fails.

```typescript
interface FallbackCoordinator {
  // Detect MCP availability
  isMCPAvailable(): boolean;
  
  // Switch to CLI mode
  enableCLIFallback(): void;
  
  // Queue operations for when MCP returns
  queueOperation(op: Operation): void;
  
  // Replay queued operations
  replayQueue(): Promise<void>;
}
```

Fallback behavior:
- Detect MCP connection loss
- Switch to direct CLI command execution
- Queue non-critical MCP operations
- Notify user of degraded mode
- Attempt background reconnection
- Restore full functionality when connected

### 5. Connection State Persistence
Maintain state across disconnections.

```typescript
interface ConnectionStateManager {
  // Save current state
  saveState(state: ConnectionState): void;
  
  // Restore previous state
  restoreState(): ConnectionState | null;
  
  // Track connection history
  recordEvent(event: ConnectionEvent): void;
  
  // Get connection metrics
  getMetrics(): ConnectionMetrics;
}

interface ConnectionState {
  sessionId: string;
  lastConnected: Date;
  pendingRequests: MCPRequest[];
  configuration: MCPConfig;
}
```

### 6. Resource Cleanup
Proper cleanup on disconnect to prevent resource leaks.

```typescript
interface ResourceCleanup {
  // Register resources for cleanup
  register(resource: CleanupResource): void;
  
  // Cleanup on disconnect
  cleanup(): Promise<void>;
  
  // Cleanup specific resource
  cleanupResource(id: string): Promise<void>;
}
```

## Implementation Plan

### Phase 1: Core Infrastructure
1. Create `ConnectionHealthMonitor` class
2. Implement heartbeat protocol in transports
3. Add connection events to `MCPClient`
4. Create `ReconnectionManager` class

### Phase 2: Recovery Logic
1. Implement exponential backoff algorithm
2. Add state preservation mechanisms
3. Create fallback coordinator
4. Integrate with existing lifecycle manager

### Phase 3: Integration
1. Update `StdioTransport` with recovery
2. Update `HttpTransport` with recovery
3. Add recovery to `MCPClient`
4. Create recovery configuration options

### Phase 4: Testing & Monitoring
1. Create connection failure simulations
2. Test recovery under various scenarios
3. Add recovery metrics and logging
4. Create recovery dashboard

## Configuration

Default recovery configuration:
```typescript
const defaultRecoveryConfig = {
  // Health monitoring
  enableHealthMonitoring: true,
  heartbeatInterval: 5000,
  heartbeatTimeout: 10000,
  maxMissedHeartbeats: 3,
  
  // Reconnection
  enableAutoReconnect: true,
  maxReconnectAttempts: 10,
  reconnectInitialDelay: 1000,
  reconnectMaxDelay: 30000,
  reconnectBackoffMultiplier: 2,
  
  // Fallback
  enableCLIFallback: true,
  fallbackQueueSize: 100,
  
  // State persistence
  enableStatePersistence: true,
  stateStorageLocation: '.mcp-state',
  
  // Resource cleanup
  cleanupTimeout: 5000,
};
```

## Integration Points

### 1. Transport Layer
- Add heartbeat support to all transports
- Implement reconnection logic
- Add connection event emitters

### 2. Client Layer
- Integrate health monitor
- Add recovery manager
- Implement request queuing

### 3. Server Layer
- Add heartbeat endpoint
- Support stateful reconnection
- Implement session recovery

### 4. CLI Integration
- Add fallback detection
- Implement command queuing
- Show connection status

## Error Handling

### Connection Errors
1. **Timeout**: Trigger reconnection after heartbeat timeout
2. **Network Error**: Immediate reconnection attempt
3. **Protocol Error**: Log and attempt recovery
4. **Authentication Error**: Prompt for re-authentication

### Recovery Errors
1. **Max Retries Exceeded**: Switch to CLI fallback
2. **State Corruption**: Clear state and fresh start
3. **Resource Leak**: Force cleanup and restart

## Monitoring & Metrics

### Key Metrics
- Connection uptime percentage
- Average reconnection time
- Heartbeat latency (p50, p95, p99)
- Failed connection attempts
- Fallback activation frequency
- State recovery success rate

### Logging
- Connection state changes
- Heartbeat failures
- Reconnection attempts
- Fallback activations
- Resource cleanup events

## Testing Strategy

### Unit Tests
- Health monitor logic
- Reconnection algorithm
- State persistence
- Resource cleanup

### Integration Tests
- Full recovery flow
- Fallback scenarios
- Multi-transport recovery
- Load testing

### Failure Scenarios
1. Network disconnection
2. Server crash
3. Timeout conditions
4. Partial failures
5. Rapid connect/disconnect

## Future Enhancements

1. **Circuit Breaker Pattern**: Prevent cascading failures
2. **Connection Pooling**: Multiple connection redundancy
3. **Geographic Failover**: Multi-region support
4. **Predictive Health**: ML-based failure prediction
5. **Custom Recovery Strategies**: Plugin architecture