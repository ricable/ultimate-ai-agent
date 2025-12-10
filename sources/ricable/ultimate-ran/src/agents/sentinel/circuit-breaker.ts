/**
 * Circuit Breaker Implementation
 *
 * Implements the circuit breaker pattern with three states:
 * - CLOSED: Normal operations
 * - OPEN: All optimizations frozen
 * - HALF_OPEN: Testing recovery
 */

export enum CircuitState {
  CLOSED = 'CLOSED',
  OPEN = 'OPEN',
  HALF_OPEN = 'HALF_OPEN'
}

export interface CircuitBreakerConfig {
  failureThreshold: number;
  successThreshold: number;
  timeout: number;
}

export class CircuitBreaker {
  private state: CircuitState = CircuitState.CLOSED;
  private failureCount: number = 0;
  private successCount: number = 0;
  private lastFailureTime: number = 0;
  private config: CircuitBreakerConfig;

  constructor(config: CircuitBreakerConfig = {
    failureThreshold: 5,
    successThreshold: 2,
    timeout: 60000
  }) {
    this.config = config;
  }

  getState(): CircuitState {
    return this.state;
  }

  canProceed(): boolean {
    if (this.state === CircuitState.CLOSED) {
      return true;
    }

    if (this.state === CircuitState.OPEN) {
      if (this.shouldAttemptReset()) {
        this.halfOpen();
        return true;
      }
      return false;
    }

    // HALF_OPEN state - allow limited operations
    return true;
  }

  recordFailure(): void {
    this.failureCount++;
    this.lastFailureTime = Date.now();

    if (this.state === CircuitState.HALF_OPEN) {
      this.open();
      return;
    }

    if (this.failureCount >= this.config.failureThreshold) {
      this.open();
    }
  }

  recordSuccess(): void {
    if (this.state === CircuitState.HALF_OPEN) {
      this.successCount++;
      if (this.successCount >= this.config.successThreshold) {
        this.close();
      }
    }
  }

  recordAttempt(): void {
    // Track attempts in HALF_OPEN state
    // Used for monitoring recovery progress
  }

  open(): void {
    this.state = CircuitState.OPEN;
    this.lastFailureTime = Date.now();
  }

  halfOpen(): void {
    this.state = CircuitState.HALF_OPEN;
    this.successCount = 0;
  }

  close(): void {
    this.state = CircuitState.CLOSED;
    this.failureCount = 0;
    this.successCount = 0;
  }

  reset(): void {
    this.close();
  }

  shouldAttemptReset(): boolean {
    if (this.state !== CircuitState.OPEN) {
      return false;
    }

    const timeSinceLastFailure = Date.now() - this.lastFailureTime;
    return timeSinceLastFailure >= this.config.timeout;
  }
}
