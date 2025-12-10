# SECURITY FIXES REQUIRED - AgentDB MCP Server

**Priority:** 🔴 CRITICAL
**Status:** Action Required Before Production
**Timeline:** 1-2 Weeks

---

## Quick Summary

The AgentDB MCP server security audit identified **3 critical SQL injection vulnerabilities**, **1 code execution risk**, and **15+ missing input validations** that must be fixed before production deployment.

**Overall Security Rating: ⚠️ MODERATE RISK**

---

## CRITICAL Fixes (Must Fix This Week)

### 1. SQL Injection in agentdb_delete Tool 🔴

**File:** `packages/agentdb/src/mcp/agentdb-mcp-server.ts:877-914`

**Current Code (VULNERABLE):**
```typescript
case 'agentdb_delete': {
  let deleted = 0;
  const id = args?.id as number | undefined;
  const filters = args?.filters as any;

  if (id !== undefined) {
    const stmt = db.prepare('DELETE FROM episodes WHERE id = ?');
    const result = stmt.run(id);
    deleted = result.changes;
  } else if (filters) {
    // ⚠️ VULNERABLE: filters passed to bulkDelete without validation
    deleted = batchOps.bulkDelete('episodes', conditions);
  }
}
```

**Attack Example:**
```javascript
// Malicious MCP request
{
  "name": "agentdb_delete",
  "arguments": {
    "filters": {
      "session_id": "'; DROP TABLE episodes; --"
    }
  }
}
```

**Fix:**
```typescript
case 'agentdb_delete': {
  let deleted = 0;
  const id = args?.id as number | undefined;
  const filters = args?.filters as any;

  if (id !== undefined) {
    // Validate ID is a positive integer
    if (!Number.isInteger(id) || id <= 0) {
      throw new Error('Invalid ID: must be a positive integer');
    }

    const stmt = db.prepare('DELETE FROM episodes WHERE id = ?');
    const result = stmt.run(id);
    deleted = result.changes;
  } else if (filters) {
    // Build parameterized WHERE clause
    const whereConditions: string[] = [];
    const params: any[] = [];

    if (filters.session_id) {
      if (typeof filters.session_id !== 'string') {
        throw new Error('Invalid session_id: must be a string');
      }
      whereConditions.push('session_id = ?');
      params.push(filters.session_id);
    }

    if (filters.before_timestamp) {
      if (!Number.isInteger(filters.before_timestamp)) {
        throw new Error('Invalid before_timestamp: must be an integer');
      }
      whereConditions.push('ts < ?');
      params.push(filters.before_timestamp);
    }

    if (whereConditions.length === 0) {
      throw new Error('No valid filters provided');
    }

    const sql = `DELETE FROM episodes WHERE ${whereConditions.join(' AND ')}`;
    const stmt = db.prepare(sql);
    const result = stmt.run(...params);
    deleted = result.changes;
  } else {
    throw new Error('Either id or filters must be provided');
  }

  return {
    content: [
      {
        type: 'text',
        text: `✅ Delete operation completed!\n📊 Deleted: ${deleted} vector(s)`,
      },
    ],
  };
}
```

---

### 2. PRAGMA Injection Vulnerability 🔴

**File:** `packages/agentdb/src/db-fallback.ts:154-157`

**Current Code (VULNERABLE):**
```typescript
pragma(pragma: string, options?: any) {
  const result = this.db.exec(`PRAGMA ${pragma}`);
  return result[0]?.values[0]?.[0];
}
```

**Attack Example:**
```javascript
db.pragma("journal_mode = DELETE; DROP TABLE episodes; --")
```

**Fix:**
```typescript
pragma(pragma: string, options?: any) {
  // Whitelist of allowed PRAGMA commands
  const allowedPragmas: Record<string, RegExp> = {
    'journal_mode': /^journal_mode\s*=\s*(DELETE|WAL|TRUNCATE|PERSIST|MEMORY|OFF)$/i,
    'synchronous': /^synchronous\s*=\s*(OFF|NORMAL|FULL|EXTRA)$/i,
    'cache_size': /^cache_size\s*=\s*-?\d+$/,
    'foreign_keys': /^foreign_keys\s*=\s*(ON|OFF)$/i,
    'page_count': /^page_count$/i,
    'page_size': /^page_size$/i,
  };

  // Extract PRAGMA name (first word)
  const pragmaName = pragma.trim().split(/[\s=]/)[0].toLowerCase();

  // Check if PRAGMA is allowed
  const allowedPattern = allowedPragmas[pragmaName];
  if (!allowedPattern) {
    throw new Error(`PRAGMA ${pragmaName} is not allowed`);
  }

  // Validate full PRAGMA syntax
  if (!allowedPattern.test(pragma.trim())) {
    throw new Error(`Invalid PRAGMA syntax: ${pragma}`);
  }

  // Safe to execute
  const result = this.db.exec(`PRAGMA ${pragma}`);
  return result[0]?.values[0]?.[0];
}
```

---

### 3. Remove eval() Code Execution Risk 🔴

**File:** `packages/agentdb/src/db-fallback.ts:59, 143`

**Current Code (DANGEROUS):**
```typescript
const fs = eval('require')('fs');
```

**Fix:**
```typescript
// Option 1: Direct require (preferred for Node.js)
import * as fs from 'fs';

// Option 2: Conditional require without eval
let fs: any = null;
try {
  // Check if we're in Node.js environment
  if (typeof process !== 'undefined' && process.versions?.node) {
    fs = require('fs');
  }
} catch (error) {
  console.warn('Filesystem not available (browser environment)');
}

// Then use fs conditionally
if (filename !== ':memory:') {
  if (!fs) {
    throw new Error('Filesystem operations not supported in this environment');
  }

  try {
    if (fs.existsSync(filename)) {
      const buffer = fs.readFileSync(filename);
      this.db = new SQL.Database(buffer);
    } else {
      this.db = new SQL.Database();
    }
  } catch (error) {
    this.db = new SQL.Database();
  }
}
```

---

## HIGH Priority Fixes (This Week)

### 4. Add Input Validation for All Tools 🟠

**Create Validation Utilities:**

```typescript
// packages/agentdb/src/utils/validation.ts

export class InputValidator {
  private static MAX_TEXT_LENGTH = 100000; // 100KB
  private static MAX_BATCH_SIZE = 1000;
  private static MAX_METADATA_SIZE = 10000; // 10KB
  private static MAX_ARRAY_LENGTH = 1000;

  static validateText(text: string, maxLength: number = this.MAX_TEXT_LENGTH): void {
    if (typeof text !== 'string') {
      throw new Error('Text must be a string');
    }
    if (text.length === 0) {
      throw new Error('Text cannot be empty');
    }
    if (text.length > maxLength) {
      throw new Error(`Text exceeds maximum length of ${maxLength} characters`);
    }
  }

  static validateNumber(
    value: number,
    min: number,
    max: number,
    paramName: string
  ): void {
    if (typeof value !== 'number' || isNaN(value) || !isFinite(value)) {
      throw new Error(`${paramName} must be a valid number`);
    }
    if (value < min || value > max) {
      throw new Error(`${paramName} must be between ${min} and ${max}`);
    }
  }

  static validateArray<T>(
    arr: T[],
    maxLength: number = this.MAX_ARRAY_LENGTH,
    paramName: string = 'array'
  ): void {
    if (!Array.isArray(arr)) {
      throw new Error(`${paramName} must be an array`);
    }
    if (arr.length > maxLength) {
      throw new Error(`${paramName} exceeds maximum length of ${maxLength}`);
    }
  }

  static validateMetadata(metadata: any): void {
    if (metadata === null || metadata === undefined) {
      return; // Optional
    }

    if (typeof metadata !== 'object' || Array.isArray(metadata)) {
      throw new Error('Metadata must be an object');
    }

    const metadataStr = JSON.stringify(metadata);
    if (metadataStr.length > this.MAX_METADATA_SIZE) {
      throw new Error(`Metadata exceeds maximum size of ${this.MAX_METADATA_SIZE} bytes`);
    }
  }

  static validateSessionId(sessionId: string): void {
    if (typeof sessionId !== 'string') {
      throw new Error('Session ID must be a string');
    }
    if (sessionId.length === 0 || sessionId.length > 256) {
      throw new Error('Session ID must be between 1 and 256 characters');
    }
    // Allow alphanumeric, hyphens, underscores
    if (!/^[a-zA-Z0-9_-]+$/.test(sessionId)) {
      throw new Error('Session ID contains invalid characters');
    }
  }

  static validateReward(reward: number): void {
    this.validateNumber(reward, 0, 1, 'reward');
  }

  static validateSimilarity(similarity: number): void {
    this.validateNumber(similarity, 0, 1, 'similarity');
  }

  static validateK(k: number): void {
    this.validateNumber(k, 1, 1000, 'k');
  }

  static validateEpochs(epochs: number): void {
    this.validateNumber(epochs, 1, 10000, 'epochs');
  }

  static validateBatchSize(batchSize: number): void {
    this.validateNumber(batchSize, 1, this.MAX_BATCH_SIZE, 'batchSize');
  }
}
```

**Apply Validation to Tools:**

```typescript
// Import validator
import { InputValidator } from '../utils/validation.js';

// Example: agentdb_insert
case 'agentdb_insert': {
  const text = args?.text as string;
  const sessionId = (args?.session_id as string) || 'default';
  const tags = (args?.tags as string[]) || [];
  const metadata = (args?.metadata as Record<string, any>) || {};

  // VALIDATE ALL INPUTS
  InputValidator.validateText(text);
  InputValidator.validateSessionId(sessionId);
  InputValidator.validateArray(tags, 100, 'tags');
  InputValidator.validateMetadata(metadata);

  // Validate each tag
  for (const tag of tags) {
    InputValidator.validateText(tag, 100);
  }

  // ... proceed with insert
}

// Example: learning_train
case 'learning_train': {
  const sessionId = args?.session_id as string;
  const epochs = (args?.epochs as number) || 50;
  const batchSize = (args?.batch_size as number) || 32;
  const learningRate = (args?.learning_rate as number) || 0.01;

  // VALIDATE ALL INPUTS
  InputValidator.validateSessionId(sessionId);
  InputValidator.validateEpochs(epochs);
  InputValidator.validateBatchSize(batchSize);
  InputValidator.validateNumber(learningRate, 0.00001, 1, 'learningRate');

  // ... proceed with training
}
```

---

### 5. Implement Rate Limiting 🟠

**Create Rate Limiter:**

```typescript
// packages/agentdb/src/utils/rate-limiter.ts

export class RateLimiter {
  private requests: Map<string, number[]> = new Map();
  private maxRequestsPerWindow: number;
  private windowMs: number;

  constructor(maxRequestsPerWindow: number = 100, windowMs: number = 60000) {
    this.maxRequestsPerWindow = maxRequestsPerWindow;
    this.windowMs = windowMs;

    // Cleanup old entries every 5 minutes
    setInterval(() => this.cleanup(), 5 * 60 * 1000);
  }

  checkLimit(clientId: string): boolean {
    const now = Date.now();
    const timestamps = this.requests.get(clientId) || [];

    // Remove timestamps outside current window
    const validTimestamps = timestamps.filter(ts => now - ts < this.windowMs);

    if (validTimestamps.length >= this.maxRequestsPerWindow) {
      return false; // Rate limit exceeded
    }

    validTimestamps.push(now);
    this.requests.set(clientId, validTimestamps);
    return true;
  }

  private cleanup(): void {
    const now = Date.now();
    for (const [clientId, timestamps] of this.requests.entries()) {
      const validTimestamps = timestamps.filter(ts => now - ts < this.windowMs);
      if (validTimestamps.length === 0) {
        this.requests.delete(clientId);
      } else {
        this.requests.set(clientId, validTimestamps);
      }
    }
  }

  reset(clientId: string): void {
    this.requests.delete(clientId);
  }
}
```

**Apply to MCP Server:**

```typescript
// Initialize rate limiter
const rateLimiter = new RateLimiter(
  parseInt(process.env.RATE_LIMIT_PER_MINUTE || '100'),
  60000 // 1 minute window
);

// Add to request handler
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  // Extract client identifier (use session ID or IP if available)
  const clientId = (args?.session_id as string) || 'anonymous';

  // Check rate limit
  if (!rateLimiter.checkLimit(clientId)) {
    return {
      content: [{
        type: 'text',
        text: '⚠️ Rate limit exceeded. Please try again in a few moments.',
      }],
      isError: true,
    };
  }

  // ... proceed with request handling
});
```

---

### 6. Improve Error Handling 🟠

**Create Secure Error Handler:**

```typescript
// packages/agentdb/src/utils/error-handler.ts

const isProduction = process.env.NODE_ENV === 'production';

export function generateErrorCode(error: Error): string {
  // Create deterministic but non-revealing error code
  const hash = error.message
    .split('')
    .reduce((acc, char) => ((acc << 5) - acc) + char.charCodeAt(0), 0);

  return `ERR-${Math.abs(hash).toString(16).toUpperCase().substring(0, 8)}`;
}

export function formatMCPError(error: any, toolName: string, args: any): any {
  // Log detailed error server-side (sanitize sensitive data)
  const sanitizedArgs = { ...args };
  delete sanitizedArgs.password;
  delete sanitizedArgs.api_key;

  console.error('MCP Tool Error:', {
    tool: toolName,
    error: error.message,
    code: generateErrorCode(error),
    timestamp: new Date().toISOString(),
    // Don't log full stack in production
    ...(isProduction ? {} : { stack: error.stack, args: sanitizedArgs })
  });

  // Return appropriate error to client
  return {
    content: [{
      type: 'text',
      text: isProduction
        ? `❌ Operation failed. Error code: ${generateErrorCode(error)}\nPlease contact support with this code.`
        : `❌ Error: ${error.message}\n${error.stack || ''}`,
    }],
    isError: true,
  };
}
```

**Apply to Handler:**

```typescript
import { formatMCPError } from '../utils/error-handler.js';

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    // ... handle request
  } catch (error: any) {
    return formatMCPError(error, name, args);
  }
});
```

---

## MEDIUM Priority (Week 2)

### 7. Add Graceful Shutdown 🟡

```typescript
// packages/agentdb/src/mcp/agentdb-mcp-server.ts

let isShuttingDown = false;

async function gracefulShutdown(signal: string) {
  if (isShuttingDown) return;
  isShuttingDown = true;

  console.error(`\n🛑 Received ${signal}, shutting down gracefully...`);

  try {
    // Close database connection
    console.error('  Closing database connection...');
    db.close();

    // Close MCP server
    console.error('  Closing MCP server...');
    await server.close();

    console.error('✅ Shutdown complete');
    process.exit(0);
  } catch (error) {
    console.error('❌ Error during shutdown:', error);
    process.exit(1);
  }
}

// Register shutdown handlers
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

// Unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Promise Rejection:', reason);
  // Log but don't crash - MCP server should stay running
});

// Uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  gracefulShutdown('UNCAUGHT_EXCEPTION');
});

// Start server
main().catch((error) => {
  console.error('Fatal error starting server:', error);
  process.exit(1);
});
```

---

### 8. Safe JSON Parsing 🟡

```typescript
// packages/agentdb/src/utils/safe-json.ts

export function safeJSONParse<T>(json: string | null | undefined, defaultValue: T): T {
  if (json === null || json === undefined || json === '') {
    return defaultValue;
  }

  try {
    return JSON.parse(json);
  } catch (error) {
    console.error('JSON parse error:', error, 'Input:', json?.substring(0, 100));
    return defaultValue;
  }
}

export function safeJSONStringify(value: any, defaultValue: string = '{}'): string {
  try {
    return JSON.stringify(value);
  } catch (error) {
    console.error('JSON stringify error:', error);
    return defaultValue;
  }
}
```

**Apply Throughout Codebase:**

```typescript
// Replace all instances of JSON.parse() with safeJSONParse()
import { safeJSONParse } from '../utils/safe-json.js';

// Before
config: JSON.parse(row.config),

// After
config: safeJSONParse(row.config, {}),
```

---

## Testing Requirements

### Security Test Suite

```typescript
// packages/agentdb/tests/security.test.ts

import { describe, test, expect } from 'vitest';

describe('Security Tests', () => {
  describe('SQL Injection Protection', () => {
    test('should reject malicious session_id in delete', async () => {
      const maliciousInput = {
        filters: {
          session_id: "'; DROP TABLE episodes; --"
        }
      };

      await expect(
        server.callTool('agentdb_delete', maliciousInput)
      ).rejects.toThrow();
    });

    test('should sanitize LIKE patterns in tag search', async () => {
      const maliciousTag = "%'; DROP TABLE reasoning_patterns; --";

      const result = await server.callTool('agentdb_pattern_search', {
        task: 'test',
        filters: { tags: [maliciousTag] }
      });

      // Should not throw or execute malicious code
      expect(result).toBeDefined();
    });
  });

  describe('Input Validation', () => {
    test('should reject oversized batch insert', async () => {
      const oversizedBatch = {
        items: Array(100000).fill({ text: 'test' })
      };

      await expect(
        server.callTool('agentdb_insert_batch', oversizedBatch)
      ).rejects.toThrow('exceeds maximum');
    });

    test('should reject invalid reward values', async () => {
      await expect(
        server.callTool('learning_feedback', {
          session_id: 'test',
          state: 'test',
          action: 'test',
          reward: -100, // Invalid
          success: true
        })
      ).rejects.toThrow('reward must be between 0 and 1');
    });

    test('should reject NaN and Infinity values', async () => {
      await expect(
        server.callTool('learning_train', {
          session_id: 'test',
          epochs: NaN
        })
      ).rejects.toThrow('must be a valid number');
    });
  });

  describe('Rate Limiting', () => {
    test('should enforce rate limits', async () => {
      const requests = Array(101).fill(null).map(() =>
        server.callTool('db_stats', {})
      );

      const results = await Promise.allSettled(requests);
      const rejected = results.filter(r => r.status === 'rejected');

      expect(rejected.length).toBeGreaterThan(0);
    });
  });

  describe('Error Handling', () => {
    test('should not leak stack traces in production', async () => {
      process.env.NODE_ENV = 'production';

      try {
        await server.callTool('invalid_tool', {});
      } catch (error: any) {
        expect(error.message).not.toContain('packages/agentdb');
        expect(error.message).toMatch(/ERR-[A-F0-9]{8}/);
      }
    });
  });
});
```

---

## Verification Checklist

Before considering fixes complete:

- [ ] All 3 critical SQL injection vulnerabilities fixed
- [ ] eval() removed from db-fallback.ts
- [ ] Input validation added to all 29 MCP tools
- [ ] Rate limiting implemented and tested
- [ ] Error handling improved with production mode
- [ ] Graceful shutdown handlers added
- [ ] Safe JSON parsing applied throughout
- [ ] Security test suite created and passing
- [ ] Manual penetration testing completed
- [ ] Code review by security team
- [ ] Documentation updated

---

## Timeline

**Week 1 (Days 1-3):**
- Fix critical SQL injection vulnerabilities
- Remove eval() usage
- Add input validation framework

**Week 1 (Days 4-5):**
- Implement rate limiting
- Improve error handling
- Create security tests

**Week 2 (Days 1-2):**
- Add graceful shutdown
- Safe JSON parsing
- Code review

**Week 2 (Days 3-5):**
- Testing and validation
- Documentation
- Deployment preparation

---

## Support

For questions about these fixes:
- **GitHub Issues:** https://github.com/ruvnet/agentic-flow/issues
- **Security Contact:** [security@example.com]
- **Documentation:** /docs/AGENTDB-MCP-SECURITY-AUDIT.md

---

**Document Version:** 1.0
**Last Updated:** 2025-10-25
**Next Review:** After fixes implementation
