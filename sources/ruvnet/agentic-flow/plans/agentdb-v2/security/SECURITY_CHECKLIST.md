# AgentDB v2 Security Checklist

## Overview

Security review checklist for RuVector integration ensuring no vulnerabilities are introduced.

## Security Categories

### 1. Dependency Security

#### npm Audit
- [ ] Run `npm audit` on all dependencies
- [ ] No critical vulnerabilities
- [ ] No high vulnerabilities
- [ ] Document and justify any accepted medium/low risks

#### Supply Chain
- [ ] Verify @ruvector/core package authenticity
- [ ] Check package maintainer reputation
- [ ] Review package.json for suspicious scripts
- [ ] Verify native binding sources

```bash
# Dependency audit commands
npm audit
npm audit --audit-level=moderate

# Check package info
npm view @ruvector/core
npm view @ruvector/gnn
```

#### Lockfile Integrity
- [ ] package-lock.json committed
- [ ] Integrity hashes verified
- [ ] No unexpected dependency changes

### 2. Native Code Security

#### Binary Verification
- [ ] Native bindings from trusted source
- [ ] WASM fallback available
- [ ] No unsigned binaries
- [ ] Checksum verification for releases

#### Memory Safety
- [ ] RuVector uses Rust (memory-safe by default)
- [ ] No buffer overflow risks in bindings
- [ ] Proper error handling for native calls
- [ ] Memory limits enforced

```typescript
// Safe native call wrapper
async function safeNativeCall<T>(fn: () => T): Promise<T> {
  try {
    return fn();
  } catch (error) {
    if (error instanceof Error && error.message.includes('memory')) {
      throw new SecurityError('Memory limit exceeded');
    }
    throw error;
  }
}
```

### 3. Input Validation

#### Vector Input
- [ ] Dimension validation (matches config)
- [ ] NaN/Infinity detection
- [ ] Maximum vector size enforced
- [ ] ID sanitization

```typescript
// packages/agentdb/src/security/validation.ts

export function validateVector(
  embedding: Float32Array,
  expectedDim: number
): void {
  if (embedding.length !== expectedDim) {
    throw new ValidationError(
      `Invalid dimension: expected ${expectedDim}, got ${embedding.length}`
    );
  }

  for (let i = 0; i < embedding.length; i++) {
    if (!Number.isFinite(embedding[i])) {
      throw new ValidationError(`Invalid value at index ${i}: ${embedding[i]}`);
    }
  }
}

export function validateId(id: string): void {
  if (typeof id !== 'string' || id.length === 0 || id.length > 256) {
    throw new ValidationError('Invalid ID: must be non-empty string <= 256 chars');
  }

  // Prevent path traversal in IDs used for file operations
  if (id.includes('..') || id.includes('/') || id.includes('\\')) {
    throw new ValidationError('Invalid ID: contains path characters');
  }
}
```

#### Query Input
- [ ] K-value bounds (1 <= k <= maxK)
- [ ] Threshold bounds (0 <= threshold <= 1)
- [ ] efSearch bounds
- [ ] Filter sanitization

```typescript
export function validateSearchOptions(options: SearchOptions): void {
  if (options.k !== undefined) {
    if (!Number.isInteger(options.k) || options.k < 1 || options.k > 10000) {
      throw new ValidationError('k must be integer between 1 and 10000');
    }
  }

  if (options.threshold !== undefined) {
    if (typeof options.threshold !== 'number' ||
        options.threshold < 0 || options.threshold > 1) {
      throw new ValidationError('threshold must be number between 0 and 1');
    }
  }
}
```

### 4. Path Security

#### File Operations
- [ ] Path traversal prevention
- [ ] Symlink handling
- [ ] Permissions verification
- [ ] Temp file cleanup

```typescript
import * as path from 'path';
import * as fs from 'fs';

export function validatePath(filePath: string, baseDir: string): string {
  const resolved = path.resolve(baseDir, filePath);
  const relative = path.relative(baseDir, resolved);

  // Ensure path doesn't escape base directory
  if (relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new SecurityError('Path traversal attempt detected');
  }

  return resolved;
}

export async function secureWrite(
  filePath: string,
  data: Buffer,
  baseDir: string
): Promise<void> {
  const safePath = validatePath(filePath, baseDir);

  // Check if path exists and is a symlink
  try {
    const stats = await fs.promises.lstat(safePath);
    if (stats.isSymbolicLink()) {
      throw new SecurityError('Cannot write to symbolic link');
    }
  } catch (error) {
    // File doesn't exist, which is fine
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
      throw error;
    }
  }

  await fs.promises.writeFile(safePath, data);
}
```

### 5. Denial of Service Prevention

#### Resource Limits
- [ ] Maximum vectors per database
- [ ] Maximum memory usage
- [ ] Query timeout
- [ ] Batch size limits

```typescript
export const SECURITY_LIMITS = {
  MAX_VECTORS: 10_000_000,        // 10M vectors max
  MAX_DIMENSION: 4096,            // Dimension limit
  MAX_BATCH_SIZE: 10_000,         // Batch insert limit
  MAX_K: 10_000,                  // Search result limit
  QUERY_TIMEOUT_MS: 30_000,       // 30s query timeout
  MAX_MEMORY_MB: 16_384,          // 16GB memory limit
  MAX_ID_LENGTH: 256,             // ID string length
  MAX_METADATA_SIZE: 65_536       // 64KB metadata per vector
};

export function enforceLimit(
  value: number,
  limit: number,
  name: string
): void {
  if (value > limit) {
    throw new SecurityError(`${name} exceeds limit: ${value} > ${limit}`);
  }
}
```

#### Rate Limiting
- [ ] Insert rate limiting
- [ ] Search rate limiting
- [ ] API endpoint protection

### 6. Data Protection

#### Sensitive Data
- [ ] No secrets in vector metadata
- [ ] No PII logging
- [ ] Secure deletion support
- [ ] Encryption at rest (optional)

```typescript
// Metadata sanitization
export function sanitizeMetadata(
  metadata: Record<string, any>
): Record<string, any> {
  const sanitized = { ...metadata };

  // Remove potential sensitive fields
  const sensitiveKeys = ['password', 'secret', 'token', 'key', 'credential'];
  for (const key of Object.keys(sanitized)) {
    if (sensitiveKeys.some(s => key.toLowerCase().includes(s))) {
      delete sanitized[key];
      console.warn(`Removed sensitive metadata field: ${key}`);
    }
  }

  return sanitized;
}
```

#### Logging Security
- [ ] No vector data in logs
- [ ] No IDs in debug logs (unless debug mode)
- [ ] Log rotation configured
- [ ] Audit trail for mutations

```typescript
export function safeLog(message: string, data?: any): void {
  if (data && typeof data === 'object') {
    // Remove potentially sensitive fields
    const safe = { ...data };
    delete safe.embedding;
    delete safe.vector;
    delete safe.metadata;
    console.log(message, safe);
  } else {
    console.log(message);
  }
}
```

### 7. Error Handling

#### Information Disclosure
- [ ] No stack traces to users
- [ ] Generic error messages externally
- [ ] Detailed errors in logs only
- [ ] No internal paths exposed

```typescript
export class PublicError extends Error {
  constructor(
    public readonly publicMessage: string,
    public readonly internalMessage: string,
    public readonly code: string
  ) {
    super(publicMessage);
  }
}

export function handleError(error: unknown): { message: string; code: string } {
  if (error instanceof PublicError) {
    console.error(`[${error.code}] ${error.internalMessage}`);
    return { message: error.publicMessage, code: error.code };
  }

  // Log full error internally
  console.error('Unexpected error:', error);

  // Return generic message externally
  return {
    message: 'An internal error occurred',
    code: 'INTERNAL_ERROR'
  };
}
```

### 8. Graph Query Security (RuVector Graph)

#### Cypher Injection Prevention
- [ ] Parameterized queries only
- [ ] No string concatenation in queries
- [ ] Query complexity limits
- [ ] Result size limits

```typescript
// UNSAFE - Never do this
const UNSAFE = `MATCH (n {name: '${userInput}'}) RETURN n`;

// SAFE - Use parameters
const SAFE = {
  query: 'MATCH (n {name: $name}) RETURN n',
  params: { name: userInput }
};

export function executeCypher(
  cypher: string,
  params: Record<string, any>
): Promise<any> {
  // Validate no string interpolation
  if (cypher.includes('${') || cypher.includes("'+")) {
    throw new SecurityError('Potential Cypher injection detected');
  }

  // Validate params
  for (const [key, value] of Object.entries(params)) {
    if (typeof value === 'string' && value.length > 10000) {
      throw new SecurityError('Parameter value too long');
    }
  }

  return graphDb.execute(cypher, params);
}
```

## Security Scanning

### Automated Scans

```yaml
# Run in CI
- npm audit --audit-level=high
- npx snyk test
- npx retire
- trivy fs --security-checks vuln .
```

### Manual Review

- [ ] Code review for security patterns
- [ ] Dependency review
- [ ] Native code audit (if modified)
- [ ] Penetration testing (pre-release)

## Security Testing

```typescript
// packages/agentdb/tests/security/injection.test.ts

describe('Security: Input Validation', () => {
  it('should reject path traversal in IDs', () => {
    expect(() => validateId('../../../etc/passwd')).toThrow(ValidationError);
    expect(() => validateId('..\\..\\windows\\system32')).toThrow(ValidationError);
  });

  it('should reject oversized vectors', () => {
    const huge = new Float32Array(100000);
    expect(() => validateVector(huge, 384)).toThrow(ValidationError);
  });

  it('should reject NaN in vectors', () => {
    const invalid = new Float32Array([1.0, NaN, 0.5]);
    expect(() => validateVector(invalid, 3)).toThrow(ValidationError);
  });

  it('should sanitize sensitive metadata', () => {
    const result = sanitizeMetadata({
      name: 'test',
      password: 'secret123',
      apiKey: 'abc'
    });
    expect(result.name).toBe('test');
    expect(result.password).toBeUndefined();
    expect(result.apiKey).toBeUndefined();
  });
});
```

## Incident Response

### Security Issue Reporting
- Email: security@ruv.io
- Responsible disclosure policy
- 90-day fix timeline

### Patch Process
1. Assess severity (CVSS score)
2. Develop fix in private branch
3. Security advisory draft
4. Coordinated release
5. Post-mortem

## Compliance

- [ ] OWASP Top 10 review
- [ ] CWE/SANS Top 25 check
- [ ] GDPR considerations (if PII in vectors)
- [ ] SOC 2 alignment (enterprise)

## Sign-off

| Reviewer | Role | Date | Status |
|----------|------|------|--------|
| | Security Lead | | Pending |
| | Backend Lead | | Pending |
| | DevOps | | Pending |
