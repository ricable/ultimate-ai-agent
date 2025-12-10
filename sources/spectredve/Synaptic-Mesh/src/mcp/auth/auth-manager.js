/**
 * Authentication and Authorization Manager for MCP
 * Handles security, rate limiting, and access control
 */

import crypto from 'crypto';
import { EventEmitter } from 'events';

export class AuthManager extends EventEmitter {
  constructor(config) {
    super();
    this.config = config;
    this.apiKeys = new Map();
    this.sessions = new Map();
    this.rateLimiters = new Map();
    this.blacklist = new Set();
    
    this.defaultRateLimits = {
      requests: 100,
      window: 60000, // 1 minute
      burst: 10
    };

    this.initializeAuth();
  }

  initializeAuth() {
    // Generate default API key if none configured
    if (!this.config.apiKeys || this.config.apiKeys.length === 0) {
      const defaultKey = this.generateApiKey();
      this.apiKeys.set(defaultKey, {
        name: 'default',
        permissions: ['*'],
        createdAt: Date.now(),
        lastUsed: null,
        rateLimits: this.defaultRateLimits
      });
      
      console.log(`🔑 Default API key generated: ${defaultKey}`);
    } else {
      // Load configured API keys
      this.config.apiKeys.forEach(keyConfig => {
        this.apiKeys.set(keyConfig.key, {
          name: keyConfig.name,
          permissions: keyConfig.permissions || ['*'],
          createdAt: Date.now(),
          lastUsed: null,
          rateLimits: keyConfig.rateLimits || this.defaultRateLimits
        });
      });
    }
  }

  generateApiKey() {
    return 'snm_' + crypto.randomBytes(32).toString('hex');
  }

  async authorize(request) {
    if (!this.config.enableAuth) {
      return { authorized: true, user: 'anonymous' };
    }

    const authHeader = request.headers?.authorization;
    const apiKey = this.extractApiKey(authHeader);

    if (!apiKey) {
      throw new Error('Missing API key');
    }

    if (this.blacklist.has(apiKey)) {
      throw new Error('API key is blacklisted');
    }

    const keyInfo = this.apiKeys.get(apiKey);
    if (!keyInfo) {
      throw new Error('Invalid API key');
    }

    // Check rate limits
    await this.checkRateLimit(apiKey, keyInfo.rateLimits);

    // Check permissions
    const hasPermission = await this.checkPermissions(
      keyInfo.permissions,
      request.params?.name || request.method
    );

    if (!hasPermission) {
      throw new Error('Insufficient permissions');
    }

    // Update last used
    keyInfo.lastUsed = Date.now();

    return {
      authorized: true,
      user: keyInfo.name,
      permissions: keyInfo.permissions
    };
  }

  extractApiKey(authHeader) {
    if (!authHeader) return null;
    
    // Support multiple auth formats
    if (authHeader.startsWith('Bearer ')) {
      return authHeader.substring(7);
    } else if (authHeader.startsWith('ApiKey ')) {
      return authHeader.substring(7);
    } else {
      return authHeader;
    }
  }

  async checkRateLimit(apiKey, limits) {
    const now = Date.now();
    const windowStart = now - limits.window;

    let limiter = this.rateLimiters.get(apiKey);
    if (!limiter) {
      limiter = {
        requests: [],
        burstCount: 0,
        lastBurst: 0
      };
      this.rateLimiters.set(apiKey, limiter);
    }

    // Clean old requests
    limiter.requests = limiter.requests.filter(time => time > windowStart);

    // Check burst limit
    if (now - limiter.lastBurst < 1000) { // 1 second burst window
      if (limiter.burstCount >= limits.burst) {
        throw new Error('Burst rate limit exceeded');
      }
      limiter.burstCount++;
    } else {
      limiter.burstCount = 1;
      limiter.lastBurst = now;
    }

    // Check window limit
    if (limiter.requests.length >= limits.requests) {
      throw new Error('Rate limit exceeded');
    }

    limiter.requests.push(now);
  }

  async checkPermissions(userPermissions, resource) {
    // Wildcard permission
    if (userPermissions.includes('*')) {
      return true;
    }

    // Exact match
    if (userPermissions.includes(resource)) {
      return true;
    }

    // Pattern matching (e.g., "neural_*" matches "neural_mesh_init")
    for (const permission of userPermissions) {
      if (permission.includes('*')) {
        const pattern = permission.replace(/\*/g, '.*');
        const regex = new RegExp(`^${pattern}$`);
        if (regex.test(resource)) {
          return true;
        }
      }
    }

    return false;
  }

  createSession(userId, metadata = {}) {
    const sessionId = crypto.randomBytes(32).toString('hex');
    const session = {
      id: sessionId,
      userId,
      createdAt: Date.now(),
      lastActivity: Date.now(),
      metadata,
      expiresAt: Date.now() + (24 * 60 * 60 * 1000) // 24 hours
    };

    this.sessions.set(sessionId, session);
    
    // Clean expired sessions
    this.cleanExpiredSessions();
    
    return sessionId;
  }

  getSession(sessionId) {
    const session = this.sessions.get(sessionId);
    if (!session) return null;

    if (Date.now() > session.expiresAt) {
      this.sessions.delete(sessionId);
      return null;
    }

    session.lastActivity = Date.now();
    return session;
  }

  revokeSession(sessionId) {
    return this.sessions.delete(sessionId);
  }

  blacklistApiKey(apiKey, reason = 'Manual blacklist') {
    this.blacklist.add(apiKey);
    this.emit('keyBlacklisted', { apiKey, reason, timestamp: Date.now() });
    console.log(`🚫 API key blacklisted: ${apiKey.slice(0, 12)}... (${reason})`);
  }

  unblacklistApiKey(apiKey) {
    this.blacklist.delete(apiKey);
    this.emit('keyUnblacklisted', { apiKey, timestamp: Date.now() });
    console.log(`✅ API key unblacklisted: ${apiKey.slice(0, 12)}...`);
  }

  createApiKey(name, permissions = ['*'], rateLimits = null) {
    const apiKey = this.generateApiKey();
    const keyInfo = {
      name,
      permissions,
      createdAt: Date.now(),
      lastUsed: null,
      rateLimits: rateLimits || this.defaultRateLimits
    };

    this.apiKeys.set(apiKey, keyInfo);
    this.emit('keyCreated', { apiKey, keyInfo });
    
    return apiKey;
  }

  revokeApiKey(apiKey) {
    const removed = this.apiKeys.delete(apiKey);
    if (removed) {
      this.emit('keyRevoked', { apiKey, timestamp: Date.now() });
    }
    return removed;
  }

  listApiKeys() {
    return Array.from(this.apiKeys.entries()).map(([key, info]) => ({
      key: key.slice(0, 12) + '...',
      name: info.name,
      permissions: info.permissions,
      createdAt: info.createdAt,
      lastUsed: info.lastUsed
    }));
  }

  cleanExpiredSessions() {
    const now = Date.now();
    const expired = [];
    
    for (const [sessionId, session] of this.sessions) {
      if (now > session.expiresAt) {
        expired.push(sessionId);
      }
    }

    expired.forEach(sessionId => {
      this.sessions.delete(sessionId);
    });

    if (expired.length > 0) {
      console.log(`🧹 Cleaned ${expired.length} expired sessions`);
    }
  }

  getAuthStats() {
    return {
      totalApiKeys: this.apiKeys.size,
      activeSessions: this.sessions.size,
      blacklistedKeys: this.blacklist.size,
      rateLimiters: this.rateLimiters.size
    };
  }

  // Security utilities
  hashPassword(password, salt = null) {
    if (!salt) salt = crypto.randomBytes(16).toString('hex');
    const hash = crypto.pbkdf2Sync(password, salt, 10000, 64, 'sha512').toString('hex');
    return { hash, salt };
  }

  verifyPassword(password, hash, salt) {
    const { hash: computedHash } = this.hashPassword(password, salt);
    return hash === computedHash;
  }

  encryptData(data, key = null) {
    if (!key) key = crypto.randomBytes(32);
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipher('aes-256-cbc', key);
    
    let encrypted = cipher.update(JSON.stringify(data), 'utf8', 'hex');
    encrypted += cipher.final('hex');
    
    return {
      encrypted,
      iv: iv.toString('hex'),
      key: key.toString('hex')
    };
  }

  decryptData(encryptedData, key, iv) {
    const decipher = crypto.createDecipher('aes-256-cbc', Buffer.from(key, 'hex'));
    
    let decrypted = decipher.update(encryptedData, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    
    return JSON.parse(decrypted);
  }
}

export default AuthManager;