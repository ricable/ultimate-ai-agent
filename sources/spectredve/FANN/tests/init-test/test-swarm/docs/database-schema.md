# Test-Swarm Database Schema

## Database Overview

**Database Engine**: SQLite 3.40+  
**Connection Mode**: WAL (Write-Ahead Logging)  
**Encoding**: UTF-8  
**Page Size**: 4096 bytes  
**Cache Size**: 10000 pages (~40MB)  

## Schema Design Principles

1. **Normalization**: Third Normal Form (3NF) for data integrity
2. **Indexing**: Strategic indexes for query performance
3. **Constraints**: Foreign keys, unique constraints, and check constraints
4. **Audit Trail**: Created/updated timestamps on all tables
5. **Soft Deletes**: Logical deletion with `deleted_at` timestamps where appropriate

## Tables

### 1. Users Table

Stores user account information and authentication data.

```sql
CREATE TABLE users (
    -- Primary Key
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Unique Identifiers
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    
    -- Authentication
    password_hash VARCHAR(255) NOT NULL,
    password_salt VARCHAR(255) NOT NULL,
    password_iterations INTEGER NOT NULL DEFAULT 100000,
    
    -- Account Status
    is_active BOOLEAN NOT NULL DEFAULT 1,
    is_verified BOOLEAN NOT NULL DEFAULT 0,
    verification_token VARCHAR(255),
    verification_expires_at DATETIME,
    
    -- Security
    failed_login_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until DATETIME,
    last_password_change DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    password_reset_token VARCHAR(255),
    password_reset_expires_at DATETIME,
    
    -- Profile
    full_name VARCHAR(255),
    avatar_url VARCHAR(500),
    bio TEXT,
    
    -- Timestamps
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login_at DATETIME,
    deleted_at DATETIME,
    
    -- Constraints
    CHECK (length(username) >= 3),
    CHECK (length(password_hash) > 0),
    CHECK (failed_login_attempts >= 0),
    CHECK (password_iterations >= 10000)
);

-- Indexes
CREATE INDEX idx_users_email ON users(email) WHERE deleted_at IS NULL;
CREATE INDEX idx_users_username ON users(username) WHERE deleted_at IS NULL;
CREATE INDEX idx_users_verification_token ON users(verification_token) WHERE verification_token IS NOT NULL;
CREATE INDEX idx_users_password_reset_token ON users(password_reset_token) WHERE password_reset_token IS NOT NULL;
CREATE INDEX idx_users_created_at ON users(created_at);
CREATE INDEX idx_users_is_active ON users(is_active) WHERE deleted_at IS NULL;

-- Triggers
CREATE TRIGGER update_users_updated_at 
    AFTER UPDATE ON users
    FOR EACH ROW
    WHEN NEW.updated_at = OLD.updated_at
BEGIN
    UPDATE users SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
```

### 2. Sessions Table

Manages user sessions and refresh tokens.

```sql
CREATE TABLE sessions (
    -- Primary Key
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Foreign Keys
    user_id INTEGER NOT NULL,
    
    -- Token Management
    refresh_token VARCHAR(500) UNIQUE NOT NULL,
    refresh_token_family VARCHAR(255) NOT NULL,
    access_token_jti VARCHAR(255),
    
    -- Device Information
    device_fingerprint VARCHAR(255),
    device_name VARCHAR(255),
    device_type VARCHAR(50), -- desktop, mobile, tablet, other
    
    -- Network Information
    ip_address VARCHAR(45) NOT NULL,
    ip_country VARCHAR(2),
    ip_region VARCHAR(255),
    ip_city VARCHAR(255),
    
    -- User Agent
    user_agent TEXT,
    browser_name VARCHAR(50),
    browser_version VARCHAR(20),
    os_name VARCHAR(50),
    os_version VARCHAR(20),
    
    -- Session Management
    is_active BOOLEAN NOT NULL DEFAULT 1,
    last_activity DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME NOT NULL,
    
    -- Timestamps
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    revoked_at DATETIME,
    revoked_reason VARCHAR(255),
    
    -- Foreign Key Constraints
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    
    -- Check Constraints
    CHECK (expires_at > created_at),
    CHECK (device_type IN ('desktop', 'mobile', 'tablet', 'other'))
);

-- Indexes
CREATE INDEX idx_sessions_user_id ON sessions(user_id) WHERE is_active = 1;
CREATE INDEX idx_sessions_refresh_token ON sessions(refresh_token);
CREATE INDEX idx_sessions_refresh_token_family ON sessions(refresh_token_family);
CREATE INDEX idx_sessions_expires_at ON sessions(expires_at) WHERE is_active = 1;
CREATE INDEX idx_sessions_device_fingerprint ON sessions(device_fingerprint) WHERE device_fingerprint IS NOT NULL;
CREATE INDEX idx_sessions_created_at ON sessions(created_at);

-- Cleanup old sessions trigger
CREATE TRIGGER cleanup_expired_sessions
    AFTER INSERT ON sessions
BEGIN
    DELETE FROM sessions 
    WHERE expires_at < datetime('now', '-7 days') 
       OR (is_active = 0 AND revoked_at < datetime('now', '-30 days'));
END;
```

### 3. Roles Table

Defines available user roles.

```sql
CREATE TABLE roles (
    -- Primary Key
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Role Definition
    name VARCHAR(50) UNIQUE NOT NULL,
    display_name VARCHAR(100) NOT NULL,
    description TEXT,
    
    -- Role Hierarchy
    parent_role_id INTEGER,
    hierarchy_level INTEGER NOT NULL DEFAULT 0,
    
    -- Status
    is_active BOOLEAN NOT NULL DEFAULT 1,
    is_system BOOLEAN NOT NULL DEFAULT 0,
    
    -- Timestamps
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign Key Constraints
    FOREIGN KEY (parent_role_id) REFERENCES roles(id) ON DELETE SET NULL,
    
    -- Check Constraints
    CHECK (hierarchy_level >= 0)
);

-- Default Roles
INSERT INTO roles (name, display_name, description, is_system, hierarchy_level) VALUES
    ('super_admin', 'Super Administrator', 'Full system access', 1, 0),
    ('admin', 'Administrator', 'Administrative access', 1, 1),
    ('moderator', 'Moderator', 'Content moderation access', 1, 2),
    ('user', 'User', 'Standard user access', 1, 3);

-- Indexes
CREATE INDEX idx_roles_name ON roles(name);
CREATE INDEX idx_roles_parent_role_id ON roles(parent_role_id);
CREATE INDEX idx_roles_hierarchy_level ON roles(hierarchy_level);
```

### 4. User Roles Table

Many-to-many relationship between users and roles.

```sql
CREATE TABLE user_roles (
    -- Primary Key
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Foreign Keys
    user_id INTEGER NOT NULL,
    role_id INTEGER NOT NULL,
    
    -- Assignment Information
    assigned_by INTEGER,
    assigned_reason TEXT,
    
    -- Validity Period
    valid_from DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    valid_until DATETIME,
    
    -- Status
    is_active BOOLEAN NOT NULL DEFAULT 1,
    
    -- Timestamps
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    revoked_at DATETIME,
    revoked_by INTEGER,
    revoked_reason TEXT,
    
    -- Foreign Key Constraints
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE CASCADE,
    FOREIGN KEY (assigned_by) REFERENCES users(id) ON DELETE SET NULL,
    FOREIGN KEY (revoked_by) REFERENCES users(id) ON DELETE SET NULL,
    
    -- Unique Constraint
    UNIQUE(user_id, role_id, valid_from)
);

-- Indexes
CREATE INDEX idx_user_roles_user_id ON user_roles(user_id) WHERE is_active = 1;
CREATE INDEX idx_user_roles_role_id ON user_roles(role_id) WHERE is_active = 1;
CREATE INDEX idx_user_roles_valid_until ON user_roles(valid_until) WHERE valid_until IS NOT NULL;
```

### 5. Performance Metrics Table

Stores API performance metrics for monitoring.

```sql
CREATE TABLE performance_metrics (
    -- Primary Key
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Request Information
    request_id VARCHAR(255) UNIQUE NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    
    -- User Information
    user_id INTEGER,
    session_id INTEGER,
    ip_address VARCHAR(45),
    
    -- Performance Metrics
    response_time_ms INTEGER NOT NULL,
    database_time_ms INTEGER,
    processing_time_ms INTEGER,
    
    -- Resource Usage
    memory_usage_mb REAL,
    cpu_usage_percent REAL,
    database_queries INTEGER,
    cache_hits INTEGER,
    cache_misses INTEGER,
    
    -- Request/Response Size
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    
    -- Error Information
    error_code VARCHAR(50),
    error_message TEXT,
    error_stack TEXT,
    
    -- Timestamp
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign Key Constraints
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE SET NULL,
    
    -- Check Constraints
    CHECK (method IN ('GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'HEAD', 'OPTIONS')),
    CHECK (status_code >= 100 AND status_code < 600),
    CHECK (response_time_ms >= 0)
);

-- Indexes
CREATE INDEX idx_metrics_timestamp ON performance_metrics(timestamp);
CREATE INDEX idx_metrics_endpoint ON performance_metrics(endpoint);
CREATE INDEX idx_metrics_user_id ON performance_metrics(user_id);
CREATE INDEX idx_metrics_status_code ON performance_metrics(status_code);
CREATE INDEX idx_metrics_response_time ON performance_metrics(response_time_ms);

-- Partitioning simulation using views
CREATE VIEW performance_metrics_recent AS
    SELECT * FROM performance_metrics 
    WHERE timestamp > datetime('now', '-7 days');

-- Cleanup old metrics
CREATE TRIGGER cleanup_old_metrics
    AFTER INSERT ON performance_metrics
    WHEN (SELECT COUNT(*) FROM performance_metrics) > 1000000
BEGIN
    DELETE FROM performance_metrics 
    WHERE timestamp < datetime('now', '-30 days')
    ORDER BY timestamp ASC
    LIMIT 10000;
END;
```

### 6. Audit Log Table

Tracks all significant system events for security and compliance.

```sql
CREATE TABLE audit_log (
    -- Primary Key
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Event Information
    event_type VARCHAR(100) NOT NULL,
    event_category VARCHAR(50) NOT NULL,
    event_description TEXT,
    
    -- Actor Information
    user_id INTEGER,
    session_id INTEGER,
    ip_address VARCHAR(45),
    user_agent TEXT,
    
    -- Target Information
    target_type VARCHAR(50),
    target_id INTEGER,
    target_data JSON,
    
    -- Change Information
    old_values JSON,
    new_values JSON,
    
    -- Status
    success BOOLEAN NOT NULL DEFAULT 1,
    error_message TEXT,
    
    -- Timestamp
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign Key Constraints
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE SET NULL,
    
    -- Check Constraints
    CHECK (event_category IN ('authentication', 'authorization', 'user_management', 'data_access', 'system', 'security'))
);

-- Indexes
CREATE INDEX idx_audit_log_created_at ON audit_log(created_at);
CREATE INDEX idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX idx_audit_log_event_type ON audit_log(event_type);
CREATE INDEX idx_audit_log_event_category ON audit_log(event_category);
CREATE INDEX idx_audit_log_target ON audit_log(target_type, target_id);
```

### 7. API Keys Table

Manages API keys for programmatic access.

```sql
CREATE TABLE api_keys (
    -- Primary Key
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Foreign Keys
    user_id INTEGER NOT NULL,
    
    -- Key Information
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    key_prefix VARCHAR(10) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Permissions
    scopes JSON NOT NULL DEFAULT '[]',
    allowed_ips JSON,
    allowed_origins JSON,
    
    -- Rate Limiting
    rate_limit_per_hour INTEGER DEFAULT 1000,
    rate_limit_per_day INTEGER DEFAULT 10000,
    
    -- Usage Tracking
    last_used_at DATETIME,
    usage_count INTEGER NOT NULL DEFAULT 0,
    
    -- Status
    is_active BOOLEAN NOT NULL DEFAULT 1,
    expires_at DATETIME,
    
    -- Timestamps
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    revoked_at DATETIME,
    revoked_reason VARCHAR(255),
    
    -- Foreign Key Constraints
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    
    -- Check Constraints
    CHECK (rate_limit_per_hour > 0),
    CHECK (rate_limit_per_day > 0),
    CHECK (usage_count >= 0)
);

-- Indexes
CREATE INDEX idx_api_keys_key_prefix ON api_keys(key_prefix);
CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX idx_api_keys_expires_at ON api_keys(expires_at) WHERE is_active = 1;
CREATE INDEX idx_api_keys_last_used_at ON api_keys(last_used_at);
```

### 8. Cache Table

Simple key-value cache for performance optimization.

```sql
CREATE TABLE cache (
    -- Primary Key
    key VARCHAR(255) PRIMARY KEY,
    
    -- Cache Data
    value TEXT NOT NULL,
    
    -- Metadata
    tags JSON,
    hit_count INTEGER NOT NULL DEFAULT 0,
    
    -- Expiration
    expires_at DATETIME NOT NULL,
    
    -- Timestamps
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    accessed_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Check Constraints
    CHECK (expires_at > created_at),
    CHECK (hit_count >= 0)
);

-- Indexes
CREATE INDEX idx_cache_expires_at ON cache(expires_at);
CREATE INDEX idx_cache_accessed_at ON cache(accessed_at);

-- Auto-cleanup expired entries
CREATE TRIGGER cleanup_expired_cache
    AFTER INSERT ON cache
BEGIN
    DELETE FROM cache WHERE expires_at < CURRENT_TIMESTAMP;
END;
```

## Views

### 1. Active Users View

```sql
CREATE VIEW active_users AS
SELECT 
    u.id,
    u.username,
    u.email,
    u.created_at,
    u.last_login_at,
    COUNT(DISTINCT s.id) as active_sessions,
    GROUP_CONCAT(DISTINCT r.name) as roles
FROM users u
LEFT JOIN sessions s ON u.id = s.user_id AND s.is_active = 1
LEFT JOIN user_roles ur ON u.id = ur.user_id AND ur.is_active = 1
LEFT JOIN roles r ON ur.role_id = r.id
WHERE u.is_active = 1 
    AND u.deleted_at IS NULL
GROUP BY u.id;
```

### 2. Performance Summary View

```sql
CREATE VIEW performance_summary AS
SELECT 
    endpoint,
    method,
    COUNT(*) as request_count,
    AVG(response_time_ms) as avg_response_time,
    MIN(response_time_ms) as min_response_time,
    MAX(response_time_ms) as max_response_time,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY response_time_ms) as p50,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY response_time_ms) as p99,
    SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as error_count,
    AVG(memory_usage_mb) as avg_memory_usage,
    DATE(timestamp) as date
FROM performance_metrics
WHERE timestamp > datetime('now', '-7 days')
GROUP BY endpoint, method, DATE(timestamp);
```

## Database Maintenance

### 1. Vacuum Schedule

```sql
-- Run weekly
VACUUM;
ANALYZE;
```

### 2. Index Maintenance

```sql
-- Check index usage
SELECT 
    name,
    tbl_name,
    sql
FROM sqlite_master 
WHERE type = 'index' 
    AND name NOT LIKE 'sqlite_%'
ORDER BY tbl_name;

-- Reindex if needed
REINDEX;
```

### 3. Data Retention

```sql
-- Clean up old data (run daily)
DELETE FROM performance_metrics WHERE timestamp < datetime('now', '-90 days');
DELETE FROM audit_log WHERE created_at < datetime('now', '-365 days');
DELETE FROM sessions WHERE expires_at < datetime('now', '-30 days') AND is_active = 0;
```

## Migration Scripts

### Initial Schema Creation

```sql
-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- Enable WAL mode
PRAGMA journal_mode = WAL;

-- Set page size
PRAGMA page_size = 4096;

-- Set cache size
PRAGMA cache_size = 10000;

-- Create tables in order
-- 1. Users
-- 2. Roles
-- 3. Sessions
-- 4. User Roles
-- 5. Performance Metrics
-- 6. Audit Log
-- 7. API Keys
-- 8. Cache

-- Create indexes
-- Create triggers
-- Create views
-- Insert default data
```

### Schema Version Tracking

```sql
CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    applied_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Track migrations
INSERT INTO schema_migrations (version, name) VALUES
    (1, 'initial_schema'),
    (2, 'add_api_keys'),
    (3, 'add_cache_table'),
    (4, 'add_audit_indexes');
```

## Performance Considerations

### 1. Query Optimization

- Use covering indexes for frequently accessed columns
- Avoid SELECT * queries
- Use EXPLAIN QUERY PLAN for optimization
- Batch INSERT operations when possible

### 2. Connection Settings

```sql
-- Recommended connection settings
PRAGMA synchronous = NORMAL;
PRAGMA temp_store = MEMORY;
PRAGMA mmap_size = 30000000000;
PRAGMA busy_timeout = 5000;
```

### 3. Backup Strategy

```bash
# Daily backups
sqlite3 test-swarm.db ".backup backup-$(date +%Y%m%d).db"

# Continuous backup using WAL
sqlite3 test-swarm.db "PRAGMA wal_checkpoint(TRUNCATE);"
```

## Security Considerations

1. **Encryption**: Use SQLCipher for at-rest encryption
2. **Access Control**: Implement row-level security in application layer
3. **SQL Injection**: Always use parameterized queries
4. **Audit Trail**: Log all data modifications
5. **Backup Encryption**: Encrypt all backup files
6. **Connection Security**: Use TLS for remote connections