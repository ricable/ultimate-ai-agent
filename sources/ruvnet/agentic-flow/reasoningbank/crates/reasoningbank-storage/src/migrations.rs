//! Database schema migrations

use rusqlite::{Connection, Result as RusqliteResult};
use tracing::{info, debug};

/// Current schema version
const CURRENT_VERSION: i32 = 1;

/// Initialize or migrate database schema
pub fn migrate(conn: &Connection) -> RusqliteResult<()> {
    let current_version = get_schema_version(conn)?;

    info!("Current database schema version: {}", current_version);

    if current_version < CURRENT_VERSION {
        info!("Migrating database from version {} to {}", current_version, CURRENT_VERSION);

        if current_version < 1 {
            migrate_to_v1(conn)?;
        }

        set_schema_version(conn, CURRENT_VERSION)?;
        info!("Migration complete");
    } else {
        debug!("Database schema is up to date");
    }

    Ok(())
}

/// Get current schema version
fn get_schema_version(conn: &Connection) -> RusqliteResult<i32> {
    // Create schema_version table if it doesn't exist
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY
        )",
        [],
    )?;

    let version: i32 = conn
        .query_row("SELECT version FROM schema_version LIMIT 1", [], |row| {
            row.get(0)
        })
        .unwrap_or(0);

    Ok(version)
}

/// Set schema version
fn set_schema_version(conn: &Connection, version: i32) -> RusqliteResult<()> {
    conn.execute("DELETE FROM schema_version", [])?;
    conn.execute(
        "INSERT INTO schema_version (version) VALUES (?1)",
        [version],
    )?;
    Ok(())
}

/// Migrate to version 1
fn migrate_to_v1(conn: &Connection) -> RusqliteResult<()> {
    info!("Applying migration to v1");

    // Main patterns table
    conn.execute(
        "CREATE TABLE IF NOT EXISTS patterns (
            id TEXT PRIMARY KEY,
            task_description TEXT NOT NULL,
            task_category TEXT NOT NULL,
            strategy TEXT NOT NULL,
            context TEXT NOT NULL,
            outcome TEXT NOT NULL,
            metadata TEXT NOT NULL,
            embedding BLOB,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        )",
        [],
    )?;

    // Indexes for fast lookups
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_patterns_category
         ON patterns(task_category)",
        [],
    )?;

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_patterns_strategy
         ON patterns(strategy)",
        [],
    )?;

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_patterns_created
         ON patterns(created_at DESC)",
        [],
    )?;

    // Full-text search on task descriptions
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS patterns_fts
         USING fts5(id, task_description, content='patterns', content_rowid='rowid')",
        [],
    )?;

    // Trigger to keep FTS in sync
    conn.execute(
        "CREATE TRIGGER IF NOT EXISTS patterns_fts_insert
         AFTER INSERT ON patterns BEGIN
            INSERT INTO patterns_fts(rowid, id, task_description)
            VALUES (new.rowid, new.id, new.task_description);
         END",
        [],
    )?;

    conn.execute(
        "CREATE TRIGGER IF NOT EXISTS patterns_fts_update
         AFTER UPDATE ON patterns BEGIN
            UPDATE patterns_fts
            SET task_description = new.task_description
            WHERE rowid = new.rowid;
         END",
        [],
    )?;

    conn.execute(
        "CREATE TRIGGER IF NOT EXISTS patterns_fts_delete
         AFTER DELETE ON patterns BEGIN
            DELETE FROM patterns_fts WHERE rowid = old.rowid;
         END",
        [],
    )?;

    info!("Migration to v1 complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_migration() {
        let conn = Connection::open_in_memory().unwrap();
        migrate(&conn).unwrap();

        let version = get_schema_version(&conn).unwrap();
        assert_eq!(version, CURRENT_VERSION);
    }

    #[test]
    fn test_schema_tables_created() {
        let conn = Connection::open_in_memory().unwrap();
        migrate(&conn).unwrap();

        // Check that patterns table exists
        let table_exists: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type='table' AND name='patterns'",
                [],
                |row| row.get(0),
            )
            .unwrap();

        assert!(table_exists);
    }
}
