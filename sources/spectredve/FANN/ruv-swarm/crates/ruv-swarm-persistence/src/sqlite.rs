//! SQLite backend implementation for native platforms

use crate::{models::*, Storage, StorageError, Transaction as TransactionTrait};
use async_trait::async_trait;
use parking_lot::Mutex;
use r2d2::{Pool, PooledConnection};
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::{params, OptionalExtension};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info};

type SqlitePool = Pool<SqliteConnectionManager>;
type SqliteConn = PooledConnection<SqliteConnectionManager>;

/// SQLite storage implementation
pub struct SqliteStorage {
    pool: Arc<SqlitePool>,
    path: String,
}

impl SqliteStorage {
    /// Create new SQLite storage instance
    pub async fn new(path: &str) -> Result<Self, StorageError> {
        let manager = SqliteConnectionManager::file(path);

        let pool = Pool::builder()
            .max_size(16)
            .min_idle(Some(2))
            .connection_timeout(Duration::from_secs(30))
            .idle_timeout(Some(Duration::from_secs(300)))
            .build(manager)
            .map_err(|e| StorageError::Pool(e.to_string()))?;

        let storage = Self {
            pool: Arc::new(pool),
            path: path.to_string(),
        };

        // Initialize schema
        storage.init_schema().await?;

        info!("SQLite storage initialized at: {}", path);
        Ok(storage)
    }

    /// Get connection from pool
    fn get_conn(&self) -> Result<SqliteConn, StorageError> {
        self.pool
            .get()
            .map_err(|e| StorageError::Pool(e.to_string()))
    }

    /// Initialize database schema
    async fn init_schema(&self) -> Result<(), StorageError> {
        let conn = self.get_conn()?;

        conn.execute_batch(include_str!("../sql/schema.sql"))
            .map_err(|e| StorageError::Migration(format!("Schema initialization failed: {}", e)))?;

        Ok(())
    }
}

#[async_trait]
impl Storage for SqliteStorage {
    type Error = StorageError;

    // Agent operations
    async fn store_agent(&self, agent: &AgentModel) -> Result<(), Self::Error> {
        let conn = self.get_conn()?;
        let json = serde_json::to_string(agent)?;

        conn.execute(
            "INSERT INTO agents (id, name, agent_type, status, capabilities, metadata, heartbeat, created_at, updated_at, data) 
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                agent.id,
                agent.name,
                agent.agent_type,
                agent.status.to_string(),
                serde_json::to_string(&agent.capabilities)?,
                serde_json::to_string(&agent.metadata)?,
                agent.heartbeat.timestamp(),
                agent.created_at.timestamp(),
                agent.updated_at.timestamp(),
                json
            ],
        )
        .map_err(|e| StorageError::Database(e.to_string()))?;

        debug!("Stored agent: {}", agent.id);
        Ok(())
    }

    async fn get_agent(&self, id: &str) -> Result<Option<AgentModel>, Self::Error> {
        let conn = self.get_conn()?;

        let result = conn
            .query_row(
                "SELECT data FROM agents WHERE id = ?1",
                params![id],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(|e| StorageError::Database(e.to_string()))?;

        match result {
            Some(json) => Ok(Some(serde_json::from_str(&json)?)),
            None => Ok(None),
        }
    }

    async fn update_agent(&self, agent: &AgentModel) -> Result<(), Self::Error> {
        let conn = self.get_conn()?;
        let json = serde_json::to_string(agent)?;

        let rows = conn
            .execute(
                "UPDATE agents 
             SET name = ?2, agent_type = ?3, status = ?4, capabilities = ?5, 
                 metadata = ?6, heartbeat = ?7, updated_at = ?8, data = ?9
             WHERE id = ?1",
                params![
                    agent.id,
                    agent.name,
                    agent.agent_type,
                    agent.status.to_string(),
                    serde_json::to_string(&agent.capabilities)?,
                    serde_json::to_string(&agent.metadata)?,
                    agent.heartbeat.timestamp(),
                    agent.updated_at.timestamp(),
                    json
                ],
            )
            .map_err(|e| StorageError::Database(e.to_string()))?;

        if rows == 0 {
            return Err(StorageError::NotFound(format!(
                "Agent {} not found",
                agent.id
            )));
        }

        debug!("Updated agent: {}", agent.id);
        Ok(())
    }

    async fn delete_agent(&self, id: &str) -> Result<(), Self::Error> {
        let conn = self.get_conn()?;

        let rows = conn
            .execute("DELETE FROM agents WHERE id = ?1", params![id])
            .map_err(|e| StorageError::Database(e.to_string()))?;

        if rows == 0 {
            return Err(StorageError::NotFound(format!("Agent {} not found", id)));
        }

        debug!("Deleted agent: {}", id);
        Ok(())
    }

    async fn list_agents(&self) -> Result<Vec<AgentModel>, Self::Error> {
        let conn = self.get_conn()?;

        let mut stmt = conn
            .prepare("SELECT data FROM agents ORDER BY created_at DESC")
            .map_err(|e| StorageError::Database(e.to_string()))?;

        let agents = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(|e| StorageError::Database(e.to_string()))?
            .filter_map(|r| r.ok())
            .filter_map(|json| serde_json::from_str(&json).ok())
            .collect();

        Ok(agents)
    }

    async fn list_agents_by_status(&self, status: &str) -> Result<Vec<AgentModel>, Self::Error> {
        let conn = self.get_conn()?;

        let mut stmt = conn
            .prepare("SELECT data FROM agents WHERE status = ?1 ORDER BY created_at DESC")
            .map_err(|e| StorageError::Database(e.to_string()))?;

        let agents = stmt
            .query_map(params![status], |row| row.get::<_, String>(0))
            .map_err(|e| StorageError::Database(e.to_string()))?
            .filter_map(|r| r.ok())
            .filter_map(|json| serde_json::from_str(&json).ok())
            .collect();

        Ok(agents)
    }

    // Task operations
    async fn store_task(&self, task: &TaskModel) -> Result<(), Self::Error> {
        let conn = self.get_conn()?;
        let json = serde_json::to_string(task)?;

        conn.execute(
            "INSERT INTO tasks (id, task_type, priority, status, assigned_to, payload, 
                                created_at, updated_at, data) 
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                task.id,
                task.task_type,
                task.priority as i32,
                serde_json::to_string(&task.status)?,
                task.assigned_to,
                serde_json::to_string(&task.payload)?,
                task.created_at.timestamp(),
                task.updated_at.timestamp(),
                json
            ],
        )
        .map_err(|e| StorageError::Database(e.to_string()))?;

        debug!("Stored task: {}", task.id);
        Ok(())
    }

    async fn get_task(&self, id: &str) -> Result<Option<TaskModel>, Self::Error> {
        let conn = self.get_conn()?;

        let result = conn
            .query_row("SELECT data FROM tasks WHERE id = ?1", params![id], |row| {
                row.get::<_, String>(0)
            })
            .optional()
            .map_err(|e| StorageError::Database(e.to_string()))?;

        match result {
            Some(json) => Ok(Some(serde_json::from_str(&json)?)),
            None => Ok(None),
        }
    }

    async fn update_task(&self, task: &TaskModel) -> Result<(), Self::Error> {
        let conn = self.get_conn()?;
        let json = serde_json::to_string(task)?;

        let rows = conn
            .execute(
                "UPDATE tasks 
             SET task_type = ?2, priority = ?3, status = ?4, assigned_to = ?5, 
                 payload = ?6, updated_at = ?7, data = ?8
             WHERE id = ?1",
                params![
                    task.id,
                    task.task_type,
                    task.priority as i32,
                    serde_json::to_string(&task.status)?,
                    task.assigned_to,
                    serde_json::to_string(&task.payload)?,
                    task.updated_at.timestamp(),
                    json
                ],
            )
            .map_err(|e| StorageError::Database(e.to_string()))?;

        if rows == 0 {
            return Err(StorageError::NotFound(format!(
                "Task {} not found",
                task.id
            )));
        }

        debug!("Updated task: {}", task.id);
        Ok(())
    }

    async fn get_pending_tasks(&self) -> Result<Vec<TaskModel>, Self::Error> {
        let conn = self.get_conn()?;

        let mut stmt = conn
            .prepare(
                "SELECT data FROM tasks 
                 WHERE status = 'pending' 
                 ORDER BY priority DESC, created_at ASC",
            )
            .map_err(|e| StorageError::Database(e.to_string()))?;

        let tasks = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(|e| StorageError::Database(e.to_string()))?
            .filter_map(|r| r.ok())
            .filter_map(|json| serde_json::from_str(&json).ok())
            .collect();

        Ok(tasks)
    }

    async fn get_tasks_by_agent(&self, agent_id: &str) -> Result<Vec<TaskModel>, Self::Error> {
        let conn = self.get_conn()?;

        let mut stmt = conn
            .prepare(
                "SELECT data FROM tasks 
                 WHERE assigned_to = ?1 
                 ORDER BY priority DESC, created_at ASC",
            )
            .map_err(|e| StorageError::Database(e.to_string()))?;

        let tasks = stmt
            .query_map(params![agent_id], |row| row.get::<_, String>(0))
            .map_err(|e| StorageError::Database(e.to_string()))?
            .filter_map(|r| r.ok())
            .filter_map(|json| serde_json::from_str(&json).ok())
            .collect();

        Ok(tasks)
    }

    async fn claim_task(&self, task_id: &str, agent_id: &str) -> Result<bool, Self::Error> {
        let conn = self.get_conn()?;

        let rows = conn
            .execute(
                "UPDATE tasks 
             SET assigned_to = ?2, status = 'assigned', updated_at = ?3 
             WHERE id = ?1 AND status = 'pending'",
                params![task_id, agent_id, chrono::Utc::now().timestamp()],
            )
            .map_err(|e| StorageError::Database(e.to_string()))?;

        Ok(rows > 0)
    }

    // Event operations
    async fn store_event(&self, event: &EventModel) -> Result<(), Self::Error> {
        let conn = self.get_conn()?;
        let json = serde_json::to_string(event)?;

        conn.execute(
            "INSERT INTO events (id, event_type, agent_id, task_id, payload, metadata, 
                                 timestamp, sequence, data) 
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                event.id,
                event.event_type,
                event.agent_id,
                event.task_id,
                serde_json::to_string(&event.payload)?,
                serde_json::to_string(&event.metadata)?,
                event.timestamp.timestamp(),
                event.sequence as i64,
                json
            ],
        )
        .map_err(|e| StorageError::Database(e.to_string()))?;

        debug!("Stored event: {}", event.id);
        Ok(())
    }

    async fn get_events_by_agent(
        &self,
        agent_id: &str,
        limit: usize,
    ) -> Result<Vec<EventModel>, Self::Error> {
        let conn = self.get_conn()?;

        let mut stmt = conn
            .prepare(
                "SELECT data FROM events 
                 WHERE agent_id = ?1 
                 ORDER BY timestamp DESC 
                 LIMIT ?2",
            )
            .map_err(|e| StorageError::Database(e.to_string()))?;

        let events = stmt
            .query_map(params![agent_id, limit], |row| row.get::<_, String>(0))
            .map_err(|e| StorageError::Database(e.to_string()))?
            .filter_map(|r| r.ok())
            .filter_map(|json| serde_json::from_str(&json).ok())
            .collect();

        Ok(events)
    }

    async fn get_events_by_type(
        &self,
        event_type: &str,
        limit: usize,
    ) -> Result<Vec<EventModel>, Self::Error> {
        let conn = self.get_conn()?;

        let mut stmt = conn
            .prepare(
                "SELECT data FROM events 
                 WHERE event_type = ?1 
                 ORDER BY timestamp DESC 
                 LIMIT ?2",
            )
            .map_err(|e| StorageError::Database(e.to_string()))?;

        let events = stmt
            .query_map(params![event_type, limit], |row| {
                row.get::<_, String>(0)
            })
            .map_err(|e| StorageError::Database(e.to_string()))?
            .filter_map(|r| r.ok())
            .filter_map(|json| serde_json::from_str(&json).ok())
            .collect();

        Ok(events)
    }

    async fn get_events_since(&self, timestamp: i64) -> Result<Vec<EventModel>, Self::Error> {
        let conn = self.get_conn()?;

        let mut stmt = conn
            .prepare(
                "SELECT data FROM events 
                 WHERE timestamp > ?1 
                 ORDER BY timestamp ASC",
            )
            .map_err(|e| StorageError::Database(e.to_string()))?;

        let events = stmt
            .query_map(params![timestamp], |row| row.get::<_, String>(0))
            .map_err(|e| StorageError::Database(e.to_string()))?
            .filter_map(|r| r.ok())
            .filter_map(|json| serde_json::from_str(&json).ok())
            .collect();

        Ok(events)
    }

    // Message operations
    async fn store_message(&self, message: &MessageModel) -> Result<(), Self::Error> {
        let conn = self.get_conn()?;
        let json = serde_json::to_string(message)?;

        conn.execute(
            "INSERT INTO messages (id, from_agent, to_agent, message_type, content, 
                                   priority, read, created_at, data) 
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                message.id,
                message.from_agent,
                message.to_agent,
                message.message_type,
                serde_json::to_string(&message.content)?,
                serde_json::to_string(&message.priority)?,
                message.read as i32,
                message.created_at.timestamp(),
                json
            ],
        )
        .map_err(|e| StorageError::Database(e.to_string()))?;

        debug!("Stored message: {}", message.id);
        Ok(())
    }

    async fn get_messages_between(
        &self,
        agent1: &str,
        agent2: &str,
        limit: usize,
    ) -> Result<Vec<MessageModel>, Self::Error> {
        let conn = self.get_conn()?;

        let mut stmt = conn
            .prepare(
                "SELECT data FROM messages 
                 WHERE (from_agent = ?1 AND to_agent = ?2) OR (from_agent = ?2 AND to_agent = ?1) 
                 ORDER BY created_at DESC 
                 LIMIT ?3",
            )
            .map_err(|e| StorageError::Database(e.to_string()))?;

        let messages = stmt
            .query_map(params![agent1, agent2, limit], |row| {
                row.get::<_, String>(0)
            })
            .map_err(|e| StorageError::Database(e.to_string()))?
            .filter_map(|r| r.ok())
            .filter_map(|json| serde_json::from_str(&json).ok())
            .collect();

        Ok(messages)
    }

    async fn get_unread_messages(&self, agent_id: &str) -> Result<Vec<MessageModel>, Self::Error> {
        let conn = self.get_conn()?;

        let mut stmt = conn
            .prepare(
                "SELECT data FROM messages 
                 WHERE to_agent = ?1 AND read = 0 
                 ORDER BY created_at ASC",
            )
            .map_err(|e| StorageError::Database(e.to_string()))?;

        let messages = stmt
            .query_map(params![agent_id], |row| row.get::<_, String>(0))
            .map_err(|e| StorageError::Database(e.to_string()))?
            .filter_map(|r| r.ok())
            .filter_map(|json| serde_json::from_str(&json).ok())
            .collect();

        Ok(messages)
    }

    async fn mark_message_read(&self, message_id: &str) -> Result<(), Self::Error> {
        let conn = self.get_conn()?;

        let rows = conn
            .execute(
                "UPDATE messages SET read = 1, read_at = ?2 WHERE id = ?1",
                params![message_id, chrono::Utc::now().timestamp()],
            )
            .map_err(|e| StorageError::Database(e.to_string()))?;

        if rows == 0 {
            return Err(StorageError::NotFound(format!(
                "Message {} not found",
                message_id
            )));
        }

        Ok(())
    }

    // Metric operations
    async fn store_metric(&self, metric: &MetricModel) -> Result<(), Self::Error> {
        let conn = self.get_conn()?;
        let json = serde_json::to_string(metric)?;

        conn.execute(
            "INSERT INTO metrics (id, metric_type, agent_id, value, unit, tags, timestamp, data) 
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                metric.id,
                metric.metric_type,
                metric.agent_id,
                metric.value,
                metric.unit,
                serde_json::to_string(&metric.tags)?,
                metric.timestamp.timestamp(),
                json
            ],
        )
        .map_err(|e| StorageError::Database(e.to_string()))?;

        debug!("Stored metric: {}", metric.id);
        Ok(())
    }

    async fn get_metrics_by_agent(
        &self,
        agent_id: &str,
        metric_type: &str,
    ) -> Result<Vec<MetricModel>, Self::Error> {
        let conn = self.get_conn()?;

        let mut stmt = conn
            .prepare(
                "SELECT data FROM metrics 
                 WHERE agent_id = ?1 AND metric_type = ?2 
                 ORDER BY timestamp DESC",
            )
            .map_err(|e| StorageError::Database(e.to_string()))?;

        let metrics = stmt
            .query_map(params![agent_id, metric_type], |row| {
                row.get::<_, String>(0)
            })
            .map_err(|e| StorageError::Database(e.to_string()))?
            .filter_map(|r| r.ok())
            .filter_map(|json| serde_json::from_str(&json).ok())
            .collect();

        Ok(metrics)
    }

    async fn get_aggregated_metrics(
        &self,
        metric_type: &str,
        start_time: i64,
        end_time: i64,
    ) -> Result<Vec<MetricModel>, Self::Error> {
        let conn = self.get_conn()?;

        let mut stmt = conn
            .prepare(
                "SELECT metric_type, agent_id, AVG(value) as value, unit, 
                        MIN(timestamp) as timestamp, COUNT(*) as count
                 FROM metrics 
                 WHERE metric_type = ?1 AND timestamp >= ?2 AND timestamp <= ?3 
                 GROUP BY metric_type, agent_id, unit",
            )
            .map_err(|e| StorageError::Database(e.to_string()))?;

        let metrics = stmt
            .query_map(params![metric_type, start_time, end_time], |row| {
                let mut metric = MetricModel::new(
                    row.get::<_, String>(0)?,
                    row.get::<_, f64>(2)?,
                    row.get::<_, String>(3)?,
                );
                metric.agent_id = row.get::<_, Option<String>>(1)?;
                metric
                    .tags
                    .insert("count".to_string(), row.get::<_, i64>(5)?.to_string());
                Ok(metric)
            })
            .map_err(|e| StorageError::Database(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(metrics)
    }

    // Transaction support
    async fn begin_transaction(&self) -> Result<Box<dyn TransactionTrait>, Self::Error> {
        let conn = self.get_conn()?;
        Ok(Box::new(SqliteTransaction::new(conn)))
    }

    // Maintenance operations
    async fn vacuum(&self) -> Result<(), Self::Error> {
        let conn = self.get_conn()?;
        conn.execute("VACUUM", [])
            .map_err(|e| StorageError::Database(e.to_string()))?;
        info!("Database vacuumed");
        Ok(())
    }

    async fn checkpoint(&self) -> Result<(), Self::Error> {
        let conn = self.get_conn()?;
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)", [])
            .map_err(|e| StorageError::Database(e.to_string()))?;
        info!("Database checkpoint completed");
        Ok(())
    }

    async fn get_storage_size(&self) -> Result<u64, Self::Error> {
        let metadata =
            std::fs::metadata(&self.path).map_err(|e| StorageError::Other(e.to_string()))?;
        Ok(metadata.len())
    }
}

/// SQLite transaction wrapper
struct SqliteTransaction {
    conn: Mutex<Option<SqliteConn>>,
}

impl SqliteTransaction {
    fn new(conn: SqliteConn) -> Self {
        Self {
            conn: Mutex::new(Some(conn)),
        }
    }
}

#[async_trait]
impl TransactionTrait for SqliteTransaction {
    async fn commit(self: Box<Self>) -> Result<(), StorageError> {
        if let Some(conn) = self.conn.lock().take() {
            // In SQLite, we'd need to use explicit transactions
            // For now, we're using auto-commit mode
            drop(conn);
        }
        Ok(())
    }

    async fn rollback(self: Box<Self>) -> Result<(), StorageError> {
        if let Some(conn) = self.conn.lock().take() {
            // In SQLite, we'd need to use explicit transactions
            // For now, we're using auto-commit mode
            drop(conn);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_sqlite_storage() {
        let temp_file = NamedTempFile::new().unwrap();
        let storage = SqliteStorage::new(temp_file.path().to_str().unwrap())
            .await
            .unwrap();

        // Test agent operations
        let agent = AgentModel::new(
            "test-agent".to_string(),
            "worker".to_string(),
            vec!["compute".to_string()],
        );

        storage.store_agent(&agent).await.unwrap();
        let retrieved = storage.get_agent(&agent.id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "test-agent");

        // Test task operations
        let task = TaskModel::new(
            "process".to_string(),
            serde_json::json!({"data": "test"}),
            TaskPriority::High,
        );

        storage.store_task(&task).await.unwrap();
        let pending = storage.get_pending_tasks().await.unwrap();
        assert_eq!(pending.len(), 1);
    }
}
