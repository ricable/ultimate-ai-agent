//! Job management

use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: Uuid,
    pub job_type: JobType,
    pub status: JobStatus,
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JobType {
    Alignment,
    OffTargetPrediction,
    DifferentialExpression,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum JobStatus {
    Pending,
    Processing,
    Completed,
    Failed,
}
