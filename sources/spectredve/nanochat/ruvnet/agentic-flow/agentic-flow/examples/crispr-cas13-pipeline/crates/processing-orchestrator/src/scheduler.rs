//! Job scheduling

use crate::jobs::Job;

pub struct Scheduler {
    // TODO: Implement job scheduling logic
}

impl Scheduler {
    pub fn new() -> Self {
        Self {}
    }

    pub async fn schedule(&self, job: Job) -> Result<(), Box<dyn std::error::Error>> {
        tracing::info!("Scheduling job: {:?}", job.id);
        Ok(())
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}
