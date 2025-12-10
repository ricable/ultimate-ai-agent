//! Worker pool management

pub struct WorkerPool {
    size: usize,
}

impl WorkerPool {
    pub fn new(size: usize) -> Self {
        tracing::info!("Creating worker pool with {} workers", size);
        Self { size }
    }

    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        tracing::info!("Starting worker pool");
        Ok(())
    }
}
