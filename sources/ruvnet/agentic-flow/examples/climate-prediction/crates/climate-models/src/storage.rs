//! Model serialization and storage

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::{info, warn};

use crate::{ModelConfig, Result, ModelError};

/// Model version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    /// Semantic version (e.g., "1.0.0")
    pub version: String,

    /// Model architecture
    pub architecture: String,

    /// Training timestamp
    pub trained_at: String,

    /// Training metrics
    pub metrics: std::collections::HashMap<String, f64>,

    /// Model configuration
    pub config: ModelConfig,

    /// File path to model weights
    pub weights_path: PathBuf,

    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl ModelVersion {
    /// Create a new model version
    pub fn new(
        version: impl Into<String>,
        architecture: impl Into<String>,
        config: ModelConfig,
        weights_path: PathBuf,
    ) -> Self {
        Self {
            version: version.into(),
            architecture: architecture.into(),
            trained_at: chrono::Utc::now().to_rfc3339(),
            metrics: std::collections::HashMap::new(),
            config,
            weights_path,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add a metric to the version
    pub fn with_metric(mut self, name: impl Into<String>, value: f64) -> Self {
        self.metrics.insert(name.into(), value);
        self
    }

    /// Add metadata to the version
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Model storage manager
pub struct ModelStorage {
    base_path: PathBuf,
}

impl ModelStorage {
    /// Create a new model storage at the given path
    pub fn new(base_path: impl Into<PathBuf>) -> Result<Self> {
        let base_path = base_path.into();

        // Create directory if it doesn't exist
        if !base_path.exists() {
            std::fs::create_dir_all(&base_path)?;
        }

        Ok(Self { base_path })
    }

    /// Save a model version
    pub fn save_version(&self, version: &ModelVersion) -> Result<()> {
        let version_dir = self.base_path.join(&version.architecture).join(&version.version);

        std::fs::create_dir_all(&version_dir)?;

        // Save version metadata
        let metadata_path = version_dir.join("metadata.json");
        let json = serde_json::to_string_pretty(version)?;
        std::fs::write(metadata_path, json)?;

        // Copy weights file if it's not already in the version directory
        if !version.weights_path.starts_with(&version_dir) {
            let weights_filename = version.weights_path
                .file_name()
                .ok_or_else(|| ModelError::SerializationError("Invalid weights path".to_string()))?;
            let dest_path = version_dir.join(weights_filename);

            std::fs::copy(&version.weights_path, &dest_path)?;
        }

        info!("Saved model version {} to {}", version.version, version_dir.display());
        Ok(())
    }

    /// Load a specific model version
    pub fn load_version(&self, architecture: &str, version: &str) -> Result<ModelVersion> {
        let version_dir = self.base_path.join(architecture).join(version);
        let metadata_path = version_dir.join("metadata.json");

        if !metadata_path.exists() {
            return Err(ModelError::ModelNotFound(
                format!("Version {} of {} not found", version, architecture)
            ));
        }

        let json = std::fs::read_to_string(metadata_path)?;
        let model_version: ModelVersion = serde_json::from_str(&json)?;

        Ok(model_version)
    }

    /// List all versions of a model architecture
    pub fn list_versions(&self, architecture: &str) -> Result<Vec<ModelVersion>> {
        let arch_dir = self.base_path.join(architecture);

        if !arch_dir.exists() {
            return Ok(Vec::new());
        }

        let mut versions = Vec::new();

        for entry in std::fs::read_dir(arch_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                let version_name = entry.file_name().to_string_lossy().to_string();

                match self.load_version(architecture, &version_name) {
                    Ok(version) => versions.push(version),
                    Err(e) => warn!("Failed to load version {}: {}", version_name, e),
                }
            }
        }

        // Sort by version (most recent first)
        versions.sort_by(|a, b| b.version.cmp(&a.version));

        Ok(versions)
    }

    /// Get the latest version of a model
    pub fn latest_version(&self, architecture: &str) -> Result<ModelVersion> {
        let versions = self.list_versions(architecture)?;

        versions.into_iter().next()
            .ok_or_else(|| ModelError::ModelNotFound(
                format!("No versions found for {}", architecture)
            ))
    }

    /// Delete a model version
    pub fn delete_version(&self, architecture: &str, version: &str) -> Result<()> {
        let version_dir = self.base_path.join(architecture).join(version);

        if version_dir.exists() {
            std::fs::remove_dir_all(version_dir)?;
            info!("Deleted version {} of {}", version, architecture);
        }

        Ok(())
    }

    /// Export a model version to ONNX
    pub fn export_to_onnx(&self, version: &ModelVersion, output_path: &Path) -> Result<()> {
        // Copy the weights file (assuming it's already in ONNX format)
        std::fs::copy(&version.weights_path, output_path)?;

        info!("Exported model to ONNX: {}", output_path.display());
        Ok(())
    }

    /// Create a model registry manifest
    pub fn create_manifest(&self) -> Result<ModelManifest> {
        let mut manifest = ModelManifest {
            created_at: chrono::Utc::now().to_rfc3339(),
            models: Vec::new(),
        };

        // Scan all architectures
        for entry in std::fs::read_dir(&self.base_path)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                let architecture = entry.file_name().to_string_lossy().to_string();
                let versions = self.list_versions(&architecture)?;

                if !versions.is_empty() {
                    manifest.models.push(ModelEntry {
                        architecture,
                        versions,
                    });
                }
            }
        }

        Ok(manifest)
    }
}

/// Model registry manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub created_at: String,
    pub models: Vec<ModelEntry>,
}

/// Entry in the model manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub architecture: String,
    pub versions: Vec<ModelVersion>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_model_version_creation() {
        let config = ModelConfig::lstm(10, 64, 2, 3);
        let version = ModelVersion::new(
            "1.0.0",
            "LSTM",
            config,
            PathBuf::from("model.onnx"),
        );

        assert_eq!(version.version, "1.0.0");
        assert_eq!(version.architecture, "LSTM");
    }

    #[test]
    fn test_storage_operations() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let storage = ModelStorage::new(temp_dir.path())?;

        let config = ModelConfig::lstm(10, 64, 2, 3);

        // Create a temporary weights file
        let weights_path = temp_dir.path().join("test_model.onnx");
        std::fs::write(&weights_path, b"test weights")?;

        let version = ModelVersion::new("1.0.0", "LSTM", config, weights_path)
            .with_metric("rmse", 0.5)
            .with_metadata("author", "test");

        storage.save_version(&version)?;

        let loaded = storage.load_version("LSTM", "1.0.0")?;
        assert_eq!(loaded.version, "1.0.0");
        assert_eq!(loaded.metrics.get("rmse").unwrap(), &0.5);

        Ok(())
    }
}
