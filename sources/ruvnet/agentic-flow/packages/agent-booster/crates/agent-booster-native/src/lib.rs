#![deny(clippy::all)]

use agent_booster::{AgentBooster, Config, EditRequest, EditResult, Language};
use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi]
pub struct AgentBoosterNative {
    inner: AgentBooster,
}

#[napi(object)]
pub struct JsConfig {
    /// Minimum confidence threshold (0.0 - 1.0)
    pub confidence_threshold: Option<f64>,
    /// Maximum number of chunks to consider
    pub max_chunks: Option<u32>,
}

impl From<JsConfig> for Config {
    fn from(js_config: JsConfig) -> Self {
        let mut config = Config::default();
        if let Some(threshold) = js_config.confidence_threshold {
            config.confidence_threshold = threshold as f32;
        }
        if let Some(max_chunks) = js_config.max_chunks {
            config.max_chunks = max_chunks as usize;
        }
        config
    }
}

#[napi(object)]
pub struct JsEditRequest {
    /// Original source code
    pub original_code: String,
    /// Edit snippet to apply
    pub edit_snippet: String,
    /// Programming language ("javascript" or "typescript")
    pub language: String,
    /// Optional confidence threshold (0.0 - 1.0)
    pub confidence_threshold: Option<f64>,
}

impl TryFrom<JsEditRequest> for EditRequest {
    type Error = napi::Error;

    fn try_from(js_req: JsEditRequest) -> Result<Self> {
        let language = Language::from_str(&js_req.language)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;

        Ok(EditRequest {
            original_code: js_req.original_code,
            edit_snippet: js_req.edit_snippet,
            language,
            confidence_threshold: js_req.confidence_threshold.unwrap_or(0.5) as f32,
        })
    }
}

#[napi(object)]
pub struct JsEditResult {
    /// Merged code after applying edit
    pub merged_code: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Strategy used for merging
    pub strategy: String,
    /// Metadata about the edit
    pub metadata: JsEditMetadata,
}

#[napi(object)]
pub struct JsEditMetadata {
    /// Number of chunks found in original code
    pub chunks_found: u32,
    /// Best similarity score found
    pub best_similarity: f64,
    /// Whether syntax validation passed
    pub syntax_valid: bool,
    /// Processing time in milliseconds
    pub processing_time_ms: Option<f64>,
}

impl From<EditResult> for JsEditResult {
    fn from(result: EditResult) -> Self {
        JsEditResult {
            merged_code: result.merged_code,
            confidence: result.confidence as f64,
            strategy: result.strategy.as_str().to_string(),
            metadata: JsEditMetadata {
                chunks_found: result.metadata.chunks_found as u32,
                best_similarity: result.metadata.best_similarity as f64,
                syntax_valid: result.metadata.syntax_valid,
                processing_time_ms: result.metadata.processing_time_ms.map(|t| t as f64),
            },
        }
    }
}

#[napi]
impl AgentBoosterNative {
    /// Create a new AgentBooster instance
    #[napi(constructor)]
    pub fn new(config: Option<JsConfig>) -> Result<Self> {
        let config = config.map(Config::from).unwrap_or_default();
        let inner = AgentBooster::new(config)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;

        Ok(Self { inner })
    }

    /// Apply a single edit to code
    #[napi]
    pub fn apply_edit(&mut self, request: JsEditRequest) -> Result<JsEditResult> {
        let edit_request = EditRequest::try_from(request)?;

        let result = self
            .inner
            .apply_edit(edit_request)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;

        Ok(JsEditResult::from(result))
    }

    /// Apply multiple edits in batch
    #[napi]
    pub fn batch_apply(&mut self, requests: Vec<JsEditRequest>) -> Result<Vec<JsEditResult>> {
        let edit_requests: Result<Vec<EditRequest>> = requests
            .into_iter()
            .map(EditRequest::try_from)
            .collect();

        let edit_requests = edit_requests?;

        let results = self
            .inner
            .batch_apply(edit_requests)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;

        Ok(results.into_iter().map(JsEditResult::from).collect())
    }

    /// Get the current configuration
    #[napi]
    pub fn get_config(&self) -> JsConfig {
        let config = self.inner.config();
        JsConfig {
            confidence_threshold: Some(config.confidence_threshold as f64),
            max_chunks: Some(config.max_chunks as u32),
        }
    }

    /// Update the configuration
    #[napi]
    pub fn set_config(&mut self, config: JsConfig) {
        self.inner.set_config(Config::from(config));
    }
}

/// Quick helper function to apply a single edit
#[napi]
pub fn apply_edit_simple(
    original_code: String,
    edit_snippet: String,
    language: String,
) -> Result<JsEditResult> {
    let mut booster = AgentBoosterNative::new(None)?;
    booster.apply_edit(JsEditRequest {
        original_code,
        edit_snippet,
        language,
        confidence_threshold: None,
    })
}

/// Get version information
#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Get supported languages
#[napi]
pub fn supported_languages() -> Vec<String> {
    vec!["javascript".to_string(), "typescript".to_string()]
}
