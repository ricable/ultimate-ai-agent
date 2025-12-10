use wasm_bindgen::prelude::*;
use agent_booster::{
    models::{
        Language, MergeStrategy, CodeChunk, EditRequest, EditResult,
        EditMetadata, Config, AgentBoosterError, Result as AgentBoosterResult,
    },
};
use serde::{Deserialize, Serialize};

// Re-export types for WASM
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

    #[wasm_bindgen(js_namespace = console)]
    fn error(s: &str);
}

/// Initialize panic hook for better error messages in WASM
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// JavaScript-compatible Language enum
#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub enum WasmLanguage {
    JavaScript,
    TypeScript,
    Python,
    Rust,
    Go,
    Java,
    C,
    Cpp,
}

impl From<WasmLanguage> for Language {
    fn from(lang: WasmLanguage) -> Self {
        match lang {
            WasmLanguage::JavaScript => Language::JavaScript,
            WasmLanguage::TypeScript => Language::TypeScript,
            WasmLanguage::Python => Language::Python,
            WasmLanguage::Rust => Language::Rust,
            WasmLanguage::Go => Language::Go,
            WasmLanguage::Java => Language::Java,
            WasmLanguage::C => Language::C,
            WasmLanguage::Cpp => Language::Cpp,
        }
    }
}

impl From<Language> for WasmLanguage {
    fn from(lang: Language) -> Self {
        match lang {
            Language::JavaScript => WasmLanguage::JavaScript,
            Language::TypeScript => WasmLanguage::TypeScript,
            Language::Python => WasmLanguage::Python,
            Language::Rust => WasmLanguage::Rust,
            Language::Go => WasmLanguage::Go,
            Language::Java => WasmLanguage::Java,
            Language::C => WasmLanguage::C,
            Language::Cpp => WasmLanguage::Cpp,
        }
    }
}

/// JavaScript-compatible MergeStrategy enum
#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub enum WasmMergeStrategy {
    ExactReplace,
    FuzzyReplace,
    InsertAfter,
    InsertBefore,
    Append,
}

impl From<WasmMergeStrategy> for MergeStrategy {
    fn from(strategy: WasmMergeStrategy) -> Self {
        match strategy {
            WasmMergeStrategy::ExactReplace => MergeStrategy::ExactReplace,
            WasmMergeStrategy::FuzzyReplace => MergeStrategy::FuzzyReplace,
            WasmMergeStrategy::InsertAfter => MergeStrategy::InsertAfter,
            WasmMergeStrategy::InsertBefore => MergeStrategy::InsertBefore,
            WasmMergeStrategy::Append => MergeStrategy::Append,
        }
    }
}

impl From<MergeStrategy> for WasmMergeStrategy {
    fn from(strategy: MergeStrategy) -> Self {
        match strategy {
            MergeStrategy::ExactReplace => WasmMergeStrategy::ExactReplace,
            MergeStrategy::FuzzyReplace => WasmMergeStrategy::FuzzyReplace,
            MergeStrategy::InsertAfter => WasmMergeStrategy::InsertAfter,
            MergeStrategy::InsertBefore => WasmMergeStrategy::InsertBefore,
            MergeStrategy::Append => WasmMergeStrategy::Append,
        }
    }
}

/// JavaScript-compatible CodeChunk
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WasmCodeChunk {
    inner: CodeChunk,
}

#[wasm_bindgen]
impl WasmCodeChunk {
    #[wasm_bindgen(getter)]
    pub fn code(&self) -> String {
        self.inner.code.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn start_byte(&self) -> usize {
        self.inner.start_byte
    }

    #[wasm_bindgen(getter)]
    pub fn end_byte(&self) -> usize {
        self.inner.end_byte
    }

    #[wasm_bindgen(getter)]
    pub fn start_line(&self) -> usize {
        self.inner.start_line
    }

    #[wasm_bindgen(getter)]
    pub fn end_line(&self) -> usize {
        self.inner.end_line
    }

    #[wasm_bindgen(getter)]
    pub fn node_type(&self) -> String {
        self.inner.node_type.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn parent_type(&self) -> Option<String> {
        self.inner.parent_type.clone()
    }

    pub fn line_count(&self) -> usize {
        self.inner.line_count()
    }

    /// Convert to JSON string
    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.inner)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

/// JavaScript-compatible EditResult
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WasmEditResult {
    inner: EditResult,
}

#[wasm_bindgen]
impl WasmEditResult {
    #[wasm_bindgen(getter)]
    pub fn merged_code(&self) -> String {
        self.inner.merged_code.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn confidence(&self) -> f32 {
        self.inner.confidence
    }

    #[wasm_bindgen(getter)]
    pub fn strategy(&self) -> WasmMergeStrategy {
        self.inner.strategy.into()
    }

    #[wasm_bindgen(getter)]
    pub fn chunks_found(&self) -> usize {
        self.inner.metadata.chunks_found
    }

    #[wasm_bindgen(getter)]
    pub fn best_similarity(&self) -> f32 {
        self.inner.metadata.best_similarity
    }

    #[wasm_bindgen(getter)]
    pub fn syntax_valid(&self) -> bool {
        self.inner.metadata.syntax_valid
    }

    #[wasm_bindgen(getter)]
    pub fn processing_time_ms(&self) -> Option<u64> {
        self.inner.metadata.processing_time_ms
    }

    /// Convert to JSON string
    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.inner)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

/// JavaScript-compatible Config
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WasmConfig {
    inner: Config,
}

#[wasm_bindgen]
impl WasmConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: Config::default(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn confidence_threshold(&self) -> f32 {
        self.inner.confidence_threshold
    }

    #[wasm_bindgen(setter)]
    pub fn set_confidence_threshold(&mut self, threshold: f32) {
        self.inner.confidence_threshold = threshold;
    }

    #[wasm_bindgen(getter)]
    pub fn max_chunks(&self) -> usize {
        self.inner.max_chunks
    }

    #[wasm_bindgen(setter)]
    pub fn set_max_chunks(&mut self, max: usize) {
        self.inner.max_chunks = max;
    }

    /// Create from JSON string
    pub fn from_json(json: &str) -> Result<WasmConfig, JsValue> {
        let inner: Config = serde_json::from_str(json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Convert to JSON string
    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.inner)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl Default for WasmConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Main AgentBooster WASM interface
#[wasm_bindgen]
pub struct AgentBoosterWasm {
    booster: agent_booster::AgentBooster,
    config: Config,
}

#[wasm_bindgen]
impl AgentBoosterWasm {
    /// Create a new AgentBooster instance with default config
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<AgentBoosterWasm, JsValue> {
        let config = Config::default();
        let booster = agent_booster::AgentBooster::new(config.clone())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(Self {
            booster,
            config,
        })
    }

    /// Create with custom config
    pub fn with_config(config: WasmConfig) -> Result<AgentBoosterWasm, JsValue> {
        let booster = agent_booster::AgentBooster::new(config.inner.clone())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(Self {
            booster,
            config: config.inner,
        })
    }

    /// Parse a language string into WasmLanguage
    pub fn parse_language(lang: &str) -> Result<WasmLanguage, JsValue> {
        Language::from_str(lang)
            .map(|l| l.into())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Apply an edit to code
    ///
    /// Parameters:
    /// - original_code: The original source code
    /// - edit_snippet: The code snippet to apply
    /// - language: The programming language (use parse_language helper)
    ///
    /// Apply a code edit using tree-sitter and similarity matching
    pub fn apply_edit(
        &mut self,
        original_code: &str,
        edit_snippet: &str,
        language: WasmLanguage,
    ) -> Result<WasmEditResult, JsValue> {
        let request = EditRequest {
            original_code: original_code.to_string(),
            edit_snippet: edit_snippet.to_string(),
            language: language.into(),
            confidence_threshold: self.config.confidence_threshold,
        };

        let result = self.booster.apply_edit(request)
            .map_err(|e| JsValue::from_str(&format!("Edit failed: {}", e)))?;

        Ok(WasmEditResult {
            inner: result,
        })
    }

    /// Apply an edit from a JSON EditRequest
    pub fn apply_edit_json(&mut self, request_json: &str) -> Result<WasmEditResult, JsValue> {
        let request: EditRequest = serde_json::from_str(request_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let result = self.booster.apply_edit(request)
            .map_err(|e| JsValue::from_str(&format!("Edit failed: {}", e)))?;

        Ok(WasmEditResult {
            inner: result,
        })
    }

    /// Get current configuration
    pub fn get_config(&self) -> WasmConfig {
        WasmConfig {
            inner: self.config.clone(),
        }
    }

    /// Update configuration
    pub fn set_config(&mut self, config: WasmConfig) {
        self.config = config.inner;
    }

    /// Get library version
    pub fn version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
}

impl Default for AgentBoosterWasm {
    fn default() -> Self {
        Self::new().expect("Failed to create default AgentBoosterWasm")
    }
}

// Helper functions for JavaScript interop

/// Helper function to parse EditRequest from JSON
#[wasm_bindgen]
pub fn parse_edit_request(json: &str) -> Result<JsValue, JsValue> {
    let request: EditRequest = serde_json::from_str(json)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&request)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Helper function to create EditRequest JSON
#[wasm_bindgen]
pub fn create_edit_request(
    original_code: &str,
    edit_snippet: &str,
    language: WasmLanguage,
    confidence_threshold: Option<f32>,
) -> Result<String, JsValue> {
    let request = EditRequest {
        original_code: original_code.to_string(),
        edit_snippet: edit_snippet.to_string(),
        language: language.into(),
        confidence_threshold: confidence_threshold.unwrap_or(0.5),
    };

    serde_json::to_string(&request)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_language_conversion() {
        let js_lang = WasmLanguage::JavaScript;
        let rust_lang: Language = js_lang.into();
        let back_to_js: WasmLanguage = rust_lang.into();

        // Test round-trip conversion
        match back_to_js {
            WasmLanguage::JavaScript => (),
            _ => panic!("Language conversion failed"),
        }
    }

    #[wasm_bindgen_test]
    fn test_config_creation() {
        let config = WasmConfig::new();
        assert_eq!(config.confidence_threshold(), 0.5);
        assert_eq!(config.max_chunks(), 50);
    }

    #[wasm_bindgen_test]
    fn test_agent_booster_creation() {
        let booster = AgentBoosterWasm::new();
        let config = booster.get_config();
        assert_eq!(config.confidence_threshold(), 0.5);
    }

    #[wasm_bindgen_test]
    fn test_parse_language() {
        let result = AgentBoosterWasm::parse_language("javascript");
        assert!(result.is_ok());

        let result = AgentBoosterWasm::parse_language("typescript");
        assert!(result.is_ok());

        let result = AgentBoosterWasm::parse_language("python");
        assert!(result.is_ok());

        let result = AgentBoosterWasm::parse_language("rust");
        assert!(result.is_ok());

        let result = AgentBoosterWasm::parse_language("invalid");
        assert!(result.is_err());
    }

    #[wasm_bindgen_test]
    fn test_create_edit_request() {
        let result = create_edit_request(
            "const x = 1;",
            "const x = 2;",
            WasmLanguage::JavaScript,
            Some(0.7),
        );

        assert!(result.is_ok());
        let json = result.unwrap();
        assert!(json.contains("const x"));
    }
}
