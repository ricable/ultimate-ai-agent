//! Prompt compression algorithms for token efficiency

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::{Prompt, Result, GenAIError};

/// Compression statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    pub original_tokens: usize,
    pub compressed_tokens: usize,
    pub compression_ratio: f32,
    pub technique: String,
}

/// Trait for prompt compression techniques
pub trait PromptCompressor: Send + Sync {
    /// Compress a prompt to reduce token usage
    fn compress(&self, prompt: &Prompt) -> Result<(Prompt, CompressionStats)>;
    
    /// Get the name of the compression technique
    fn technique_name(&self) -> &str;
}

/// Simple text compression using common substitutions
pub struct SubstitutionCompressor {
    substitutions: HashMap<String, String>,
}

impl SubstitutionCompressor {
    pub fn new() -> Self {
        let mut substitutions = HashMap::new();
        
        // Common word substitutions
        substitutions.insert("you are".to_string(), "u r".to_string());
        substitutions.insert("please".to_string(), "pls".to_string());
        substitutions.insert("because".to_string(), "bc".to_string());
        substitutions.insert("information".to_string(), "info".to_string());
        substitutions.insert("configuration".to_string(), "config".to_string());
        substitutions.insert("performance".to_string(), "perf".to_string());
        substitutions.insert("management".to_string(), "mgmt".to_string());
        substitutions.insert("network".to_string(), "net".to_string());
        substitutions.insert("radio access network".to_string(), "RAN".to_string());
        substitutions.insert("base station".to_string(), "BS".to_string());
        substitutions.insert("key performance indicator".to_string(), "KPI".to_string());
        
        Self { substitutions }
    }
    
    fn compress_text(&self, text: &str) -> String {
        let mut result = text.to_string();
        
        for (original, replacement) in &self.substitutions {
            result = result.replace(original, replacement);
            result = result.replace(&original.to_uppercase(), &replacement.to_uppercase());
        }
        
        // Remove extra whitespace
        result = result.split_whitespace().collect::<Vec<_>>().join(" ");
        
        result
    }
}

impl PromptCompressor for SubstitutionCompressor {
    fn compress(&self, prompt: &Prompt) -> Result<(Prompt, CompressionStats)> {
        let original_tokens = self.estimate_tokens(&prompt.user);
        
        let compressed_user = self.compress_text(&prompt.user);
        let compressed_system = prompt.system.as_ref()
            .map(|s| self.compress_text(s));
        
        let compressed_prompt = Prompt {
            system: compressed_system,
            user: compressed_user,
            context: prompt.context.clone(),
            max_tokens: prompt.max_tokens,
            temperature: prompt.temperature,
        };
        
        let compressed_tokens = self.estimate_tokens(&compressed_prompt.user);
        let compression_ratio = if original_tokens > 0 {
            compressed_tokens as f32 / original_tokens as f32
        } else {
            1.0
        };
        
        let stats = CompressionStats {
            original_tokens,
            compressed_tokens,
            compression_ratio,
            technique: self.technique_name().to_string(),
        };
        
        Ok((compressed_prompt, stats))
    }
    
    fn technique_name(&self) -> &str {
        "substitution"
    }
}

impl SubstitutionCompressor {
    fn estimate_tokens(&self, text: &str) -> usize {
        // Simple approximation: ~4 chars per token
        text.len() / 4
    }
}

/// Template-based compression for repeated patterns
pub struct TemplateCompressor {
    templates: HashMap<String, String>,
}

impl TemplateCompressor {
    pub fn new() -> Self {
        let mut templates = HashMap::new();
        
        // Common RAN operation templates
        templates.insert(
            "analyze_kpi".to_string(),
            "Analyze KPI: {kpi} for cell {cell} showing {issue}".to_string()
        );
        templates.insert(
            "fault_diagnosis".to_string(),
            "Diagnose: {fault} in {component} at {time}".to_string()
        );
        templates.insert(
            "optimization_request".to_string(),
            "Optimize {parameter} for {objective} in {area}".to_string()
        );
        
        Self { templates }
    }
    
    /// Try to match text against templates
    fn match_template(&self, text: &str) -> Option<(String, HashMap<String, String>)> {
        for (template_name, template) in &self.templates {
            if let Some(params) = self.extract_parameters(text, template) {
                return Some((template_name.clone(), params));
            }
        }
        None
    }
    
    /// Extract parameters from text using template
    fn extract_parameters(&self, text: &str, template: &str) -> Option<HashMap<String, String>> {
        // Simple parameter extraction (could be improved with regex)
        let template_parts: Vec<&str> = template.split('{').collect();
        if template_parts.len() < 2 {
            return None;
        }
        
        // For now, just return empty params if text contains template keywords
        let keywords: Vec<&str> = template_parts[0].split_whitespace().collect();
        if keywords.iter().any(|&keyword| text.contains(keyword)) {
            return Some(HashMap::new());
        }
        
        None
    }
}

impl PromptCompressor for TemplateCompressor {
    fn compress(&self, prompt: &Prompt) -> Result<(Prompt, CompressionStats)> {
        let original_tokens = self.estimate_tokens(&prompt.user);
        
        let compressed_user = if let Some((template_name, _params)) = self.match_template(&prompt.user) {
            format!("TEMPLATE:{}", template_name)
        } else {
            prompt.user.clone()
        };
        
        let compressed_prompt = Prompt {
            system: prompt.system.clone(),
            user: compressed_user,
            context: prompt.context.clone(),
            max_tokens: prompt.max_tokens,
            temperature: prompt.temperature,
        };
        
        let compressed_tokens = self.estimate_tokens(&compressed_prompt.user);
        let compression_ratio = if original_tokens > 0 {
            compressed_tokens as f32 / original_tokens as f32
        } else {
            1.0
        };
        
        let stats = CompressionStats {
            original_tokens,
            compressed_tokens,
            compression_ratio,
            technique: self.technique_name().to_string(),
        };
        
        Ok((compressed_prompt, stats))
    }
    
    fn technique_name(&self) -> &str {
        "template"
    }
}

impl TemplateCompressor {
    fn estimate_tokens(&self, text: &str) -> usize {
        text.len() / 4
    }
}

/// Composite compressor that applies multiple techniques
pub struct CompositeCompressor {
    compressors: Vec<Box<dyn PromptCompressor>>,
}

impl CompositeCompressor {
    pub fn new() -> Self {
        let compressors: Vec<Box<dyn PromptCompressor>> = vec![
            Box::new(SubstitutionCompressor::new()),
            Box::new(TemplateCompressor::new()),
        ];
        
        Self { compressors }
    }
    
    pub fn with_compressors(compressors: Vec<Box<dyn PromptCompressor>>) -> Self {
        Self { compressors }
    }
}

impl PromptCompressor for CompositeCompressor {
    fn compress(&self, prompt: &Prompt) -> Result<(Prompt, CompressionStats)> {
        let mut current_prompt = prompt.clone();
        let mut total_stats = CompressionStats {
            original_tokens: self.estimate_tokens(&prompt.user),
            compressed_tokens: 0,
            compression_ratio: 1.0,
            technique: "composite".to_string(),
        };
        
        for compressor in &self.compressors {
            let (compressed_prompt, stats) = compressor.compress(&current_prompt)?;
            current_prompt = compressed_prompt;
            
            // Update total compression ratio
            total_stats.compression_ratio *= stats.compression_ratio;
        }
        
        total_stats.compressed_tokens = self.estimate_tokens(&current_prompt.user);
        
        Ok((current_prompt, total_stats))
    }
    
    fn technique_name(&self) -> &str {
        "composite"
    }
}

impl CompositeCompressor {
    fn estimate_tokens(&self, text: &str) -> usize {
        text.len() / 4
    }
}

/// Compression service that manages different compression strategies
pub struct CompressionService {
    compressors: HashMap<String, Box<dyn PromptCompressor>>,
    default_compressor: String,
}

impl CompressionService {
    pub fn new() -> Self {
        let mut compressors: HashMap<String, Box<dyn PromptCompressor>> = HashMap::new();
        
        compressors.insert(
            "substitution".to_string(),
            Box::new(SubstitutionCompressor::new())
        );
        compressors.insert(
            "template".to_string(),
            Box::new(TemplateCompressor::new())
        );
        compressors.insert(
            "composite".to_string(),
            Box::new(CompositeCompressor::new())
        );
        
        Self {
            compressors,
            default_compressor: "composite".to_string(),
        }
    }
    
    /// Compress a prompt using the specified technique
    pub fn compress(&self, prompt: &Prompt, technique: Option<&str>) -> Result<(Prompt, CompressionStats)> {
        let technique = technique.unwrap_or(&self.default_compressor);
        
        let compressor = self.compressors.get(technique)
            .ok_or_else(|| GenAIError::Config(format!("Unknown compression technique: {}", technique)))?;
        
        compressor.compress(prompt)
    }
    
    /// Get available compression techniques
    pub fn available_techniques(&self) -> Vec<String> {
        self.compressors.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_substitution_compressor() {
        let compressor = SubstitutionCompressor::new();
        
        let prompt = Prompt {
            system: Some("You are a helpful assistant".to_string()),
            user: "Please provide information about network performance".to_string(),
            context: None,
            max_tokens: None,
            temperature: None,
        };
        
        let (compressed, stats) = compressor.compress(&prompt).unwrap();
        
        assert!(compressed.user.contains("pls"));
        assert!(compressed.user.contains("info"));
        assert!(compressed.user.contains("net"));
        assert!(compressed.user.contains("perf"));
        assert!(compressed.system.as_ref().unwrap().contains("u r"));
        assert!(stats.compression_ratio < 1.0);
    }
    
    #[test]
    fn test_template_compressor() {
        let compressor = TemplateCompressor::new();
        
        let prompt = Prompt {
            system: None,
            user: "Analyze KPI pmRrcConnEstabSucc for cell ABC123".to_string(),
            context: None,
            max_tokens: None,
            temperature: None,
        };
        
        let (compressed, stats) = compressor.compress(&prompt).unwrap();
        
        // Should match template and compress
        assert!(compressed.user.starts_with("TEMPLATE:"));
        assert!(stats.compression_ratio < 1.0);
    }
    
    #[test]
    fn test_compression_service() {
        let service = CompressionService::new();
        
        let prompt = Prompt {
            system: None,
            user: "Please analyze network performance information".to_string(),
            context: None,
            max_tokens: None,
            temperature: None,
        };
        
        let (compressed, stats) = service.compress(&prompt, None).unwrap();
        
        assert!(stats.compression_ratio <= 1.0);
        assert_eq!(stats.technique, "composite");
        
        let techniques = service.available_techniques();
        assert!(techniques.contains(&"substitution".to_string()));
        assert!(techniques.contains(&"template".to_string()));
        assert!(techniques.contains(&"composite".to_string()));
    }
}