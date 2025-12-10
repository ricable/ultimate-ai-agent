use std::collections::HashMap;
use std::str::FromStr;
use chrono::{DateTime, Utc, NaiveDateTime};
use serde::{Deserialize, Serialize};

/// Ericsson-specific log parser
#[derive(Debug)]
pub struct EricssonLogParser {
    amos_patterns: HashMap<String, AMOSPattern>,
    syslog_patterns: Vec<SyslogPattern>,
    field_extractors: HashMap<String, FieldExtractor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: String,
    pub level: LogLevel,
    pub log_type: String,
    pub content: String,
    pub structured_data: HashMap<String, String>,
    pub amos_command: Option<AMOSCommand>,
    pub raw_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Error,
    Warning,
    Info,
    Debug,
    Trace,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AMOSCommand {
    pub command: String,
    pub parameters: HashMap<String, String>,
    pub output: String,
    pub return_code: Option<i32>,
}

#[derive(Debug, Clone)]
struct AMOSPattern {
    command: String,
    pattern: String,
    parameter_extractors: Vec<ParameterExtractor>,
}

#[derive(Debug, Clone)]
struct SyslogPattern {
    pattern: String,
    priority_pos: Option<usize>,
    timestamp_pos: Option<usize>,
    hostname_pos: Option<usize>,
    process_pos: Option<usize>,
    message_pos: Option<usize>,
}

#[derive(Debug, Clone)]
struct FieldExtractor {
    pattern: String,
    field_name: String,
    extractor_type: ExtractorType,
}

#[derive(Debug, Clone)]
enum ExtractorType {
    Regex(String),
    KeyValue,
    Delimited(char),
    Fixed(usize, usize),
}

#[derive(Debug, Clone)]
struct ParameterExtractor {
    name: String,
    pattern: String,
    required: bool,
}

impl EricssonLogParser {
    pub fn new() -> Self {
        let mut parser = EricssonLogParser {
            amos_patterns: HashMap::new(),
            syslog_patterns: Vec::new(),
            field_extractors: HashMap::new(),
        };
        
        parser.initialize_patterns();
        parser
    }

    fn initialize_patterns(&mut self) {
        // Initialize AMOS command patterns
        self.add_amos_pattern("alt", vec![
            ParameterExtractor {
                name: "cell".to_string(),
                pattern: r"cell=(\d+)".to_string(),
                required: false,
            },
            ParameterExtractor {
                name: "state".to_string(),
                pattern: r"state=(\w+)".to_string(),
                required: false,
            },
        ]);

        self.add_amos_pattern("lget", vec![
            ParameterExtractor {
                name: "mo".to_string(),
                pattern: r"mo=([^\s]+)".to_string(),
                required: true,
            },
            ParameterExtractor {
                name: "attribute".to_string(),
                pattern: r"(\w+)=".to_string(),
                required: false,
            },
        ]);

        self.add_amos_pattern("cvc", vec![
            ParameterExtractor {
                name: "command".to_string(),
                pattern: r"cvc\s+(\w+)".to_string(),
                required: true,
            },
            ParameterExtractor {
                name: "parameters".to_string(),
                pattern: r"cvc\s+\w+\s+(.+)".to_string(),
                required: false,
            },
        ]);

        // Initialize syslog patterns
        self.add_syslog_pattern(
            r"^<(\d+)>(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\S+):\s+(.+)$",
            Some(1), Some(2), Some(3), Some(4), Some(5)
        );

        self.add_syslog_pattern(
            r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+(\w+)\s+(.+)$",
            None, Some(1), None, Some(2), Some(3)
        );

        // Initialize field extractors
        self.add_field_extractor("key_value", ExtractorType::KeyValue);
        self.add_field_extractor("comma_separated", ExtractorType::Delimited(','));
        self.add_field_extractor("space_separated", ExtractorType::Delimited(' '));
        self.add_field_extractor("ip_address", ExtractorType::Regex(
            r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b".to_string()
        ));
        self.add_field_extractor("cell_id", ExtractorType::Regex(
            r"cell[=:](\d+)".to_string()
        ));
        self.add_field_extractor("node_id", ExtractorType::Regex(
            r"node[=:](\w+)".to_string()
        ));
    }

    fn add_amos_pattern(&mut self, command: &str, extractors: Vec<ParameterExtractor>) {
        let pattern = AMOSPattern {
            command: command.to_string(),
            pattern: format!(r"(?i){}\s+(.+)", command),
            parameter_extractors: extractors,
        };
        self.amos_patterns.insert(command.to_string(), pattern);
    }

    fn add_syslog_pattern(
        &mut self,
        pattern: &str,
        priority_pos: Option<usize>,
        timestamp_pos: Option<usize>,
        hostname_pos: Option<usize>,
        process_pos: Option<usize>,
        message_pos: Option<usize>,
    ) {
        self.syslog_patterns.push(SyslogPattern {
            pattern: pattern.to_string(),
            priority_pos,
            timestamp_pos,
            hostname_pos,
            process_pos,
            message_pos,
        });
    }

    fn add_field_extractor(&mut self, name: &str, extractor_type: ExtractorType) {
        let pattern = match &extractor_type {
            ExtractorType::Regex(pattern) => pattern.clone(),
            ExtractorType::KeyValue => r"(\w+)=([^\s]+)".to_string(),
            ExtractorType::Delimited(delimiter) => format!(r"[^{}]+", delimiter),
            ExtractorType::Fixed(_, _) => String::new(),
        };

        self.field_extractors.insert(name.to_string(), FieldExtractor {
            pattern,
            field_name: name.to_string(),
            extractor_type,
        });
    }

    /// Parse a log entry
    pub fn parse(&self, log_line: &str) -> Result<LogEntry, ParseError> {
        let mut entry = LogEntry {
            timestamp: String::new(),
            level: LogLevel::Unknown,
            log_type: "UNKNOWN".to_string(),
            content: log_line.to_string(),
            structured_data: HashMap::new(),
            amos_command: None,
            raw_text: log_line.to_string(),
        };

        // Try to parse as AMOS command
        if let Some(amos_cmd) = self.parse_amos_command(log_line) {
            entry.amos_command = Some(amos_cmd);
            entry.log_type = "AMOS".to_string();
        }

        // Try to parse as syslog
        if let Some(syslog_data) = self.parse_syslog(log_line) {
            entry.timestamp = syslog_data.get("timestamp").unwrap_or(&String::new()).clone();
            entry.level = self.parse_log_level(syslog_data.get("level").unwrap_or(&String::new()));
            entry.log_type = "SYSLOG".to_string();
            entry.structured_data.extend(syslog_data);
        }

        // Extract structured data
        self.extract_structured_data(&mut entry);

        // Extract timestamp if not already found
        if entry.timestamp.is_empty() {
            entry.timestamp = self.extract_timestamp(log_line);
        }

        // Detect log level if not already found
        if matches!(entry.level, LogLevel::Unknown) {
            entry.level = self.detect_log_level(log_line);
        }

        Ok(entry)
    }

    fn parse_amos_command(&self, log_line: &str) -> Option<AMOSCommand> {
        for (command, pattern) in &self.amos_patterns {
            if let Ok(regex) = regex::Regex::new(&pattern.pattern) {
                if let Some(captures) = regex.captures(log_line) {
                    let mut parameters = HashMap::new();
                    let command_line = captures.get(1)?.as_str();

                    // Extract parameters
                    for extractor in &pattern.parameter_extractors {
                        if let Ok(param_regex) = regex::Regex::new(&extractor.pattern) {
                            if let Some(param_match) = param_regex.captures(command_line) {
                                if let Some(value) = param_match.get(1) {
                                    parameters.insert(extractor.name.clone(), value.as_str().to_string());
                                }
                            }
                        }
                    }

                    return Some(AMOSCommand {
                        command: command.clone(),
                        parameters,
                        output: command_line.to_string(),
                        return_code: None,
                    });
                }
            }
        }

        None
    }

    fn parse_syslog(&self, log_line: &str) -> Option<HashMap<String, String>> {
        for pattern in &self.syslog_patterns {
            if let Ok(regex) = regex::Regex::new(&pattern.pattern) {
                if let Some(captures) = regex.captures(log_line) {
                    let mut data = HashMap::new();

                    if let Some(pos) = pattern.priority_pos {
                        if let Some(priority) = captures.get(pos) {
                            data.insert("priority".to_string(), priority.as_str().to_string());
                        }
                    }

                    if let Some(pos) = pattern.timestamp_pos {
                        if let Some(timestamp) = captures.get(pos) {
                            data.insert("timestamp".to_string(), timestamp.as_str().to_string());
                        }
                    }

                    if let Some(pos) = pattern.hostname_pos {
                        if let Some(hostname) = captures.get(pos) {
                            data.insert("hostname".to_string(), hostname.as_str().to_string());
                        }
                    }

                    if let Some(pos) = pattern.process_pos {
                        if let Some(process) = captures.get(pos) {
                            data.insert("process".to_string(), process.as_str().to_string());
                            data.insert("level".to_string(), process.as_str().to_string());
                        }
                    }

                    if let Some(pos) = pattern.message_pos {
                        if let Some(message) = captures.get(pos) {
                            data.insert("message".to_string(), message.as_str().to_string());
                        }
                    }

                    return Some(data);
                }
            }
        }

        None
    }

    fn extract_structured_data(&self, entry: &mut LogEntry) {
        // Extract key-value pairs
        if let Some(kv_extractor) = self.field_extractors.get("key_value") {
            if let Ok(regex) = regex::Regex::new(&kv_extractor.pattern) {
                for capture in regex.captures_iter(&entry.content) {
                    if let (Some(key), Some(value)) = (capture.get(1), capture.get(2)) {
                        entry.structured_data.insert(
                            key.as_str().to_string(),
                            value.as_str().to_string(),
                        );
                    }
                }
            }
        }

        // Extract IP addresses
        if let Some(ip_extractor) = self.field_extractors.get("ip_address") {
            if let Ok(regex) = regex::Regex::new(&ip_extractor.pattern) {
                let mut ip_addresses = Vec::new();
                for capture in regex.captures_iter(&entry.content) {
                    if let Some(ip) = capture.get(0) {
                        ip_addresses.push(ip.as_str().to_string());
                    }
                }
                if !ip_addresses.is_empty() {
                    entry.structured_data.insert("ip_addresses".to_string(), ip_addresses.join(","));
                }
            }
        }

        // Extract cell IDs
        if let Some(cell_extractor) = self.field_extractors.get("cell_id") {
            if let Ok(regex) = regex::Regex::new(&cell_extractor.pattern) {
                if let Some(capture) = regex.captures(&entry.content) {
                    if let Some(cell_id) = capture.get(1) {
                        entry.structured_data.insert("cell_id".to_string(), cell_id.as_str().to_string());
                    }
                }
            }
        }

        // Extract node IDs
        if let Some(node_extractor) = self.field_extractors.get("node_id") {
            if let Ok(regex) = regex::Regex::new(&node_extractor.pattern) {
                if let Some(capture) = regex.captures(&entry.content) {
                    if let Some(node_id) = capture.get(1) {
                        entry.structured_data.insert("node_id".to_string(), node_id.as_str().to_string());
                    }
                }
            }
        }
    }

    fn extract_timestamp(&self, log_line: &str) -> String {
        // Try different timestamp formats
        let timestamp_patterns = vec![
            r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}",
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
            r"\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}",
            r"\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}",
        ];

        for pattern in timestamp_patterns {
            if let Ok(regex) = regex::Regex::new(pattern) {
                if let Some(timestamp_match) = regex.find(log_line) {
                    return timestamp_match.as_str().to_string();
                }
            }
        }

        // Return current timestamp if none found
        Utc::now().format("%Y-%m-%d %H:%M:%S").to_string()
    }

    fn parse_log_level(&self, level_str: &str) -> LogLevel {
        match level_str.to_uppercase().as_str() {
            "ERROR" | "ERR" | "FATAL" | "CRIT" => LogLevel::Error,
            "WARN" | "WARNING" => LogLevel::Warning,
            "INFO" | "INFORMATION" => LogLevel::Info,
            "DEBUG" | "DBG" => LogLevel::Debug,
            "TRACE" | "TRC" => LogLevel::Trace,
            _ => LogLevel::Unknown,
        }
    }

    fn detect_log_level(&self, log_line: &str) -> LogLevel {
        let upper_line = log_line.to_uppercase();
        
        if upper_line.contains("ERROR") || upper_line.contains("FAIL") || upper_line.contains("FATAL") {
            LogLevel::Error
        } else if upper_line.contains("WARN") || upper_line.contains("ALERT") {
            LogLevel::Warning
        } else if upper_line.contains("INFO") {
            LogLevel::Info
        } else if upper_line.contains("DEBUG") {
            LogLevel::Debug
        } else if upper_line.contains("TRACE") {
            LogLevel::Trace
        } else {
            LogLevel::Unknown
        }
    }

    /// Parse semi-structured text with custom delimiters
    pub fn parse_semi_structured(&self, text: &str, delimiter: char) -> HashMap<String, String> {
        let mut data = HashMap::new();
        
        for part in text.split(delimiter) {
            let part = part.trim();
            if part.contains('=') {
                let mut split = part.splitn(2, '=');
                if let (Some(key), Some(value)) = (split.next(), split.next()) {
                    data.insert(key.trim().to_string(), value.trim().to_string());
                }
            }
        }
        
        data
    }

    /// Extract numerical values from log content
    pub fn extract_numerical_values(&self, content: &str) -> HashMap<String, f64> {
        let mut values = HashMap::new();
        
        // Pattern for extracting numerical values with labels
        let patterns = vec![
            (r"cpu[=:](\d+(?:\.\d+)?)", "cpu_usage"),
            (r"memory[=:](\d+(?:\.\d+)?)", "memory_usage"),
            (r"throughput[=:](\d+(?:\.\d+)?)", "throughput"),
            (r"latency[=:](\d+(?:\.\d+)?)", "latency"),
            (r"errors[=:](\d+)", "error_count"),
            (r"warnings[=:](\d+)", "warning_count"),
        ];

        for (pattern, name) in patterns {
            if let Ok(regex) = regex::Regex::new(pattern) {
                if let Some(capture) = regex.captures(content) {
                    if let Some(value_str) = capture.get(1) {
                        if let Ok(value) = value_str.as_str().parse::<f64>() {
                            values.insert(name.to_string(), value);
                        }
                    }
                }
            }
        }
        
        values
    }

    /// Validate parsed log entry
    pub fn validate_entry(&self, entry: &LogEntry) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        // Check timestamp format
        if entry.timestamp.is_empty() {
            errors.push(ValidationError::MissingTimestamp);
        }

        // Check required fields for AMOS commands
        if let Some(amos_cmd) = &entry.amos_command {
            if let Some(pattern) = self.amos_patterns.get(&amos_cmd.command) {
                for extractor in &pattern.parameter_extractors {
                    if extractor.required && !amos_cmd.parameters.contains_key(&extractor.name) {
                        errors.push(ValidationError::MissingRequiredParameter(extractor.name.clone()));
                    }
                }
            }
        }

        // Check content length
        if entry.content.len() > 10000 {
            errors.push(ValidationError::ContentTooLong);
        }

        errors
    }
}

#[derive(Debug, Clone)]
pub enum ParseError {
    InvalidFormat,
    MissingTimestamp,
    InvalidLogLevel,
    RegexError(String),
}

#[derive(Debug, Clone)]
pub enum ValidationError {
    MissingTimestamp,
    MissingRequiredParameter(String),
    ContentTooLong,
    InvalidFormat,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::InvalidFormat => write!(f, "Invalid log format"),
            ParseError::MissingTimestamp => write!(f, "Missing timestamp"),
            ParseError::InvalidLogLevel => write!(f, "Invalid log level"),
            ParseError::RegexError(msg) => write!(f, "Regex error: {}", msg),
        }
    }
}

impl std::error::Error for ParseError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amos_command_parsing() {
        let parser = EricssonLogParser::new();
        
        let log = "2024-01-04 10:15:23 AMOS alt cell=12345 state=active";
        let entry = parser.parse(log).unwrap();
        
        assert_eq!(entry.log_type, "AMOS");
        assert!(entry.amos_command.is_some());
        
        let amos_cmd = entry.amos_command.unwrap();
        assert_eq!(amos_cmd.command, "alt");
        assert_eq!(amos_cmd.parameters.get("cell"), Some(&"12345".to_string()));
        assert_eq!(amos_cmd.parameters.get("state"), Some(&"active".to_string()));
    }

    #[test]
    fn test_syslog_parsing() {
        let parser = EricssonLogParser::new();
        
        let log = "2024-01-04 10:15:23 ERROR Connection timeout on node RBS_01";
        let entry = parser.parse(log).unwrap();
        
        assert_eq!(entry.log_type, "SYSLOG");
        assert!(matches!(entry.level, LogLevel::Error));
        assert!(!entry.timestamp.is_empty());
    }

    #[test]
    fn test_structured_data_extraction() {
        let parser = EricssonLogParser::new();
        
        let log = "INFO: lget mo=RncFunction=1,UtranCell=12345 cell=67890 node=RBS_01";
        let entry = parser.parse(log).unwrap();
        
        assert!(entry.structured_data.contains_key("cell"));
        assert!(entry.structured_data.contains_key("node"));
        assert_eq!(entry.structured_data.get("cell"), Some(&"67890".to_string()));
        assert_eq!(entry.structured_data.get("node"), Some(&"RBS_01".to_string()));
    }

    #[test]
    fn test_numerical_value_extraction() {
        let parser = EricssonLogParser::new();
        
        let content = "Performance: cpu=85.2 memory=1024.5 throughput=100 errors=5";
        let values = parser.extract_numerical_values(content);
        
        assert_eq!(values.get("cpu_usage"), Some(&85.2));
        assert_eq!(values.get("memory_usage"), Some(&1024.5));
        assert_eq!(values.get("throughput"), Some(&100.0));
        assert_eq!(values.get("error_count"), Some(&5.0));
    }

    #[test]
    fn test_log_level_detection() {
        let parser = EricssonLogParser::new();
        
        assert!(matches!(parser.detect_log_level("ERROR: Something went wrong"), LogLevel::Error));
        assert!(matches!(parser.detect_log_level("WARNING: Low memory"), LogLevel::Warning));
        assert!(matches!(parser.detect_log_level("INFO: System started"), LogLevel::Info));
        assert!(matches!(parser.detect_log_level("DEBUG: Variable value"), LogLevel::Debug));
    }
}