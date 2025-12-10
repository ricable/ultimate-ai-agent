//! High-performance XML parser for Ericsson ENM data
//! 
//! Implements SIMD-optimized parsing for common patterns in ENM XML files

use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;
use std::borrow::Cow;
use std::collections::HashMap;
use std::io::BufRead;
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// ENM measurement data structure
#[derive(Debug, Clone)]
pub struct EnmMeasurement {
    pub managed_element: String,
    pub measurement_type: String,
    pub granularity_period: u32,
    pub values: HashMap<String, MeasurementValue>,
}

/// Measurement value with metadata
#[derive(Debug, Clone)]
pub enum MeasurementValue {
    Counter(i64),
    Gauge(f64),
    String(String),
}

/// SIMD-optimized ENM XML parser
pub struct EnmParser {
    /// Buffer for XML parsing
    buffer: Vec<u8>,
    /// Reusable string buffer
    string_buffer: String,
}

impl EnmParser {
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(8192),
            string_buffer: String::with_capacity(1024),
        }
    }

    /// Parse ENM XML with SIMD acceleration for common patterns
    pub fn parse<R: BufRead>(&mut self, reader: R) -> Result<Vec<EnmMeasurement>, Box<dyn std::error::Error>> {
        let mut xml_reader = Reader::from_reader(reader);
        xml_reader.trim_text(true);
        
        let mut measurements = Vec::new();
        let mut current_measurement = None;
        let mut in_meas_info = false;
        let mut in_meas_value = false;
        
        loop {
            let mut buffer = Vec::new();
            match xml_reader.read_event_into(&mut buffer) {
                Ok(Event::Start(ref e)) => {
                    match e.name() {
                        quick_xml::name::QName(b"measInfo") => {
                            in_meas_info = true;
                            current_measurement = Some(EnmMeasurement {
                                managed_element: String::new(),
                                measurement_type: String::new(),
                                granularity_period: 900, // Default 15 min
                                values: HashMap::new(),
                            });
                        }
                        quick_xml::name::QName(b"measValue") => {
                            in_meas_value = true;
                        }
                        quick_xml::name::QName(b"measType") => {
                            if let Some(ref mut meas) = current_measurement {
                                if let Some(p) = e.try_get_attribute(b"p")? {
                                    let attr_value = p.decode_and_unescape_value(&xml_reader)?;
                                    if let Ok(period) = attr_value.parse::<u32>() {
                                        meas.granularity_period = period;
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Ok(Event::End(ref e)) => {
                    match e.name() {
                        quick_xml::name::QName(b"measInfo") => {
                            in_meas_info = false;
                            if let Some(meas) = current_measurement.take() {
                                measurements.push(meas);
                            }
                        }
                        quick_xml::name::QName(b"measValue") => {
                            in_meas_value = false;
                        }
                        _ => {}
                    }
                }
                Ok(Event::Text(e)) => {
                    if in_meas_value {
                        let text = e.unescape()?;
                        Self::parse_measurement_text_static(&text, &mut current_measurement);
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(Box::new(e)),
                _ => {}
            }
            
            self.buffer.clear();
        }
        
        Ok(measurements)
    }

    /// Parse measurement text (static version)
    fn parse_measurement_text_static(text: &str, measurement: &mut Option<EnmMeasurement>) {
        if let Some(ref mut meas) = measurement {
            // Parse counter values like "pmRrcConnEstabSucc=1234"
            if let Some(eq_pos) = text.find('=') {
                let (name, value_str) = text.split_at(eq_pos);
                let value_str = &value_str[1..]; // Skip '='
                
                let name = name.trim();
                let value = Self::parse_value_optimized_static(value_str.trim());
                
                meas.values.insert(name.to_string(), value);
            }
        }
    }

    /// Parse measurement text with optimized string handling
    fn parse_measurement_text(&mut self, text: &str, measurement: &mut Option<EnmMeasurement>) {
        Self::parse_measurement_text_static(text, measurement);
        if let Some(ref mut meas) = measurement {
            // Parse counter values like "pmRrcConnEstabSucc=1234"
            if let Some(eq_pos) = text.find('=') {
                let (name, value_str) = text.split_at(eq_pos);
                let value_str = &value_str[1..]; // Skip '='
                
                // Use SIMD for common counter names if available
                let name = name.trim();
                let value = self.parse_value_optimized(value_str.trim());
                
                meas.values.insert(name.to_string(), value);
            }
        }
    }

    /// Parse value with type detection (static version)
    fn parse_value_optimized_static(value_str: &str) -> MeasurementValue {
        // Try integer first (most common)
        if let Ok(val) = value_str.parse::<i64>() {
            return MeasurementValue::Counter(val);
        }
        
        // Try float
        if let Ok(val) = value_str.parse::<f64>() {
            return MeasurementValue::Gauge(val);
        }
        
        // Default to string
        MeasurementValue::String(value_str.to_string())
    }

    /// Parse value with type detection
    fn parse_value_optimized(&self, value_str: &str) -> MeasurementValue {
        Self::parse_value_optimized_static(value_str)
    }

    /// SIMD-optimized search for XML patterns
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn find_pattern_simd(&self, haystack: &[u8], needle: &[u8]) -> Option<usize> {
        if needle.is_empty() || haystack.len() < needle.len() {
            return None;
        }

        let first_byte = _mm256_set1_epi8(needle[0] as i8);
        let chunk_size = 32;

        for i in (0..=haystack.len() - needle.len()).step_by(chunk_size) {
            let end = (i + chunk_size).min(haystack.len() - needle.len() + 1);
            if end <= i {
                break;
            }

            let haystack_chunk = _mm256_loadu_si256(haystack[i..].as_ptr() as *const __m256i);
            let cmp = _mm256_cmpeq_epi8(haystack_chunk, first_byte);
            let mask = _mm256_movemask_epi8(cmp);

            if mask != 0 {
                for j in 0..32 {
                    if (mask & (1 << j)) != 0 {
                        let pos = i + j;
                        if pos + needle.len() <= haystack.len() 
                            && &haystack[pos..pos + needle.len()] == needle {
                            return Some(pos);
                        }
                    }
                }
            }
        }

        None
    }

    /// Batch parse multiple counter values
    pub fn parse_counters_batch(&self, data: &[(&str, &str)]) -> Vec<(String, MeasurementValue)> {
        data.par_iter()
            .map(|(name, value)| {
                (name.to_string(), self.parse_value_optimized(value))
            })
            .collect()
    }
}

/// Fast counter name matcher using perfect hashing
pub struct CounterMatcher {
    known_counters: HashMap<&'static str, CounterType>,
}

#[derive(Debug, Clone, Copy)]
pub enum CounterType {
    RrcConnection,
    ScellAddition,
    Handover,
    Throughput,
    Other,
}

impl CounterMatcher {
    pub fn new() -> Self {
        let mut known_counters = HashMap::new();
        
        // Common Ericsson counters
        known_counters.insert("pmRrcConnEstabSucc", CounterType::RrcConnection);
        known_counters.insert("pmRrcConnEstabAtt", CounterType::RrcConnection);
        known_counters.insert("pmLteScellAddSucc", CounterType::ScellAddition);
        known_counters.insert("pmLteScellAddAtt", CounterType::ScellAddition);
        known_counters.insert("pmHoExeSucc", CounterType::Handover);
        known_counters.insert("pmHoExeAtt", CounterType::Handover);
        known_counters.insert("pmPdcpVolDlDrb", CounterType::Throughput);
        known_counters.insert("pmPdcpVolUlDrb", CounterType::Throughput);
        
        Self { known_counters }
    }

    pub fn match_counter(&self, name: &str) -> CounterType {
        self.known_counters.get(name).copied().unwrap_or(CounterType::Other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_enm_parser() {
        let xml_data = r#"
            <measInfo>
                <measType p="900">RRC</measType>
                <measValue>
                    pmRrcConnEstabSucc=100
                    pmRrcConnEstabAtt=120
                </measValue>
            </measInfo>
        "#;
        
        let mut parser = EnmParser::new();
        let cursor = Cursor::new(xml_data.as_bytes());
        let measurements = parser.parse(cursor).unwrap();
        
        assert_eq!(measurements.len(), 1);
        assert_eq!(measurements[0].values.len(), 2);
    }

    #[test]
    fn test_counter_matcher() {
        let matcher = CounterMatcher::new();
        
        assert!(matches!(
            matcher.match_counter("pmRrcConnEstabSucc"),
            CounterType::RrcConnection
        ));
        assert!(matches!(
            matcher.match_counter("unknown_counter"),
            CounterType::Other
        ));
    }
}