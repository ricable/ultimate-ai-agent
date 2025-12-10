//! Training data file format reader and writer

use crate::io::error::{IoError, IoResult};
use std::io::{BufRead, BufReader, Write};

// Import the mock types for now
use crate::mock_types::MockTrainingData;

/// Training data file format reader
pub struct TrainingDataReader {
    // Configuration options could go here
}

impl TrainingDataReader {
    /// Create a new training data reader
    pub fn new() -> Self {
        Self {}
    }

    /// Read training data from a FANN data format file
    pub fn read_data<R: std::io::Read>(&self, reader: &mut R) -> IoResult<MockTrainingData> {
        let mut buf_reader = BufReader::new(reader);
        let mut line = String::new();

        // Read header line
        buf_reader.read_line(&mut line)?;
        let header_parts: Vec<&str> = line.split_whitespace().collect();

        if header_parts.len() != 3 {
            return Err(IoError::InvalidFileFormat(
                "Header must contain exactly 3 numbers: num_data num_input num_output".to_string(),
            ));
        }

        let num_data: usize = header_parts[0]
            .parse()
            .map_err(|e| IoError::ParseError(format!("Invalid num_data: {e}")))?;
        let num_input: usize = header_parts[1]
            .parse()
            .map_err(|e| IoError::ParseError(format!("Invalid num_input: {e}")))?;
        let num_output: usize = header_parts[2]
            .parse()
            .map_err(|e| IoError::ParseError(format!("Invalid num_output: {e}")))?;

        let mut inputs = Vec::with_capacity(num_data);
        let mut outputs = Vec::with_capacity(num_data);

        for i in 0..num_data {
            // Read input line
            line.clear();
            buf_reader.read_line(&mut line)?;
            let input_values: Result<Vec<f32>, _> =
                line.split_whitespace().map(|s| s.parse()).collect();

            let input_values = input_values
                .map_err(|e| IoError::ParseError(format!("Invalid input at sample {i}: {e}")))?;

            if input_values.len() != num_input {
                return Err(IoError::InvalidTrainingData(format!(
                    "Expected {} inputs at sample {}, got {}",
                    num_input,
                    i,
                    input_values.len()
                )));
            }

            // Read output line
            line.clear();
            buf_reader.read_line(&mut line)?;
            let output_values: Result<Vec<f32>, _> =
                line.split_whitespace().map(|s| s.parse()).collect();

            let output_values = output_values
                .map_err(|e| IoError::ParseError(format!("Invalid output at sample {i}: {e}")))?;

            if output_values.len() != num_output {
                return Err(IoError::InvalidTrainingData(format!(
                    "Expected {} outputs at sample {}, got {}",
                    num_output,
                    i,
                    output_values.len()
                )));
            }

            inputs.push(input_values);
            outputs.push(output_values);
        }

        Ok(MockTrainingData {
            num_data,
            num_input,
            num_output,
            inputs,
            outputs,
        })
    }
}

impl Default for TrainingDataReader {
    fn default() -> Self {
        Self::new()
    }
}

/// Training data file format writer
pub struct TrainingDataWriter {
    // Configuration options could go here
}

impl TrainingDataWriter {
    /// Create a new training data writer
    pub fn new() -> Self {
        Self {}
    }

    /// Write training data to FANN data format
    pub fn write_data<W: Write>(&self, data: &MockTrainingData, writer: &mut W) -> IoResult<()> {
        // Write header
        writeln!(
            writer,
            "{} {} {}",
            data.num_data, data.num_input, data.num_output
        )?;

        // Write data samples
        for i in 0..data.num_data {
            // Write inputs
            for (j, &input) in data.inputs[i].iter().enumerate() {
                if j > 0 {
                    write!(writer, " ")?;
                }
                // Format numbers intelligently (integers as integers, floats as floats)
                if input.fract() == 0.0 && input.abs() < 1e6 {
                    write!(writer, "{}", input as i32)?;
                } else {
                    write!(writer, "{input}")?;
                }
            }
            writeln!(writer)?;

            // Write outputs
            for (j, &output) in data.outputs[i].iter().enumerate() {
                if j > 0 {
                    write!(writer, " ")?;
                }
                // Format numbers intelligently
                if output.fract() == 0.0 && output.abs() < 1e6 {
                    write!(writer, "{}", output as i32)?;
                } else {
                    write!(writer, "{output}")?;
                }
            }
            writeln!(writer)?;
        }

        Ok(())
    }
}

impl Default for TrainingDataWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Streaming reader for large training datasets
pub struct TrainingDataStreamReader {
    // Configuration options could go here
}

impl TrainingDataStreamReader {
    /// Create a new streaming reader
    pub fn new() -> Self {
        Self {}
    }

    /// Read training data with a callback for each sample
    pub fn read_stream<R, F>(&self, reader: &mut R, mut callback: F) -> IoResult<()>
    where
        R: BufRead,
        F: FnMut(&[f32], &[f32]) -> IoResult<()>,
    {
        let mut line = String::new();

        // Read header line
        reader.read_line(&mut line)?;
        let header_parts: Vec<&str> = line.split_whitespace().collect();

        if header_parts.len() != 3 {
            return Err(IoError::InvalidFileFormat(
                "Header must contain exactly 3 numbers: num_data num_input num_output".to_string(),
            ));
        }

        let num_data: usize = header_parts[0].parse()?;
        let num_input: usize = header_parts[1].parse()?;
        let num_output: usize = header_parts[2].parse()?;

        for i in 0..num_data {
            // Read input line
            line.clear();
            reader.read_line(&mut line)?;
            let input_values: Result<Vec<f32>, _> =
                line.split_whitespace().map(|s| s.parse()).collect();

            let input_values = input_values
                .map_err(|e| IoError::ParseError(format!("Invalid input at sample {i}: {e}")))?;

            if input_values.len() != num_input {
                return Err(IoError::InvalidTrainingData(format!(
                    "Expected {} inputs at sample {}, got {}",
                    num_input,
                    i,
                    input_values.len()
                )));
            }

            // Read output line
            line.clear();
            reader.read_line(&mut line)?;
            let output_values: Result<Vec<f32>, _> =
                line.split_whitespace().map(|s| s.parse()).collect();

            let output_values = output_values
                .map_err(|e| IoError::ParseError(format!("Invalid output at sample {i}: {e}")))?;

            if output_values.len() != num_output {
                return Err(IoError::InvalidTrainingData(format!(
                    "Expected {} outputs at sample {}, got {}",
                    num_output,
                    i,
                    output_values.len()
                )));
            }

            // Call callback with this sample
            callback(&input_values, &output_values)?;
        }

        Ok(())
    }
}

impl Default for TrainingDataStreamReader {
    fn default() -> Self {
        Self::new()
    }
}
