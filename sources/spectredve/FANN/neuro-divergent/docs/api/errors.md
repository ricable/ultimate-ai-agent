# Error Handling

Neuro-Divergent provides comprehensive error handling through a hierarchical error system that helps developers identify, understand, and handle different types of failures that can occur during forecasting operations.

## Error Type Hierarchy

### NeuroDivergentError

The main error type for all library operations:

```rust
pub enum NeuroDivergentError {
    ConfigError(String),
    DataError(String),
    TrainingError(String),
    PredictionError(String),
    IoError(#[from] std::io::Error),
    SerializationError(String),
    NetworkError(String),
    MathError(String),
    TimeSeriesError(String),
    FannError(String),
    GpuError(String),      // Feature-gated
    AsyncError(String),    // Feature-gated
    Generic { message: String },
    Multiple { errors: Vec<NeuroDivergentError> },
}
```

### Result Type Alias

All library functions return `NeuroDivergentResult<T>`:

```rust
pub type NeuroDivergentResult<T> = Result<T, NeuroDivergentError>;
```

## Error Categories

### Configuration Errors

Errors related to invalid model or system configuration.

#### Common Causes

- Invalid parameter values
- Missing required configuration
- Incompatible parameter combinations
- Out-of-range values

#### Examples

```rust
// Invalid horizon
let config = LSTMConfig::builder()
    .horizon(0)  // Error: horizon must be > 0
    .build();

match config {
    Err(NeuroDivergentError::ConfigError(msg)) => {
        eprintln!("Configuration error: {}", msg);
    },
    Ok(c) => { /* use config */ }
}
```

#### Creating Configuration Errors

```rust
// Using the config_error! macro
return Err(config_error!("Horizon must be positive, got {}", horizon));

// Using the error constructor
return Err(NeuroDivergentError::config("Invalid learning rate"));
```

### Data Errors

Errors related to data validation, compatibility, and processing.

#### Common Causes

- Missing required columns
- Incompatible data schemas
- Empty datasets
- Invalid data types
- Data quality issues

#### Examples

```rust
// Schema validation error
let schema = TimeSeriesSchema::new("id", "date", "value");
let result = schema.validate_dataframe(&df);

match result {
    Err(NeuroDivergentError::DataError(msg)) => {
        eprintln!("Data validation failed: {}", msg);
        // Handle missing columns, wrong types, etc.
    },
    Ok(()) => { /* data is valid */ }
}
```

#### Creating Data Errors

```rust
// Using the data_error! macro
return Err(data_error!("Required column '{}' not found", column_name));

// Using the error constructor
return Err(NeuroDivergentError::data("Dataset is empty"));
```

### Training Errors

Errors that occur during model training.

#### Common Causes

- Convergence failures
- Numerical instability
- Insufficient data
- Hardware limitations
- Invalid training parameters

#### Examples

```rust
// Handle training failures
match model.fit(&dataset) {
    Ok(()) => println!("Training successful"),
    Err(NeuroDivergentError::TrainingError(msg)) => {
        eprintln!("Training failed: {}", msg);
        
        // Possible recovery strategies:
        // - Reduce learning rate
        // - Add regularization
        // - Check data quality
        // - Try different model architecture
    },
    Err(e) => eprintln!("Unexpected error: {}", e),
}
```

#### Creating Training Errors

```rust
// Using the training_error! macro
return Err(training_error!("Failed to converge after {} epochs", max_epochs));

// Using the error constructor
return Err(NeuroDivergentError::training("Gradient explosion detected"));
```

### Prediction Errors

Errors during forecast generation.

#### Common Causes

- Model not trained
- Incompatible input data
- Resource exhaustion
- Model state corruption

#### Examples

```rust
// Check if model is trained before prediction
if !model.is_trained() {
    return Err(NeuroDivergentError::prediction("Model has not been trained"));
}

match model.predict(&data) {
    Ok(forecasts) => { /* use forecasts */ },
    Err(NeuroDivergentError::PredictionError(msg)) => {
        eprintln!("Prediction failed: {}", msg);
        
        // Recovery strategies:
        // - Verify model is trained
        // - Check input data compatibility
        // - Validate data schema
    },
    Err(e) => eprintln!("Unexpected error: {}", e),
}
```

### I/O Errors

File system and serialization errors, automatically converted from `std::io::Error`.

#### Examples

```rust
// File operations
match TimeSeriesDataFrame::from_csv("data.csv") {
    Ok(df) => { /* use dataframe */ },
    Err(NeuroDivergentError::IoError(io_err)) => {
        eprintln!("File operation failed: {}", io_err);
        
        // Handle specific I/O errors
        match io_err.kind() {
            std::io::ErrorKind::NotFound => {
                eprintln!("File not found - check path");
            },
            std::io::ErrorKind::PermissionDenied => {
                eprintln!("Permission denied - check file permissions");
            },
            _ => eprintln!("I/O error: {}", io_err),
        }
    },
    Err(e) => eprintln!("Other error: {}", e),
}
```

### Math Errors

Numerical computation and mathematical operation errors.

#### Common Causes

- Array shape mismatches
- Numerical overflow/underflow
- Invalid mathematical operations
- Matrix operations on incompatible dimensions

#### Examples

```rust
// Handle numerical errors
match compute_forecast(&input_data) {
    Ok(result) => result,
    Err(NeuroDivergentError::MathError(msg)) => {
        eprintln!("Mathematical error: {}", msg);
        
        // Recovery strategies:
        // - Check input data shapes
        // - Verify numerical stability
        // - Use different precision (f32 vs f64)
        return None;
    },
    Err(e) => {
        eprintln!("Unexpected error: {}", e);
        return None;
    }
}
```

### Multiple Errors

Container for multiple related errors.

#### Examples

```rust
// Collect multiple validation errors
let mut errors = Vec::new();

if config.horizon == 0 {
    errors.push(NeuroDivergentError::config("Horizon must be positive"));
}

if config.input_size == 0 {
    errors.push(NeuroDivergentError::config("Input size must be positive"));
}

if config.learning_rate <= 0.0 {
    errors.push(NeuroDivergentError::config("Learning rate must be positive"));
}

if !errors.is_empty() {
    return Err(NeuroDivergentError::multiple(errors));
}
```

## Error Context and Chaining

### Adding Context

Use the `ErrorContext` trait to add contextual information:

```rust
use neuro_divergent::errors::ErrorContext;

// Add static context
let result = risky_operation()
    .context("Failed to process time series data");

// Add dynamic context
let result = risky_operation()
    .with_context(|| format!("Failed to process series '{}'", series_id));

// Chain multiple contexts
let result = load_data("file.csv")
    .context("Data loading failed")
    .with_context(|| format!("Processing model '{}'", model_name));
```

### Error Chaining Example

```rust
fn train_model(config: &ModelConfig, data: &TimeSeriesDataFrame) -> NeuroDivergentResult<()> {
    // Validate configuration
    config.validate()
        .context("Configuration validation failed")?;
    
    // Validate data
    validate_training_data(data)
        .with_context(|| format!("Data validation failed for {} series", data.n_series()))?;
    
    // Train model
    perform_training(config, data)
        .context("Model training failed")?;
    
    Ok(())
}
```

## Error Inspection and Handling

### Error Type Checking

```rust
fn handle_error(error: &NeuroDivergentError) {
    match error {
        NeuroDivergentError::ConfigError(_) => {
            // Configuration issues - fix configuration
            println!("Please check your model configuration");
        },
        NeuroDivergentError::DataError(_) => {
            // Data issues - fix data or preprocessing
            println!("Please check your input data");
        },
        NeuroDivergentError::TrainingError(_) => {
            // Training issues - adjust hyperparameters
            println!("Training failed - try different parameters");
        },
        NeuroDivergentError::PredictionError(_) => {
            // Prediction issues - check model state
            println!("Prediction failed - ensure model is trained");
        },
        _ => {
            println!("Unexpected error: {}", error);
        }
    }
}
```

### Error Category Methods

```rust
let error = NeuroDivergentError::config("Invalid parameter");

// Check error category
if error.is_config_error() {
    println!("Configuration error detected");
}

// Get error category as string
println!("Error category: {}", error.category());

// Pattern matching on category
match error.category() {
    "Configuration" => handle_config_error(&error),
    "Data" => handle_data_error(&error),
    "Training" => handle_training_error(&error),
    _ => handle_generic_error(&error),
}
```

## Error Recovery Strategies

### Graceful Degradation

```rust
fn robust_forecasting(models: Vec<Box<dyn BaseModel<f64>>>, data: &TimeSeriesDataFrame) -> NeuroDivergentResult<ForecastDataFrame> {
    let mut successful_forecasts = Vec::new();
    let mut errors = Vec::new();
    
    for (i, model) in models.iter().enumerate() {
        match model.predict(data) {
            Ok(forecast) => {
                successful_forecasts.push((format!("model_{}", i), forecast));
            },
            Err(e) => {
                eprintln!("Model {} failed: {}", i, e);
                errors.push(e);
            }
        }
    }
    
    if successful_forecasts.is_empty() {
        return Err(NeuroDivergentError::multiple(errors));
    }
    
    // Combine successful forecasts
    combine_forecasts(successful_forecasts)
}
```

### Retry Logic

```rust
fn train_with_retry(model: &mut dyn BaseModel<f64>, data: &TimeSeriesDataFrame, max_retries: usize) -> NeuroDivergentResult<()> {
    let mut last_error = None;
    
    for attempt in 0..max_retries {
        match model.fit(data) {
            Ok(()) => return Ok(()),
            Err(NeuroDivergentError::TrainingError(msg)) => {
                eprintln!("Training attempt {} failed: {}", attempt + 1, msg);
                
                // Reset model for retry
                model.reset()?;
                
                // Adjust parameters for retry
                if attempt < max_retries - 1 {
                    adjust_training_parameters(model, attempt)?;
                }
                
                last_error = Some(NeuroDivergentError::training(msg));
            },
            Err(e) => {
                // Non-recoverable error
                return Err(e);
            }
        }
    }
    
    Err(last_error.unwrap_or_else(|| 
        NeuroDivergentError::training("All retry attempts failed")))
}
```

### Fallback Models

```rust
fn forecasting_with_fallback(primary_model: &dyn BaseModel<f64>, fallback_model: &dyn BaseModel<f64>, data: &TimeSeriesDataFrame) -> NeuroDivergentResult<ForecastDataFrame> {
    // Try primary model
    match primary_model.predict(data) {
        Ok(forecast) => {
            println!("Primary model succeeded");
            Ok(forecast)
        },
        Err(primary_error) => {
            eprintln!("Primary model failed: {}", primary_error);
            
            // Fall back to secondary model
            match fallback_model.predict(data) {
                Ok(forecast) => {
                    println!("Fallback model succeeded");
                    Ok(forecast)
                },
                Err(fallback_error) => {
                    // Both models failed
                    Err(NeuroDivergentError::multiple(vec![
                        primary_error,
                        fallback_error,
                    ]))
                }
            }
        }
    }
}
```

## Error Reporting and Logging

### Structured Error Information

```rust
fn report_error(error: &NeuroDivergentError) {
    // Log error with structured information
    log::error!(
        "Forecasting error: {} - Category: {}", 
        error, 
        error.category()
    );
    
    // Add specific handling based on error type
    match error {
        NeuroDivergentError::TrainingError(msg) => {
            log::warn!("Consider adjusting hyperparameters: {}", msg);
        },
        NeuroDivergentError::DataError(msg) => {
            log::warn!("Check data preprocessing: {}", msg);
        },
        NeuroDivergentError::Multiple { errors } => {
            log::error!("Multiple errors occurred:");
            for (i, err) in errors.iter().enumerate() {
                log::error!("  {}: {}", i + 1, err);
            }
        },
        _ => {}
    }
}
```

### Error Metrics and Monitoring

```rust
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

struct ErrorMetrics {
    error_counts: HashMap<String, AtomicUsize>,
}

impl ErrorMetrics {
    fn record_error(&self, error: &NeuroDivergentError) {
        let category = error.category();
        self.error_counts
            .entry(category.to_string())
            .or_insert_with(|| AtomicUsize::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }
    
    fn get_error_summary(&self) -> HashMap<String, usize> {
        self.error_counts
            .iter()
            .map(|(k, v)| (k.clone(), v.load(Ordering::Relaxed)))
            .collect()
    }
}
```

## Error Prevention Best Practices

### Validation Early and Often

```rust
// Validate at construction time
impl LSTMConfig {
    pub fn new(horizon: usize, input_size: usize, hidden_size: usize) -> NeuroDivergentResult<Self> {
        if horizon == 0 {
            return Err(NeuroDivergentError::config("Horizon must be positive"));
        }
        if input_size == 0 {
            return Err(NeuroDivergentError::config("Input size must be positive"));
        }
        if hidden_size == 0 {
            return Err(NeuroDivergentError::config("Hidden size must be positive"));
        }
        
        Ok(Self {
            horizon,
            input_size,
            hidden_size,
            // ... other fields
        })
    }
}
```

### Defensive Programming

```rust
// Check preconditions
fn compute_forecast(model: &dyn BaseModel<f64>, data: &TimeSeriesDataFrame) -> NeuroDivergentResult<ForecastDataFrame> {
    // Defensive checks
    if !model.is_trained() {
        return Err(NeuroDivergentError::prediction("Model must be trained before prediction"));
    }
    
    if data.shape().0 == 0 {
        return Err(NeuroDivergentError::data("Input data is empty"));
    }
    
    // Validate data compatibility
    model.validate_data(&TimeSeriesDataset::from_dataframe(data)?)?;
    
    // Proceed with forecast
    model.predict(&TimeSeriesDataset::from_dataframe(data)?)
}
```

### Error Documentation

```rust
/// Trains the model on the provided dataset.
///
/// # Arguments
///
/// * `data` - Training dataset
///
/// # Returns
///
/// * `Ok(())` - If training succeeds
/// * `Err(NeuroDivergentError::DataError)` - If data is invalid or incompatible
/// * `Err(NeuroDivergentError::TrainingError)` - If training fails to converge
/// * `Err(NeuroDivergentError::ConfigError)` - If model configuration is invalid
///
/// # Examples
///
/// ```rust
/// let mut model = LSTM::new(config)?;
/// match model.fit(&dataset) {
///     Ok(()) => println!("Training successful"),
///     Err(e) => eprintln!("Training failed: {}", e),
/// }
/// ```
pub fn fit(&mut self, data: &TimeSeriesDataset<T>) -> NeuroDivergentResult<()> {
    // Implementation
}
```

## Integration with External Error Systems

### Converting to anyhow

```rust
use anyhow::Result;

fn external_function() -> Result<()> {
    let data = TimeSeriesDataFrame::from_csv("data.csv")
        .map_err(|e| anyhow::anyhow!("Failed to load data: {}", e))?;
    
    Ok(())
}
```

### Converting to Box<dyn Error>

```rust
fn external_function() -> Result<(), Box<dyn std::error::Error>> {
    let model = LSTM::builder()
        .horizon(7)
        .build()?;  // Automatic conversion
    
    Ok(())
}
```

The error handling system in Neuro-Divergent provides comprehensive error reporting, categorization, and recovery mechanisms to help developers build robust forecasting applications with clear error diagnostics and appropriate failure handling strategies.