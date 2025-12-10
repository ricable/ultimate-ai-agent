# Trait Design and Extensibility

## Overview
This document describes the trait-based architecture that enables extensibility in the climate prediction system.

## Core Traits

### 1. DataSource Trait
```rust
#[async_trait]
pub trait DataSource: Send + Sync {
    /// Fetch historical observations for a location and time range
    async fn fetch_observations(
        &self,
        location: Location,
        time_range: TimeRange,
        variables: Vec<ClimateVariable>,
    ) -> Result<Vec<Observation>>;

    /// Get the name of this data source
    fn name(&self) -> &str;

    /// Check if this data source is available
    async fn is_available(&self) -> bool;
}
```

**Implementations**:
- `OpenWeatherDataSource`
- `NoaaDataSource`
- `Era5DataSource`
- Custom implementations by users

**Example Implementation**:
```rust
pub struct OpenWeatherDataSource {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

#[async_trait]
impl DataSource for OpenWeatherDataSource {
    async fn fetch_observations(
        &self,
        location: Location,
        time_range: TimeRange,
        variables: Vec<ClimateVariable>,
    ) -> Result<Vec<Observation>> {
        let url = format!(
            "{}/data/2.5/weather?lat={}&lon={}&appid={}",
            self.base_url, location.latitude, location.longitude, self.api_key
        );

        let response = self.client.get(&url)
            .send()
            .await
            .map_err(|e| ClimateError::Network(e.to_string()))?;

        let data = response.json::<WeatherResponse>()
            .await
            .map_err(|e| ClimateError::DataSource(e.to_string()))?;

        Ok(self.parse_response(data, location, variables))
    }

    fn name(&self) -> &str {
        "openweather"
    }

    async fn is_available(&self) -> bool {
        self.client.get(&format!("{}/health", self.base_url))
            .send()
            .await
            .is_ok()
    }
}
```

### 2. PredictionModel Trait
```rust
#[async_trait]
pub trait PredictionModel: Send + Sync {
    /// Generate predictions for a single location
    async fn predict(
        &self,
        request: &PredictionRequest,
    ) -> Result<Vec<Prediction>>;

    /// Generate batch predictions for multiple locations
    async fn predict_batch(
        &self,
        request: &BatchPredictionRequest,
    ) -> Result<Vec<Prediction>>;

    /// Get model metadata
    fn metadata(&self) -> &ModelMetadata;

    /// Validate model inputs
    fn validate_input(&self, request: &PredictionRequest) -> Result<()>;
}
```

**Example Implementation**:
```rust
pub struct CandlePredictor {
    model: Arc<Model>,
    metadata: ModelMetadata,
    device: Device,
}

#[async_trait]
impl PredictionModel for CandlePredictor {
    async fn predict(
        &self,
        request: &PredictionRequest,
    ) -> Result<Vec<Prediction>> {
        self.validate_input(request)?;

        // Prepare input tensor
        let input = self.prepare_input(request)?;

        // Run inference (blocking, so spawn on Tokio thread pool)
        let model = Arc::clone(&self.model);
        let output = tokio::task::spawn_blocking(move || {
            model.forward(&input)
        }).await
            .map_err(|e| ClimateError::ModelInference(e.to_string()))??;

        // Parse output to predictions
        self.parse_output(output, request)
    }

    async fn predict_batch(
        &self,
        request: &BatchPredictionRequest,
    ) -> Result<Vec<Prediction>> {
        // Batch inference for efficiency
        let input_batch = self.prepare_batch_input(request)?;

        let model = Arc::clone(&self.model);
        let output = tokio::task::spawn_blocking(move || {
            model.forward(&input_batch)
        }).await
            .map_err(|e| ClimateError::ModelInference(e.to_string()))??;

        self.parse_batch_output(output, request)
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn validate_input(&self, request: &PredictionRequest) -> Result<()> {
        if !request.location.is_valid() {
            return Err(ClimateError::invalid_input("Invalid location"));
        }

        for var in &request.variables {
            if !self.metadata.variables.contains(var) {
                return Err(ClimateError::invalid_input(
                    format!("Model does not support variable: {:?}", var)
                ));
            }
        }

        Ok(())
    }
}
```

### 3. PhysicsConstraint Trait
```rust
pub trait PhysicsConstraint: Send + Sync {
    /// Apply physics constraint to predictions
    fn apply(&self, predictions: &mut [Prediction]) -> Result<()>;

    /// Validate that predictions satisfy physics laws
    fn validate(&self, predictions: &[Prediction]) -> Result<()>;

    /// Get constraint name
    fn name(&self) -> &str;
}
```

**Example Implementation**:
```rust
pub struct ThermodynamicConstraint {
    min_temp_kelvin: f64,
    max_temp_kelvin: f64,
}

impl PhysicsConstraint for ThermodynamicConstraint {
    fn apply(&self, predictions: &mut [Prediction]) -> Result<()> {
        for pred in predictions.iter_mut() {
            if pred.variable == ClimateVariable::Temperature {
                // Clamp to physical bounds
                pred.predicted_value = pred.predicted_value
                    .max(self.min_temp_kelvin)
                    .min(self.max_temp_kelvin);

                // Adjust confidence intervals
                pred.confidence_interval.lower =
                    pred.confidence_interval.lower.max(self.min_temp_kelvin);
                pred.confidence_interval.upper =
                    pred.confidence_interval.upper.min(self.max_temp_kelvin);
            }
        }
        Ok(())
    }

    fn validate(&self, predictions: &[Prediction]) -> Result<()> {
        for pred in predictions {
            if pred.variable == ClimateVariable::Temperature {
                if pred.predicted_value < self.min_temp_kelvin
                    || pred.predicted_value > self.max_temp_kelvin {
                    return Err(ClimateError::physics_constraint(
                        format!(
                            "Temperature {} K outside physical bounds [{}, {}]",
                            pred.predicted_value,
                            self.min_temp_kelvin,
                            self.max_temp_kelvin
                        )
                    ));
                }
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "thermodynamic-bounds"
    }
}
```

### 4. PostProcessor Trait
```rust
pub trait PostProcessor: Send + Sync {
    /// Process predictions after model inference
    fn process(&self, predictions: Vec<Prediction>) -> Result<Vec<Prediction>>;

    /// Get processor name
    fn name(&self) -> &str;
}
```

**Example Implementation**:
```rust
pub struct SpatialSmoothingProcessor {
    kernel_size: usize,
}

impl PostProcessor for SpatialSmoothingProcessor {
    fn process(&self, mut predictions: Vec<Prediction>) -> Result<Vec<Prediction>> {
        // Group predictions by location
        let mut location_groups = HashMap::new();
        for pred in predictions {
            location_groups.entry(pred.location)
                .or_insert_with(Vec::new)
                .push(pred);
        }

        // Apply spatial smoothing
        for (_, group) in location_groups.iter_mut() {
            self.smooth_group(group)?;
        }

        Ok(location_groups.into_iter()
            .flat_map(|(_, group)| group)
            .collect())
    }

    fn name(&self) -> &str {
        "spatial-smoothing"
    }
}
```

## Trait Object Collections

### Registry Pattern
```rust
pub struct DataSourceRegistry {
    sources: HashMap<String, Box<dyn DataSource>>,
}

impl DataSourceRegistry {
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
        }
    }

    pub fn register(&mut self, source: Box<dyn DataSource>) {
        let name = source.name().to_string();
        self.sources.insert(name, source);
    }

    pub fn get(&self, name: &str) -> Option<&dyn DataSource> {
        self.sources.get(name).map(|s| s.as_ref())
    }

    pub async fn fetch_from_all(
        &self,
        location: Location,
        time_range: TimeRange,
        variables: Vec<ClimateVariable>,
    ) -> Result<Vec<Observation>> {
        let mut all_observations = Vec::new();

        for source in self.sources.values() {
            if source.is_available().await {
                match source.fetch_observations(
                    location,
                    time_range.clone(),
                    variables.clone()
                ).await {
                    Ok(obs) => all_observations.extend(obs),
                    Err(e) => tracing::warn!(
                        "Failed to fetch from {}: {}",
                        source.name(),
                        e
                    ),
                }
            }
        }

        Ok(all_observations)
    }
}
```

## Extensibility Examples

### Adding a New Data Source
```rust
// User creates new data source
pub struct CustomDataSource {
    // Custom fields
}

#[async_trait]
impl DataSource for CustomDataSource {
    // Implement required methods
}

// Register with system
let mut registry = DataSourceRegistry::new();
registry.register(Box::new(CustomDataSource::new()));
```

### Composing Multiple Constraints
```rust
pub struct ConstraintChain {
    constraints: Vec<Box<dyn PhysicsConstraint>>,
}

impl PhysicsConstraint for ConstraintChain {
    fn apply(&self, predictions: &mut [Prediction]) -> Result<()> {
        for constraint in &self.constraints {
            constraint.apply(predictions)?;
        }
        Ok(())
    }

    fn validate(&self, predictions: &[Prediction]) -> Result<()> {
        for constraint in &self.constraints {
            constraint.validate(predictions)?;
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "constraint-chain"
    }
}
```

## Design Principles

### 1. Trait Object Safety
All traits are object-safe (can use `Box<dyn Trait>`):
- No generic methods
- No `Self: Sized` bounds
- No associated constants

### 2. Send + Sync Bounds
All traits require `Send + Sync` for thread safety:
```rust
pub trait DataSource: Send + Sync { }
```

### 3. async_trait for Async Methods
Use `async_trait` crate for traits with async methods:
```rust
#[async_trait]
pub trait DataSource {
    async fn fetch(...) -> Result<...>;
}
```

### 4. Error Propagation
All trait methods return `Result<T, ClimateError>` for consistent error handling.

## Testing Traits

### Mock Implementations
```rust
#[cfg(test)]
mod tests {
    use super::*;

    struct MockDataSource {
        observations: Vec<Observation>,
    }

    #[async_trait]
    impl DataSource for MockDataSource {
        async fn fetch_observations(...) -> Result<Vec<Observation>> {
            Ok(self.observations.clone())
        }

        fn name(&self) -> &str {
            "mock"
        }

        async fn is_available(&self) -> bool {
            true
        }
    }

    #[tokio::test]
    async fn test_data_source() {
        let source = MockDataSource {
            observations: vec![/* ... */],
        };

        let result = source.fetch_observations(...).await;
        assert!(result.is_ok());
    }
}
```
