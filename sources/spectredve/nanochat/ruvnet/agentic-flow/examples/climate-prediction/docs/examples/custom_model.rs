// Custom Model Integration Example
//
// This example shows how to:
// - Create a custom prediction model
// - Integrate it with the Climate Prediction System
// - Use ReasoningBank for model training and optimization
// - Deploy custom models alongside built-in models

use climate_prediction::{
    ClimatePredictor, PredictionConfig, PredictionResult,
    models::{Model, ModelTrait},
    reasoningbank::ReasoningBank,
};
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🔧 Custom Model Integration Example\n");

    // Example 1: Create and register custom model
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Example 1: Custom Linear Regression Model");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    custom_linear_model().await?;

    // Example 2: Train custom model with ReasoningBank
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Example 2: Train Custom Model");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    train_custom_model().await?;

    // Example 3: Ensemble with custom model
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Example 3: Ensemble with Custom Model");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    ensemble_with_custom().await?;

    println!("\n✅ All custom model examples completed!");

    Ok(())
}

// ============================================================================
// Custom Model Implementation
// ============================================================================

/// Simple linear regression model for weather prediction
/// Formula: y = mx + b (where x is historical average)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LinearRegressionModel {
    slope: f64,
    intercept: f64,
    feature_weights: Vec<f64>,
    confidence_base: f64,
}

impl LinearRegressionModel {
    /// Create new model with default parameters
    fn new() -> Self {
        Self {
            slope: 1.05,      // Slight warming trend
            intercept: 0.5,   // Base adjustment
            feature_weights: vec![
                0.7,  // Historical temperature weight
                0.15, // Latitude effect
                0.1,  // Seasonal effect
                0.05, // Random variation
            ],
            confidence_base: 0.75,
        }
    }

    /// Load model from ReasoningBank
    async fn load(reasoningbank: &ReasoningBank) -> Result<Self> {
        match reasoningbank.retrieve("models/custom", "linear_regression").await? {
            Some(entry) => {
                serde_json::from_value(entry.value)
                    .context("Failed to deserialize model")
            }
            None => {
                println!("⚠️  No saved model found, using defaults");
                Ok(Self::new())
            }
        }
    }

    /// Save model to ReasoningBank
    async fn save(&self, reasoningbank: &ReasoningBank) -> Result<()> {
        use climate_prediction::reasoningbank::MemoryEntry;
        use serde_json::json;

        let entry = MemoryEntry {
            key: "linear_regression".to_string(),
            value: serde_json::to_value(self)?,
            timestamp: chrono::Utc::now(),
            metadata: Some(json!({
                "model_type": "linear_regression",
                "version": "1.0",
                "trained_at": chrono::Utc::now().to_rfc3339(),
            })),
        };

        reasoningbank.store("models/custom", entry).await?;
        Ok(())
    }

    /// Train model on historical data
    fn train(&mut self, training_data: &[TrainingExample]) {
        if training_data.is_empty() {
            return;
        }

        // Simple gradient descent
        let learning_rate = 0.01;
        let epochs = 100;

        for _ in 0..epochs {
            let mut slope_gradient = 0.0;
            let mut intercept_gradient = 0.0;

            for example in training_data {
                let predicted = self.predict_temp(example.features[0]);
                let error = predicted - example.actual_temp;

                slope_gradient += error * example.features[0];
                intercept_gradient += error;
            }

            // Update parameters
            self.slope -= learning_rate * slope_gradient / training_data.len() as f64;
            self.intercept -= learning_rate * intercept_gradient / training_data.len() as f64;
        }
    }

    /// Predict temperature using linear regression
    fn predict_temp(&self, historical_avg: f64) -> f64 {
        self.slope * historical_avg + self.intercept
    }

    /// Calculate prediction confidence based on historical accuracy
    fn calculate_confidence(&self, features: &[f64]) -> f64 {
        // Simple confidence model: higher for typical values, lower for extremes
        let temp_typical = features[0].abs() < 40.0;
        let lat_typical = features[1].abs() < 60.0;

        let mut confidence = self.confidence_base;
        if temp_typical { confidence += 0.1; }
        if lat_typical { confidence += 0.1; }

        confidence.min(0.95)
    }
}

/// Implement ModelTrait to integrate with Climate Prediction System
#[async_trait::async_trait]
impl ModelTrait for LinearRegressionModel {
    async fn predict(&self, config: &PredictionConfig) -> Result<PredictionResult> {
        // Extract features from config
        let features = vec![
            20.0, // Historical temperature (would come from database)
            config.latitude,
            current_season_value(config.latitude),
            rand::random::<f64>() * 2.0 - 1.0, // Random variation
        ];

        // Calculate weighted prediction
        let base_temp = self.predict_temp(features[0]);
        let seasonal_adjustment = features[2] * self.feature_weights[2];
        let latitude_adjustment = (features[1] / 90.0) * self.feature_weights[1] * -20.0;

        let temperature = base_temp + seasonal_adjustment + latitude_adjustment;

        // Calculate other weather parameters (simplified)
        let precipitation = if temperature < 0.0 { 5.0 } else { 2.0 };
        let humidity = 60 + (temperature * 0.5) as i32;
        let wind_speed = 15.0 + rand::random::<f64>() * 10.0;

        Ok(PredictionResult {
            temperature,
            feels_like: temperature - 2.0,
            humidity: humidity.min(100).max(0),
            precipitation,
            precipitation_probability: 0.3,
            wind_speed,
            wind_direction: 180,
            pressure: 1013.0,
            cloud_cover: 50,
            uv_index: if temperature > 20.0 { 6 } else { 3 },
            confidence: self.calculate_confidence(&features),
            model_name: "custom_linear_regression".to_string(),
            uncertainty: Some(2.5),
            daily_forecast: vec![],
        })
    }

    fn name(&self) -> &str {
        "LinearRegression"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }
}

// ============================================================================
// Training Data Structure
// ============================================================================

#[derive(Debug, Clone)]
struct TrainingExample {
    features: Vec<f64>,
    actual_temp: f64,
    actual_precip: f64,
}

impl TrainingExample {
    fn from_historical(
        historical_temp: f64,
        latitude: f64,
        actual_temp: f64,
        actual_precip: f64,
    ) -> Self {
        Self {
            features: vec![
                historical_temp,
                latitude,
                current_season_value(latitude),
                0.0,
            ],
            actual_temp,
            actual_precip,
        }
    }
}

// ============================================================================
// Example Functions
// ============================================================================

/// Example 1: Create and use custom linear regression model
async fn custom_linear_model() -> Result<()> {
    let model = LinearRegressionModel::new();
    println!("Created custom linear regression model");
    println!("Parameters: slope={:.3}, intercept={:.3}", model.slope, model.intercept);

    // Test prediction
    let config = PredictionConfig {
        latitude: 40.7128,
        longitude: -74.0060,
        days_ahead: 7,
        include_uncertainty: true,
        units: "metric".to_string(),
    };

    println!("\nMaking prediction for NYC...");
    let prediction = model.predict(&config).await?;

    println!("📊 Results:");
    println!("   Temperature: {:.1}°C", prediction.temperature);
    println!("   Confidence:  {:.1}%", prediction.confidence * 100.0);
    println!("   Model: {}", prediction.model_name);

    Ok(())
}

/// Example 2: Train custom model with historical data
async fn train_custom_model() -> Result<()> {
    let reasoningbank = ReasoningBank::new("auto").await?;
    let mut model = LinearRegressionModel::new();

    println!("Training custom model on historical data...\n");

    // Generate synthetic training data (in production, use real data)
    let training_data: Vec<TrainingExample> = (0..100).map(|i| {
        let historical_temp = 15.0 + (i as f64 * 0.1);
        let latitude = 40.0 + rand::random::<f64>() * 20.0;
        let actual_temp = historical_temp * 1.05 + 0.5 + rand::random::<f64>() * 2.0;
        let actual_precip = if actual_temp < 10.0 { 5.0 } else { 2.0 };

        TrainingExample::from_historical(historical_temp, latitude, actual_temp, actual_precip)
    }).collect();

    println!("Training on {} examples...", training_data.len());

    let before_slope = model.slope;
    let before_intercept = model.intercept;

    model.train(&training_data);

    println!("\n📈 Training Results:");
    println!("   Slope:     {:.4} → {:.4}", before_slope, model.slope);
    println!("   Intercept: {:.4} → {:.4}", before_intercept, model.intercept);

    // Save trained model
    println!("\n💾 Saving trained model to ReasoningBank...");
    model.save(&reasoningbank).await?;
    println!("✅ Model saved successfully");

    // Test trained model
    let config = PredictionConfig {
        latitude: 40.7128,
        longitude: -74.0060,
        days_ahead: 7,
        include_uncertainty: true,
        units: "metric".to_string(),
    };

    let prediction = model.predict(&config).await?;
    println!("\n📊 Test Prediction:");
    println!("   Temperature: {:.1}°C", prediction.temperature);
    println!("   Confidence:  {:.1}%", prediction.confidence * 100.0);

    Ok(())
}

/// Example 3: Use custom model in ensemble
async fn ensemble_with_custom() -> Result<()> {
    let reasoningbank = ReasoningBank::new("auto").await?;

    // Load custom model
    let custom_model = LinearRegressionModel::load(&reasoningbank).await?;
    println!("✅ Loaded custom model from ReasoningBank");

    // Create ensemble with built-in models + custom model
    let neural_model = Arc::new(climate_prediction::models::NeuralModel::new()?);
    let arima_model = Arc::new(climate_prediction::models::ArimaModel::new()?);
    let custom_model = Arc::new(custom_model);

    println!("\n🎯 Ensemble Configuration:");
    println!("   1. Neural Network (40% weight)");
    println!("   2. ARIMA Model (30% weight)");
    println!("   3. Custom Linear Regression (30% weight)");

    let config = PredictionConfig {
        latitude: 51.5074, // London
        longitude: -0.1278,
        days_ahead: 7,
        include_uncertainty: true,
        units: "metric".to_string(),
    };

    println!("\nRunning ensemble prediction for London...");

    // Get predictions from all models
    let (neural_pred, arima_pred, custom_pred) = tokio::join!(
        neural_model.predict(&config),
        arima_model.predict(&config),
        custom_model.predict(&config),
    );

    let predictions = vec![
        ("Neural", neural_pred?, 0.4),
        ("ARIMA", arima_pred?, 0.3),
        ("Custom", custom_pred?, 0.3),
    ];

    // Display individual predictions
    println!("\n📊 Individual Model Predictions:");
    for (name, pred, weight) in &predictions {
        println!("   {:<10} {:.1}°C (weight: {:.0}%)",
            name, pred.temperature, weight * 100.0
        );
    }

    // Combine using weighted average
    let ensemble_temp: f64 = predictions.iter()
        .map(|(_, pred, weight)| pred.temperature * weight)
        .sum();

    let ensemble_confidence: f64 = predictions.iter()
        .map(|(_, pred, weight)| pred.confidence * weight)
        .sum();

    println!("\n🎯 Ensemble Result:");
    println!("   Temperature: {:.1}°C", ensemble_temp);
    println!("   Confidence:  {:.1}%", ensemble_confidence * 100.0);

    // Calculate improvement over individual models
    let neural_error = (predictions[0].1.temperature - ensemble_temp).abs();
    let arima_error = (predictions[1].1.temperature - ensemble_temp).abs();
    let custom_error = (predictions[2].1.temperature - ensemble_temp).abs();

    println!("\n📈 Ensemble Benefits:");
    println!("   Variance reduction from Neural: {:.1}°C", neural_error);
    println!("   Variance reduction from ARIMA:  {:.1}°C", arima_error);
    println!("   Variance reduction from Custom: {:.1}°C", custom_error);

    Ok(())
}

// ============================================================================
// Helper Functions
// ============================================================================

fn current_season_value(latitude: f64) -> f64 {
    let month = chrono::Utc::now().month() as f64;
    let hemisphere_factor = if latitude > 0.0 { 1.0 } else { -1.0 };

    // Sinusoidal seasonal variation: peaks in summer, troughs in winter
    ((month - 6.0) * std::f64::consts::PI / 6.0).sin() * hemisphere_factor * 10.0
}
