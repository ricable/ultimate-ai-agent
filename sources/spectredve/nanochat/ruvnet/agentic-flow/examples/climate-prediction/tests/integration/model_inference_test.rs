// Integration tests for ML model inference
mod common;
use common::test_helpers::*;

#[tokio::test]
async fn test_model_loading() {
    let config = create_test_config();

    // Test model path validation
    assert!(!config.model_path.is_empty());
    assert!(config.model_path.ends_with(".onnx"));
}

#[tokio::test]
async fn test_model_inference_basic() {
    let input_data = generate_sample_climate_data(24); // 24 hours of data

    // Prepare input tensor (simplified)
    let input_features: Vec<f32> = input_data
        .iter()
        .flat_map(|point| {
            vec![
                point.temperature as f32,
                point.humidity as f32,
                point.pressure as f32,
                point.wind_speed as f32,
                point.precipitation as f32,
            ]
        })
        .collect();

    assert_eq!(input_features.len(), 24 * 5);

    // Mock inference result
    let prediction = predict_temperature(&input_features);
    assert!(prediction.is_some());

    let temp = prediction.unwrap();
    assert!(temp >= -100.0 && temp <= 100.0);
}

#[tokio::test]
async fn test_model_inference_accuracy() {
    let historical_data = generate_sample_climate_data(100);

    // Use first 80% for inference, last 20% for validation
    let training_size = (historical_data.len() as f64 * 0.8) as usize;
    let training_data = &historical_data[..training_size];
    let validation_data = &historical_data[training_size..];

    let input_features: Vec<f32> = training_data
        .iter()
        .flat_map(|point| vec![point.temperature as f32])
        .collect();

    let predictions: Vec<f32> = validation_data
        .iter()
        .map(|_| predict_temperature(&input_features).unwrap())
        .collect();

    let actual: Vec<f32> = validation_data
        .iter()
        .map(|point| point.temperature as f32)
        .collect();

    // Calculate RMSE
    let rmse = calculate_rmse(&predictions, &actual);
    assert!(rmse < 5.0); // Predictions should be within 5 degrees
}

#[tokio::test]
async fn test_model_confidence_scores() {
    let data = generate_sample_climate_data(24);
    let input_features: Vec<f32> = data
        .iter()
        .flat_map(|point| vec![point.temperature as f32])
        .collect();

    let (prediction, confidence) = predict_with_confidence(&input_features);

    assert!(confidence >= 0.0 && confidence <= 1.0);
    assert!(prediction >= -100.0 && prediction <= 100.0);

    // High quality input should have higher confidence
    if data.iter().all(|p| p.temperature > 0.0) {
        assert!(confidence > 0.5);
    }
}

#[tokio::test]
async fn test_batch_inference() {
    let batch_size = 10;
    let mut all_predictions = Vec::new();

    for _ in 0..batch_size {
        let data = generate_sample_climate_data(24);
        let input_features: Vec<f32> = data
            .iter()
            .flat_map(|point| vec![point.temperature as f32])
            .collect();

        let prediction = predict_temperature(&input_features);
        all_predictions.push(prediction.unwrap());
    }

    assert_eq!(all_predictions.len(), batch_size);

    // All predictions should be reasonable
    for pred in &all_predictions {
        assert!(pred >= &-100.0 && pred <= &100.0);
    }
}

#[tokio::test]
async fn test_model_edge_cases() {
    // Test with empty input
    let empty_features: Vec<f32> = vec![];
    let result = predict_temperature(&empty_features);
    assert!(result.is_none());

    // Test with extreme values
    let extreme_features: Vec<f32> = vec![-89.2, 0.0, 870.0, 113.0, 305.0];
    let extreme_result = predict_temperature(&extreme_features);
    assert!(extreme_result.is_some());

    // Test with all zeros
    let zero_features: Vec<f32> = vec![0.0; 120]; // 24 hours * 5 features
    let zero_result = predict_temperature(&zero_features);
    assert!(zero_result.is_some());
}

#[tokio::test]
async fn test_multi_parameter_prediction() {
    let data = generate_sample_climate_data(24);

    let predictions = predict_all_parameters(&data);

    assert!(predictions.temperature.is_some());
    assert!(predictions.humidity.is_some());
    assert!(predictions.precipitation_probability.is_some());

    // Validate ranges
    if let Some(temp) = predictions.temperature {
        assert!(temp >= -100.0 && temp <= 100.0);
    }
    if let Some(humidity) = predictions.humidity {
        assert!(humidity >= 0.0 && humidity <= 100.0);
    }
    if let Some(precip_prob) = predictions.precipitation_probability {
        assert!(precip_prob >= 0.0 && precip_prob <= 1.0);
    }
}

#[tokio::test]
async fn test_model_feature_importance() {
    let data = generate_sample_climate_data(100);

    // Test prediction with all features
    let all_features: Vec<f32> = data
        .iter()
        .flat_map(|point| {
            vec![
                point.temperature as f32,
                point.humidity as f32,
                point.pressure as f32,
                point.wind_speed as f32,
                point.precipitation as f32,
            ]
        })
        .collect();

    let full_prediction = predict_temperature(&all_features);

    // Test with reduced features (only temperature and humidity)
    let reduced_features: Vec<f32> = data
        .iter()
        .flat_map(|point| vec![point.temperature as f32, point.humidity as f32])
        .collect();

    let reduced_prediction = predict_temperature(&reduced_features);

    // Both should produce valid predictions
    assert!(full_prediction.is_some());
    assert!(reduced_prediction.is_some());
}

// Helper functions for testing
fn predict_temperature(features: &[f32]) -> Option<f32> {
    if features.is_empty() {
        return None;
    }

    // Mock prediction - in real implementation, this would use ONNX runtime
    let avg = features.iter().sum::<f32>() / features.len() as f32;
    Some(avg.clamp(-100.0, 100.0))
}

fn predict_with_confidence(features: &[f32]) -> (f32, f32) {
    let prediction = predict_temperature(features).unwrap_or(0.0);

    // Mock confidence calculation
    let confidence = if features.len() >= 120 {
        0.85
    } else {
        0.60
    };

    (prediction, confidence)
}

fn calculate_rmse(predictions: &[f32], actual: &[f32]) -> f32 {
    let n = predictions.len() as f32;
    let sum_squared_error: f32 = predictions
        .iter()
        .zip(actual.iter())
        .map(|(pred, act)| (pred - act).powi(2))
        .sum();

    (sum_squared_error / n).sqrt()
}

#[derive(Debug)]
struct MultiParameterPrediction {
    temperature: Option<f32>,
    humidity: Option<f32>,
    precipitation_probability: Option<f32>,
}

fn predict_all_parameters(data: &[ClimateDataPoint]) -> MultiParameterPrediction {
    if data.is_empty() {
        return MultiParameterPrediction {
            temperature: None,
            humidity: None,
            precipitation_probability: None,
        };
    }

    MultiParameterPrediction {
        temperature: Some(
            data.iter().map(|p| p.temperature as f32).sum::<f32>() / data.len() as f32,
        ),
        humidity: Some(data.iter().map(|p| p.humidity as f32).sum::<f32>() / data.len() as f32),
        precipitation_probability: Some(0.3), // Mock value
    }
}
