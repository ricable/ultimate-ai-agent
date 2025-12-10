use crate::{
    Config, Result, Error, InterferenceFeatureExtractor, InterferenceClassifierFactory,
    ClassifyRequest, ClassifyResponse, ConfidenceRequest, ConfidenceResponse,
    MitigationRequest, MitigationResponse, TrainRequest, TrainResponse,
    MetricsRequest, MetricsResponse, NoiseFloorMeasurement, CellParameters,
    TrainingExample, ModelConfig,
};
use tonic::{Request, Response, Status};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};
use chrono::Utc;
use std::collections::HashMap;

pub struct InterferenceService {
    config: Config,
    classifier: Arc<RwLock<Option<Box<dyn crate::models::InterferenceClassifier>>>>,
    feature_extractor: Arc<RwLock<InterferenceFeatureExtractor>>,
    performance_monitor: Arc<RwLock<PerformanceMonitor>>,
}

#[derive(Debug)]
struct PerformanceMonitor {
    classification_history: Vec<ClassificationEvent>,
    accuracy_history: Vec<f64>,
    last_performance_check: chrono::DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct ClassificationEvent {
    timestamp: chrono::DateTime<Utc>,
    cell_id: String,
    predicted_class: String,
    confidence: f64,
    processing_time_ms: f64,
}

impl InterferenceService {
    pub fn new(config: Config) -> Result<Self> {
        config.validate()?;
        
        let feature_extractor = InterferenceFeatureExtractor::new(
            config.features.noise_floor_window_size as usize,
            config.features.frequency_bins as usize,
            config.features.fft_size as usize,
            config.features.overlap_factor,
        );
        
        let performance_monitor = PerformanceMonitor {
            classification_history: Vec::new(),
            accuracy_history: Vec::new(),
            last_performance_check: Utc::now(),
        };
        
        Ok(Self {
            config,
            classifier: Arc::new(RwLock::new(None)),
            feature_extractor: Arc::new(RwLock::new(feature_extractor)),
            performance_monitor: Arc::new(RwLock::new(performance_monitor)),
        })
    }
    
    pub async fn initialize_classifier(&self) -> Result<()> {
        let classifier = InterferenceClassifierFactory::create_classifier(&self.config)?;
        let mut classifier_lock = self.classifier.write().await;
        *classifier_lock = Some(classifier);
        info!("Interference classifier initialized with architecture: {}", self.config.model.architecture);
        Ok(())
    }
    
    async fn validate_performance(&self) -> Result<bool> {
        let classifier_lock = self.classifier.read().await;
        if let Some(classifier) = classifier_lock.as_ref() {
            classifier.validate_performance(&self.config)
        } else {
            Ok(false)
        }
    }
    
    async fn update_performance_metrics(&self, event: ClassificationEvent) -> Result<()> {
        let mut monitor = self.performance_monitor.write().await;
        monitor.classification_history.push(event);
        
        // Keep only recent history (last hour)
        let cutoff_time = Utc::now() - chrono::Duration::hours(1);
        monitor.classification_history.retain(|event| event.timestamp > cutoff_time);
        
        // Update accuracy tracking if needed
        self.check_performance_requirements(&mut monitor).await?;
        
        Ok(())
    }
    
    async fn check_performance_requirements(&self, monitor: &mut PerformanceMonitor) -> Result<()> {
        let now = Utc::now();
        let check_interval = chrono::Duration::minutes(self.config.performance.evaluation_window_minutes as i64);
        
        if now - monitor.last_performance_check > check_interval {
            monitor.last_performance_check = now;
            
            // Check if we still meet accuracy requirements
            if !self.validate_performance().await? {
                warn!("Performance below requirements, model retraining may be needed");
                
                // Log performance warning
                if let Some(classifier) = self.classifier.read().await.as_ref() {
                    let metrics = classifier.get_metrics();
                    error!(
                        "Accuracy requirement not met: {:.3} < {:.3}",
                        metrics.accuracy,
                        self.config.performance.min_accuracy
                    );
                }
            }
        }
        
        Ok(())
    }
    
    fn build_mitigation_recommendations(&self, predicted_class: &str, confidence: f64) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if confidence < self.config.classification.confidence_threshold {
            recommendations.push("Low confidence prediction - collect more data for analysis".to_string());
            return recommendations;
        }
        
        match predicted_class {
            "EXTERNAL_JAMMER" => {
                recommendations.extend([
                    "Implement frequency hopping to avoid jammer frequencies".to_string(),
                    "Increase power control to overcome interference".to_string(),
                    "Deploy directional antennas to null jammer direction".to_string(),
                    "Coordinate with spectrum monitoring authority".to_string(),
                ]);
            }
            "PIM" => {
                recommendations.extend([
                    "Inspect and clean antenna connections".to_string(),
                    "Check for corroded or loose RF connections".to_string(),
                    "Implement PIM detection and location tools".to_string(),
                    "Consider antenna replacement if PIM persists".to_string(),
                ]);
            }
            "ADJACENT_CHANNEL" => {
                recommendations.extend([
                    "Implement enhanced filtering at affected frequencies".to_string(),
                    "Coordinate with adjacent channel operators".to_string(),
                    "Adjust power levels to minimize spillover".to_string(),
                    "Consider frequency refarming if available".to_string(),
                ]);
            }
            "THERMAL_NOISE" => {
                recommendations.extend([
                    "Check receiver sensitivity and calibration".to_string(),
                    "Monitor equipment temperature and cooling".to_string(),
                    "Inspect cable losses and connections".to_string(),
                    "Consider receiver upgrade if noise floor too high".to_string(),
                ]);
            }
            "LEGITIMATE_TRAFFIC" => {
                recommendations.extend([
                    "Normal traffic load - no interference mitigation needed".to_string(),
                    "Monitor for capacity management".to_string(),
                    "Consider load balancing if approaching limits".to_string(),
                ]);
            }
            _ => {
                recommendations.push("Unknown interference type - manual investigation required".to_string());
            }
        }
        
        recommendations
    }
    
    fn calculate_priority_level(&self, predicted_class: &str, confidence: f64) -> i32 {
        let base_priority = match predicted_class {
            "EXTERNAL_JAMMER" => 5, // Highest priority
            "PIM" => 4,
            "ADJACENT_CHANNEL" => 3,
            "THERMAL_NOISE" => 2,
            "LEGITIMATE_TRAFFIC" => 1, // Lowest priority
            _ => 3, // Medium priority for unknown
        };
        
        // Adjust priority based on confidence
        let confidence_modifier = if confidence > 0.9 {
            1
        } else if confidence > 0.7 {
            0
        } else {
            -1
        };
        
        (base_priority + confidence_modifier).max(1).min(5)
    }
    
    fn estimate_impact(&self, predicted_class: &str, confidence: f64) -> String {
        let impact_level = if confidence > 0.8 {
            match predicted_class {
                "EXTERNAL_JAMMER" => "High - Significant capacity and quality degradation",
                "PIM" => "Medium - Uplink performance degradation",
                "ADJACENT_CHANNEL" => "Medium - Localized interference in affected channels",
                "THERMAL_NOISE" => "Low - Baseline noise floor elevation",
                "LEGITIMATE_TRAFFIC" => "None - Normal network operation",
                _ => "Unknown - Manual assessment required",
            }
        } else {
            "Uncertain - Low confidence prediction"
        };
        
        impact_level.to_string()
    }
}

#[tonic::async_trait]
impl crate::proto::interference_classifier::interference_classifier_server::InterferenceClassifier for InterferenceService {
    async fn classify_ul_interference(
        &self,
        request: Request<ClassifyRequest>,
    ) -> std::result::Result<Response<ClassifyResponse>, Status> {
        let start_time = std::time::Instant::now();
        let req = request.into_inner();
        
        info!("Processing interference classification request for cell: {}", req.cell_id);
        
        // Validate request
        if req.measurements.is_empty() {
            return Err(Status::invalid_argument("No measurements provided"));
        }
        
        if req.cell_params.is_none() {
            return Err(Status::invalid_argument("Cell parameters required"));
        }
        
        let cell_params = req.cell_params.unwrap();
        
        // Extract features
        let mut extractor = self.feature_extractor.write().await;
        let features = extractor.extract_features(&req.measurements, &cell_params)
            .map_err(|e| Status::internal(format!("Feature extraction failed: {}", e)))?;
        
        // Classify
        let classifier_lock = self.classifier.read().await;
        let classifier = classifier_lock.as_ref()
            .ok_or_else(|| Status::failed_precondition("Classifier not trained"))?;
        
        let result = classifier.predict(&features)
            .map_err(|e| Status::internal(format!("Classification failed: {}", e)))?;
        
        let processing_time = start_time.elapsed().as_millis() as f64;
        
        // Update performance monitoring
        let event = ClassificationEvent {
            timestamp: Utc::now(),
            cell_id: req.cell_id.clone(),
            predicted_class: result.predicted_class.clone(),
            confidence: result.confidence,
            processing_time_ms: processing_time,
        };
        
        if let Err(e) = self.update_performance_metrics(event).await {
            warn!("Failed to update performance metrics: {}", e);
        }
        
        // Convert feature vector for response
        let feature_vector = features.to_vec();
        
        let response = ClassifyResponse {
            interference_class: result.predicted_class,
            confidence: result.confidence,
            timestamp: Utc::now().to_rfc3339(),
            feature_vector,
        };
        
        info!(
            "Classification completed for cell {} in {:.2}ms: {} (confidence: {:.3})",
            req.cell_id, processing_time, response.interference_class, response.confidence
        );
        
        Ok(Response::new(response))
    }
    
    async fn get_classification_confidence(
        &self,
        request: Request<ConfidenceRequest>,
    ) -> std::result::Result<Response<ConfidenceResponse>, Status> {
        let req = request.into_inner();
        
        if req.measurements.is_empty() {
            return Err(Status::invalid_argument("No measurements provided"));
        }
        
        // For this implementation, we'll use the same classification logic
        // In a more sophisticated system, this might use ensemble uncertainty
        let classify_req = ClassifyRequest {
            cell_id: req.cell_id,
            measurements: req.measurements,
            cell_params: None, // Use default parameters for confidence scoring
        };
        
        let classify_response = self.classify_ul_interference(Request::new(classify_req)).await?;
        let classify_result = classify_response.into_inner();
        
        let response = ConfidenceResponse {
            confidence: classify_result.confidence,
            model_version: "1.0".to_string(),
            sample_count: classify_result.feature_vector.len() as i32,
        };
        
        Ok(Response::new(response))
    }
    
    async fn get_mitigation_recommendations(
        &self,
        request: Request<MitigationRequest>,
    ) -> std::result::Result<Response<MitigationResponse>, Status> {
        let req = request.into_inner();
        
        let recommendations = self.build_mitigation_recommendations(&req.interference_class, req.confidence);
        let priority_level = self.calculate_priority_level(&req.interference_class, req.confidence);
        let estimated_impact = self.estimate_impact(&req.interference_class, req.confidence);
        
        let response = MitigationResponse {
            recommendations,
            priority_level,
            estimated_impact,
        };
        
        Ok(Response::new(response))
    }
    
    async fn train_model(
        &self,
        request: Request<TrainRequest>,
    ) -> std::result::Result<Response<TrainResponse>, Status> {
        let req = request.into_inner();
        let start_time = std::time::Instant::now();
        
        info!("Starting model training with {} examples", req.examples.len());
        
        if req.examples.is_empty() {
            return Err(Status::invalid_argument("No training examples provided"));
        }
        
        // Prepare training data
        let mut features_list = Vec::new();
        let mut labels = Vec::new();
        
        let mut extractor = self.feature_extractor.write().await;
        
        for example in &req.examples {
            if example.measurements.is_empty() {
                continue;
            }
            
            let cell_params = example.cell_params.as_ref()
                .ok_or_else(|| Status::invalid_argument("Cell parameters required for training examples"))?;
            
            let features = extractor.extract_features(&example.measurements, cell_params)
                .map_err(|e| Status::internal(format!("Feature extraction failed: {}", e)))?;
            
            features_list.push(features);
            labels.push(example.true_interference_class.clone());
        }
        
        if features_list.is_empty() {
            return Err(Status::invalid_argument("No valid training examples"));
        }
        
        // Convert to ndarray format
        let n_samples = features_list.len();
        let n_features = features_list[0].len();
        let features_flat: Vec<f64> = features_list.into_iter().flat_map(|f| f.to_vec()).collect();
        
        let features_matrix = ndarray::Array2::from_shape_vec((n_samples, n_features), features_flat)
            .map_err(|e| Status::internal(format!("Failed to create feature matrix: {}", e)))?;
        
        // Train classifier
        let mut classifier_lock = self.classifier.write().await;
        if classifier_lock.is_none() {
            *classifier_lock = Some(InterferenceClassifierFactory::create_classifier(&self.config)
                .map_err(|e| Status::internal(format!("Failed to create classifier: {}", e)))?);
        }
        
        let classifier = classifier_lock.as_mut().unwrap();
        let training_result = classifier.train(&features_matrix, &labels)
            .map_err(|e| Status::internal(format!("Training failed: {}", e)))?;
        
        let training_time = start_time.elapsed().as_secs_f64();
        
        let response = TrainResponse {
            model_id: training_result.model_id,
            training_accuracy: training_result.metrics.accuracy,
            validation_accuracy: training_result.validation_metrics.accuracy,
            epochs_trained: training_result.epochs_trained as i32,
            training_time: format!("{:.2}s", training_time),
        };
        
        info!(
            "Model training completed: accuracy={:.3}, validation_accuracy={:.3}, time={:.2}s",
            response.training_accuracy, response.validation_accuracy, training_time
        );
        
        Ok(Response::new(response))
    }
    
    async fn get_model_metrics(
        &self,
        request: Request<MetricsRequest>,
    ) -> std::result::Result<Response<MetricsResponse>, Status> {
        let _req = request.into_inner();
        
        let classifier_lock = self.classifier.read().await;
        let classifier = classifier_lock.as_ref()
            .ok_or_else(|| Status::failed_precondition("Classifier not trained"))?;
        
        let metrics = classifier.get_metrics();
        
        // Convert class metrics
        let mut class_metrics_map = HashMap::new();
        for (class, class_metric) in &metrics.class_metrics {
            class_metrics_map.insert(class.clone(), class_metric.f1_score);
        }
        
        let response = MetricsResponse {
            accuracy: metrics.accuracy,
            precision: metrics.precision,
            recall: metrics.recall,
            f1_score: metrics.f1_score,
            class_metrics: class_metrics_map,
        };
        
        Ok(Response::new(response))
    }
}