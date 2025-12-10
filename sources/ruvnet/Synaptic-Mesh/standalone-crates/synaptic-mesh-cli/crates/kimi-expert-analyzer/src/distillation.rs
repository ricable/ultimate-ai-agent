//! Knowledge distillation pipeline for creating micro-experts

use crate::expert::{ExpertDomain, MicroExpert, ExpertParameters, ExpertWeights, ExpertMetrics};
use crate::analysis::{ModelArchitecture, SpecializationAnalysis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use anyhow::Result;
use ndarray::{Array2, Array1};

/// Knowledge distillation pipeline for converting Kimi-K2 experts to micro-experts
#[derive(Debug, Clone)]
pub struct DistillationPipeline {
    /// Teacher model (original Kimi-K2)
    pub teacher_model: TeacherModel,
    /// Student micro-experts being trained
    pub student_experts: HashMap<ExpertDomain, Vec<StudentExpert>>,
    /// Distillation configuration
    pub config: DistillationConfig,
    /// Training progress tracker
    pub progress: DistillationProgress,
}

impl DistillationPipeline {
    /// Create a new distillation pipeline
    pub fn new(teacher_model: TeacherModel, config: DistillationConfig) -> Self {
        Self {
            teacher_model,
            student_experts: HashMap::new(),
            config,
            progress: DistillationProgress::new(),
        }
    }

    /// Initialize student experts for all domains
    pub async fn initialize_students(&mut self) -> Result<()> {
        tracing::info!("Initializing student experts for all domains");

        for domain in ExpertDomain::all_domains() {
            let students = self.create_domain_students(&domain).await?;
            self.student_experts.insert(domain, students);
        }

        tracing::info!("Initialized {} domains with student experts", self.student_experts.len());
        Ok(())
    }

    /// Distill knowledge for a specific expert domain
    pub async fn distill_expert(&mut self, domain: ExpertDomain) -> Result<MicroExpert> {
        tracing::info!("Starting knowledge distillation for domain {:?}", domain);

        // Get relevant teacher experts for this domain
        let teacher_experts = self.teacher_model.get_domain_experts(&domain)?;
        
        // Get student expert for this domain
        let students = self.student_experts.get_mut(&domain)
            .ok_or_else(|| anyhow::anyhow!("No student experts for domain {:?}", domain))?;
        
        let student = students.first_mut()
            .ok_or_else(|| anyhow::anyhow!("No student expert available for domain {:?}", domain))?;

        // Generate training data
        let training_data = self.generate_training_data(&domain, &teacher_experts).await?;
        
        // Perform distillation training
        let distilled_weights = self.perform_distillation(
            student,
            &teacher_experts,
            &training_data
        ).await?;

        // Create micro-expert
        let micro_expert = self.create_micro_expert(
            domain.clone(),
            distilled_weights,
            &training_data
        ).await?;

        // Validate performance
        let performance = self.validate_performance(&micro_expert, &training_data).await?;
        tracing::info!("Distillation completed with performance: {:.2}%", performance.accuracy * 100.0);

        Ok(micro_expert)
    }

    /// Validate overall performance of the distillation process
    pub async fn validate_performance(&self) -> Result<PerformanceMetrics> {
        tracing::info!("Validating overall distillation performance");

        let mut metrics = PerformanceMetrics::new();
        
        // Test on validation dataset
        for (domain, students) in &self.student_experts {
            for student in students {
                let domain_metrics = self.evaluate_student(student, domain).await?;
                metrics.add_domain_metrics(domain.clone(), domain_metrics);
            }
        }

        // Calculate aggregate metrics
        metrics.calculate_aggregates();
        
        tracing::info!("Overall validation completed: {:.2}% accuracy", metrics.overall_accuracy * 100.0);
        Ok(metrics)
    }

    /// Create student experts for a specific domain
    async fn create_domain_students(&self, domain: &ExpertDomain) -> Result<Vec<StudentExpert>> {
        let mut students = Vec::new();
        
        // Create primary student expert
        let primary_student = StudentExpert::new(
            domain.clone(),
            self.config.get_domain_config(domain),
        )?;
        
        students.push(primary_student);
        
        // Create additional students if ensemble is enabled
        if self.config.use_ensemble {
            for i in 1..self.config.ensemble_size {
                let ensemble_student = StudentExpert::new_ensemble_variant(
                    domain.clone(),
                    self.config.get_domain_config(domain),
                    i,
                )?;
                students.push(ensemble_student);
            }
        }
        
        Ok(students)
    }

    /// Generate training data for knowledge distillation
    async fn generate_training_data(
        &self,
        domain: &ExpertDomain,
        teacher_experts: &[TeacherExpert],
    ) -> Result<DomainTrainingData> {
        tracing::info!("Generating training data for domain {:?}", domain);

        let mut training_data = DomainTrainingData::new(domain.clone());
        
        // Generate domain-specific prompts
        let prompts = self.generate_domain_prompts(domain).await?;
        
        // Get teacher responses for each prompt
        for prompt in prompts {
            let teacher_response = self.get_teacher_response(&prompt, teacher_experts).await?;
            let training_sample = TrainingSample {
                input: prompt,
                teacher_output: teacher_response.output,
                teacher_logits: teacher_response.logits,
                attention_weights: teacher_response.attention_weights,
                expert_activations: teacher_response.expert_activations,
            };
            training_data.add_sample(training_sample);
        }

        // Add negative samples (what this expert should NOT handle)
        let negative_samples = self.generate_negative_samples(domain).await?;
        training_data.add_negative_samples(negative_samples);

        tracing::info!("Generated {} training samples for domain {:?}", 
                      training_data.samples.len(), domain);
        Ok(training_data)
    }

    /// Perform the actual distillation training
    async fn perform_distillation(
        &self,
        student: &mut StudentExpert,
        teacher_experts: &[TeacherExpert],
        training_data: &DomainTrainingData,
    ) -> Result<ExpertWeights> {
        tracing::info!("Performing distillation training");

        let mut weights = ExpertWeights::new();
        let mut optimizer = DistillationOptimizer::new(&self.config.optimizer_config)?;

        for epoch in 0..self.config.max_epochs {
            let mut epoch_loss = 0.0;
            let mut batch_count = 0;

            // Train on batches
            for batch in training_data.iter_batches(self.config.batch_size) {
                let batch_loss = self.train_batch(student, &batch, &mut optimizer).await?;
                epoch_loss += batch_loss;
                batch_count += 1;

                // Log progress
                if batch_count % self.config.log_interval == 0 {
                    tracing::debug!("Epoch {}, Batch {}: Loss = {:.4}", 
                                  epoch, batch_count, batch_loss);
                }
            }

            let avg_loss = epoch_loss / batch_count as f32;
            tracing::info!("Epoch {} completed: Average Loss = {:.4}", epoch, avg_loss);

            // Early stopping check
            if avg_loss < self.config.convergence_threshold {
                tracing::info!("Converged early at epoch {}", epoch);
                break;
            }

            // Validation check
            if epoch % self.config.validation_interval == 0 {
                let validation_metrics = self.validate_student(student, training_data).await?;
                tracing::info!("Validation at epoch {}: Accuracy = {:.2}%", 
                              epoch, validation_metrics.accuracy * 100.0);
                
                if validation_metrics.accuracy > self.config.target_accuracy {
                    tracing::info!("Target accuracy reached at epoch {}", epoch);
                    break;
                }
            }
        }

        // Extract final weights
        weights = student.extract_weights()?;
        weights.compression_ratio = self.calculate_compression_ratio(&weights, teacher_experts)?;

        Ok(weights)
    }

    /// Create a micro-expert from distilled weights
    async fn create_micro_expert(
        &self,
        domain: ExpertDomain,
        weights: ExpertWeights,
        training_data: &DomainTrainingData,
    ) -> Result<MicroExpert> {
        let parameters = ExpertParameters {
            input_dim: domain.input_features(),
            output_dim: domain.output_dimension(),
            hidden_dims: self.config.get_domain_config(&domain).hidden_dims.clone(),
            activation: self.config.get_domain_config(&domain).activation.clone(),
            learning_rate: self.config.learning_rate,
            dropout_rate: self.config.dropout_rate,
            target_params: domain.target_parameters(),
        };

        let micro_expert = MicroExpert::new(
            self.generate_expert_id(),
            domain,
            parameters,
            weights,
        )?;

        Ok(micro_expert)
    }

    /// Generate domain-specific prompts for training
    async fn generate_domain_prompts(&self, domain: &ExpertDomain) -> Result<Vec<String>> {
        let prompts = match domain {
            ExpertDomain::Reasoning => vec![
                "Solve this logic puzzle: If all cats are animals, and Fluffy is a cat, what can we conclude?",
                "Given the premises: A→B, B→C, A is true. What can we derive?",
                "Analyze the following argument for logical validity: All birds can fly. Penguins are birds. Therefore, penguins can fly.",
            ],
            ExpertDomain::Coding => vec![
                "Write a Python function to find the maximum element in a list.",
                "Debug this code: def factorial(n): return n * factorial(n-1)",
                "Implement a binary search algorithm in Rust.",
                "Explain the time complexity of this sorting algorithm: bubble sort.",
            ],
            ExpertDomain::Language => vec![
                "Translate 'Hello, how are you?' to French.",
                "Summarize the following text in one sentence: [long text]",
                "Identify the parts of speech in: 'The quick brown fox jumps.'",
                "Correct the grammar: 'Me and him went to store yesterday.'",
            ],
            ExpertDomain::ToolUse => vec![
                "How would you use a calculator to compute 15% of 240?",
                "Call the weather API to get current conditions for New York.",
                "Use the search function to find information about quantum computing.",
            ],
            ExpertDomain::Mathematics => vec![
                "Solve for x: 2x + 5 = 13",
                "Find the derivative of f(x) = x³ + 2x² - x + 1",
                "Calculate the area of a circle with radius 7.",
                "What is the probability of getting heads twice in three coin flips?",
            ],
            ExpertDomain::Context => vec![
                "Given this long conversation history, what was the main topic discussed 10 messages ago?",
                "Summarize the key points from this 50-page document.",
                "What information from earlier in our conversation is relevant to this new question?",
            ],
        };
        
        Ok(prompts.into_iter().map(String::from).collect())
    }

    /// Get teacher model response for a prompt
    async fn get_teacher_response(
        &self,
        prompt: &str,
        teacher_experts: &[TeacherExpert],
    ) -> Result<TeacherResponse> {
        // This would interface with the actual Kimi-K2 model
        // For now, return a placeholder response
        Ok(TeacherResponse {
            output: format!("Teacher response to: {}", prompt),
            logits: vec![0.1, 0.9, 0.05], // Placeholder logits
            attention_weights: Array2::zeros((12, 64)), // Placeholder attention
            expert_activations: HashMap::new(),
        })
    }

    /// Generate negative samples (what this expert should not handle)
    async fn generate_negative_samples(&self, domain: &ExpertDomain) -> Result<Vec<NegativeSample>> {
        let mut negative_samples = Vec::new();
        
        // Generate samples from other domains
        for other_domain in ExpertDomain::all_domains() {
            if other_domain != *domain {
                let other_prompts = self.generate_domain_prompts(&other_domain).await?;
                for prompt in other_prompts.into_iter().take(2) { // Limit negative samples
                    negative_samples.push(NegativeSample {
                        input: prompt,
                        should_activate: false,
                        negative_domain: other_domain.clone(),
                    });
                }
            }
        }
        
        Ok(negative_samples)
    }

    /// Train on a single batch
    async fn train_batch(
        &self,
        student: &mut StudentExpert,
        batch: &[TrainingSample],
        optimizer: &mut DistillationOptimizer,
    ) -> Result<f32> {
        // This would implement the actual training step
        // For now, return a placeholder loss
        Ok(0.5) // Placeholder loss
    }

    /// Validate student performance
    async fn validate_student(
        &self,
        student: &StudentExpert,
        training_data: &DomainTrainingData,
    ) -> Result<ValidationMetrics> {
        // This would implement validation logic
        Ok(ValidationMetrics {
            accuracy: 0.85,
            loss: 0.3,
            inference_speed_ms: 10.0,
        })
    }

    /// Evaluate student expert
    async fn evaluate_student(
        &self,
        student: &StudentExpert,
        domain: &ExpertDomain,
    ) -> Result<ExpertMetrics> {
        Ok(ExpertMetrics {
            accuracy: 0.85,
            inference_speed_ms: 10.0,
            memory_usage: domain.target_parameters() * 4, // 4 bytes per float32
            specialization_score: 0.9,
            distillation_quality: 0.8,
        })
    }

    /// Calculate compression ratio
    fn calculate_compression_ratio(
        &self,
        micro_weights: &ExpertWeights,
        teacher_experts: &[TeacherExpert],
    ) -> Result<f32> {
        let teacher_params: u64 = teacher_experts.iter().map(|e| e.parameter_count).sum();
        let micro_params = micro_weights.parameter_count as u64;
        Ok(teacher_params as f32 / micro_params as f32)
    }

    /// Generate unique expert ID
    fn generate_expert_id(&self) -> usize {
        // This would generate a unique ID
        42 // Placeholder
    }
}

/// Teacher model wrapper
#[derive(Debug, Clone)]
pub struct TeacherModel {
    pub architecture: ModelArchitecture,
    pub model_path: PathBuf,
    pub loaded: bool,
}

impl TeacherModel {
    pub fn new(architecture: ModelArchitecture, model_path: PathBuf) -> Self {
        Self {
            architecture,
            model_path,
            loaded: false,
        }
    }

    pub fn get_domain_experts(&self, domain: &ExpertDomain) -> Result<Vec<TeacherExpert>> {
        // This would return actual teacher experts for the domain
        Ok(vec![TeacherExpert {
            layer_index: 12,
            expert_id: 0,
            parameter_count: 32_000_000,
            specialization_confidence: 0.9,
        }])
    }
}

/// Individual teacher expert
#[derive(Debug, Clone)]
pub struct TeacherExpert {
    pub layer_index: usize,
    pub expert_id: usize,
    pub parameter_count: u64,
    pub specialization_confidence: f32,
}

/// Student expert being trained
#[derive(Debug, Clone)]
pub struct StudentExpert {
    pub domain: ExpertDomain,
    pub config: DomainConfig,
    pub network: StudentNetwork,
    pub training_state: TrainingState,
}

impl StudentExpert {
    pub fn new(domain: ExpertDomain, config: DomainConfig) -> Result<Self> {
        let network = StudentNetwork::new(&domain, &config)?;
        Ok(Self {
            domain,
            config,
            network,
            training_state: TrainingState::new(),
        })
    }

    pub fn new_ensemble_variant(
        domain: ExpertDomain,
        config: DomainConfig,
        variant_id: usize,
    ) -> Result<Self> {
        let mut modified_config = config;
        modified_config.add_variant_noise(variant_id);
        Self::new(domain, modified_config)
    }

    pub fn extract_weights(&self) -> Result<ExpertWeights> {
        self.network.extract_weights()
    }
}

/// Student neural network
#[derive(Debug, Clone)]
pub struct StudentNetwork {
    pub layers: Vec<StudentLayer>,
    pub parameter_count: usize,
}

impl StudentNetwork {
    pub fn new(domain: &ExpertDomain, config: &DomainConfig) -> Result<Self> {
        let mut layers = Vec::new();
        let mut param_count = 0;

        // Create layers based on configuration
        for (i, &layer_size) in config.hidden_dims.iter().enumerate() {
            let input_size = if i == 0 {
                domain.input_features()
            } else {
                config.hidden_dims[i - 1]
            };

            let layer = StudentLayer::new(input_size, layer_size, config.activation.clone())?;
            param_count += layer.parameter_count();
            layers.push(layer);
        }

        // Output layer
        let output_layer = StudentLayer::new(
            config.hidden_dims.last().copied().unwrap_or(domain.input_features()),
            domain.output_dimension(),
            config.activation.clone(),
        )?;
        param_count += output_layer.parameter_count();
        layers.push(output_layer);

        Ok(Self {
            layers,
            parameter_count: param_count,
        })
    }

    pub fn extract_weights(&self) -> Result<ExpertWeights> {
        let mut weights = ExpertWeights::new();
        
        for (i, layer) in self.layers.iter().enumerate() {
            let (layer_weights, layer_biases) = layer.get_weights();
            weights.add_layer(i, layer_weights, layer_biases);
        }

        Ok(weights)
    }
}

/// Individual layer in student network
#[derive(Debug, Clone)]
pub struct StudentLayer {
    pub input_size: usize,
    pub output_size: usize,
    pub activation: crate::expert::ActivationFunction,
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
}

impl StudentLayer {
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: crate::expert::ActivationFunction,
    ) -> Result<Self> {
        // Initialize weights and biases
        let weights = Array2::zeros((output_size, input_size));
        let biases = Array1::zeros(output_size);

        Ok(Self {
            input_size,
            output_size,
            activation,
            weights,
            biases,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.weights.len() + self.biases.len()
    }

    pub fn get_weights(&self) -> (Array2<f32>, Array1<f32>) {
        (self.weights.clone(), self.biases.clone())
    }
}

/// Training state for student expert
#[derive(Debug, Clone)]
pub struct TrainingState {
    pub current_epoch: usize,
    pub best_loss: f32,
    pub best_accuracy: f32,
    pub plateau_count: usize,
}

impl TrainingState {
    pub fn new() -> Self {
        Self {
            current_epoch: 0,
            best_loss: f32::INFINITY,
            best_accuracy: 0.0,
            plateau_count: 0,
        }
    }
}

/// Distillation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationConfig {
    pub max_epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub dropout_rate: f32,
    pub target_accuracy: f32,
    pub convergence_threshold: f32,
    pub validation_interval: usize,
    pub log_interval: usize,
    pub use_ensemble: bool,
    pub ensemble_size: usize,
    pub domain_configs: HashMap<ExpertDomain, DomainConfig>,
    pub optimizer_config: OptimizerConfig,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        let mut domain_configs = HashMap::new();
        for domain in ExpertDomain::all_domains() {
            domain_configs.insert(domain.clone(), DomainConfig::default_for_domain(&domain));
        }

        Self {
            max_epochs: 100,
            batch_size: 32,
            learning_rate: 0.001,
            dropout_rate: 0.1,
            target_accuracy: 0.85,
            convergence_threshold: 0.001,
            validation_interval: 10,
            log_interval: 100,
            use_ensemble: false,
            ensemble_size: 3,
            domain_configs,
            optimizer_config: OptimizerConfig::default(),
        }
    }
}

impl DistillationConfig {
    pub fn get_domain_config(&self, domain: &ExpertDomain) -> DomainConfig {
        self.domain_configs.get(domain).cloned()
            .unwrap_or_else(|| DomainConfig::default_for_domain(domain))
    }
}

/// Domain-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainConfig {
    pub hidden_dims: Vec<usize>,
    pub activation: crate::expert::ActivationFunction,
    pub domain_weight: f32,
    pub specialization_bonus: f32,
}

impl DomainConfig {
    pub fn default_for_domain(domain: &ExpertDomain) -> Self {
        let hidden_dims = match domain {
            ExpertDomain::Reasoning => vec![256, 128],
            ExpertDomain::Coding => vec![512, 256, 128],
            ExpertDomain::Language => vec![384, 192],
            ExpertDomain::ToolUse => vec![192, 96],
            ExpertDomain::Mathematics => vec![320, 160],
            ExpertDomain::Context => vec![768, 384, 192],
        };

        Self {
            hidden_dims,
            activation: crate::expert::ActivationFunction::ReLU,
            domain_weight: 1.0,
            specialization_bonus: 0.1,
        }
    }

    pub fn add_variant_noise(&mut self, variant_id: usize) {
        // Add small variations for ensemble members
        let noise_factor = 0.1 * (variant_id as f32);
        for dim in &mut self.hidden_dims {
            *dim = ((*dim as f32) * (1.0 + noise_factor)) as usize;
        }
    }
}

/// Optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub optimizer_type: OptimizerType,
    pub learning_rate: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            optimizer_type: OptimizerType::Adam,
            learning_rate: 0.001,
            momentum: 0.9,
            weight_decay: 0.0001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

/// Optimizer types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerType {
    SGD,
    Adam,
    AdamW,
    RMSprop,
}

/// Distillation optimizer
#[derive(Debug)]
pub struct DistillationOptimizer {
    config: OptimizerConfig,
    state: OptimizerState,
}

impl DistillationOptimizer {
    pub fn new(config: &OptimizerConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            state: OptimizerState::new(),
        })
    }
}

/// Optimizer internal state
#[derive(Debug)]
pub struct OptimizerState {
    pub step_count: usize,
    pub momentum_buffers: HashMap<String, Array2<f32>>,
    pub squared_grad_buffers: HashMap<String, Array2<f32>>,
}

impl OptimizerState {
    pub fn new() -> Self {
        Self {
            step_count: 0,
            momentum_buffers: HashMap::new(),
            squared_grad_buffers: HashMap::new(),
        }
    }
}

/// Teacher model response
#[derive(Debug, Clone)]
pub struct TeacherResponse {
    pub output: String,
    pub logits: Vec<f32>,
    pub attention_weights: Array2<f32>,
    pub expert_activations: HashMap<usize, f32>,
}

/// Training sample for distillation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    pub input: String,
    pub teacher_output: String,
    pub teacher_logits: Vec<f32>,
    pub attention_weights: Array2<f32>,
    pub expert_activations: HashMap<usize, f32>,
}

/// Negative training sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NegativeSample {
    pub input: String,
    pub should_activate: bool,
    pub negative_domain: ExpertDomain,
}

/// Domain-specific training data
#[derive(Debug, Clone)]
pub struct DomainTrainingData {
    pub domain: ExpertDomain,
    pub samples: Vec<TrainingSample>,
    pub negative_samples: Vec<NegativeSample>,
    pub validation_samples: Vec<TrainingSample>,
}

impl DomainTrainingData {
    pub fn new(domain: ExpertDomain) -> Self {
        Self {
            domain,
            samples: Vec::new(),
            negative_samples: Vec::new(),
            validation_samples: Vec::new(),
        }
    }

    pub fn add_sample(&mut self, sample: TrainingSample) {
        self.samples.push(sample);
    }

    pub fn add_negative_samples(&mut self, samples: Vec<NegativeSample>) {
        self.negative_samples.extend(samples);
    }

    pub fn iter_batches(&self, batch_size: usize) -> BatchIterator {
        BatchIterator::new(&self.samples, batch_size)
    }
}

/// Iterator for training batches
pub struct BatchIterator<'a> {
    samples: &'a [TrainingSample],
    batch_size: usize,
    current_index: usize,
}

impl<'a> BatchIterator<'a> {
    pub fn new(samples: &'a [TrainingSample], batch_size: usize) -> Self {
        Self {
            samples,
            batch_size,
            current_index: 0,
        }
    }
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = &'a [TrainingSample];

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.samples.len() {
            return None;
        }

        let end_index = (self.current_index + self.batch_size).min(self.samples.len());
        let batch = &self.samples[self.current_index..end_index];
        self.current_index = end_index;

        Some(batch)
    }
}

/// Complete training dataset for all domains
#[derive(Debug, Clone)]
pub struct TrainingDataset {
    pub domain_data: HashMap<ExpertDomain, DomainTrainingData>,
    pub metadata: DatasetMetadata,
}

impl TrainingDataset {
    pub fn new() -> Self {
        Self {
            domain_data: HashMap::new(),
            metadata: DatasetMetadata::new(),
        }
    }

    pub fn add_domain_data(&mut self, domain: ExpertDomain, data: DomainTrainingData) {
        self.domain_data.insert(domain, data);
    }

    pub async fn save(&self, output_dir: &std::path::Path) -> Result<()> {
        tokio::fs::create_dir_all(output_dir).await?;
        
        for (domain, data) in &self.domain_data {
            let domain_file = output_dir.join(format!("{:?}_training_data.json", domain));
            let serialized = serde_json::to_string_pretty(data)?;
            tokio::fs::write(domain_file, serialized).await?;
        }

        // Save metadata
        let metadata_file = output_dir.join("dataset_metadata.json");
        let metadata_json = serde_json::to_string_pretty(&self.metadata)?;
        tokio::fs::write(metadata_file, metadata_json).await?;

        Ok(())
    }
}

/// Dataset metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub total_samples: usize,
    pub domain_sample_counts: HashMap<ExpertDomain, usize>,
    pub generation_config: String,
}

impl DatasetMetadata {
    pub fn new() -> Self {
        Self {
            created_at: chrono::Utc::now(),
            total_samples: 0,
            domain_sample_counts: HashMap::new(),
            generation_config: "default".to_string(),
        }
    }
}

/// Performance metrics for the distillation process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub domain_metrics: HashMap<ExpertDomain, ExpertMetrics>,
    pub overall_accuracy: f32,
    pub overall_speed: f32,
    pub total_parameters: usize,
    pub compression_ratio: f32,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            domain_metrics: HashMap::new(),
            overall_accuracy: 0.0,
            overall_speed: 0.0,
            total_parameters: 0,
            compression_ratio: 0.0,
        }
    }

    pub fn add_domain_metrics(&mut self, domain: ExpertDomain, metrics: ExpertMetrics) {
        self.total_parameters += metrics.memory_usage / 4; // Convert bytes to parameters
        self.domain_metrics.insert(domain, metrics);
    }

    pub fn calculate_aggregates(&mut self) {
        if self.domain_metrics.is_empty() {
            return;
        }

        let count = self.domain_metrics.len() as f32;
        self.overall_accuracy = self.domain_metrics.values()
            .map(|m| m.accuracy)
            .sum::<f32>() / count;
        
        self.overall_speed = self.domain_metrics.values()
            .map(|m| m.inference_speed_ms)
            .sum::<f32>() / count;
    }
}

/// Validation metrics for a student expert
#[derive(Debug, Clone)]
pub struct ValidationMetrics {
    pub accuracy: f32,
    pub loss: f32,
    pub inference_speed_ms: f32,
}

/// Distillation progress tracker
#[derive(Debug, Clone)]
pub struct DistillationProgress {
    pub domains_completed: Vec<ExpertDomain>,
    pub current_domain: Option<ExpertDomain>,
    pub start_time: std::time::Instant,
    pub domain_start_times: HashMap<ExpertDomain, std::time::Instant>,
}

impl DistillationProgress {
    pub fn new() -> Self {
        Self {
            domains_completed: Vec::new(),
            current_domain: None,
            start_time: std::time::Instant::now(),
            domain_start_times: HashMap::new(),
        }
    }

    pub fn start_domain(&mut self, domain: ExpertDomain) {
        self.current_domain = Some(domain.clone());
        self.domain_start_times.insert(domain, std::time::Instant::now());
    }

    pub fn complete_domain(&mut self, domain: ExpertDomain) {
        self.domains_completed.push(domain.clone());
        self.current_domain = None;
    }

    pub fn completion_percentage(&self) -> f32 {
        let total_domains = ExpertDomain::all_domains().len() as f32;
        (self.domains_completed.len() as f32) / total_domains * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distillation_config_default() {
        let config = DistillationConfig::default();
        assert_eq!(config.max_epochs, 100);
        assert_eq!(config.batch_size, 32);
        assert!(!config.use_ensemble);
    }

    #[test]
    fn test_domain_training_data() {
        let mut data = DomainTrainingData::new(ExpertDomain::Reasoning);
        assert_eq!(data.domain, ExpertDomain::Reasoning);
        assert!(data.samples.is_empty());
    }

    #[test]
    fn test_training_dataset() {
        let dataset = TrainingDataset::new();
        assert!(dataset.domain_data.is_empty());
    }
}