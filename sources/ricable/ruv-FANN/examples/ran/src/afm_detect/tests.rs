#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Result, Tensor};

    fn create_test_detector() -> Result<AFMDetector> {
        let device = Device::Cpu;
        AFMDetector::new(32, 8, device)
    }

    fn create_test_data(batch_size: usize, feature_dim: usize, device: &Device) -> Result<Tensor> {
        Tensor::randn(0.0, 1.0, &[batch_size, feature_dim], device)
    }

    #[test]
    fn test_afm_detector_creation() -> Result<()> {
        let detector = create_test_detector()?;
        assert!(true); // If we get here, creation succeeded
        Ok(())
    }

    #[test]
    fn test_basic_detection() -> Result<()> {
        let detector = create_test_detector()?;
        let device = Device::Cpu;
        let data = create_test_data(5, 32, &device)?;
        
        let result = detector.detect(&data, DetectionMode::Combined, None)?;
        
        assert!(result.score >= 0.0 && result.score <= 1.0);
        assert!(!result.method_scores.is_empty());
        assert!(result.method_scores.contains_key("autoencoder"));
        assert!(result.method_scores.contains_key("vae"));
        assert!(result.method_scores.contains_key("ocsvm"));
        
        Ok(())
    }

    #[test]
    fn test_detection_modes() -> Result<()> {
        let detector = create_test_detector()?;
        let device = Device::Cpu;
        let data = create_test_data(3, 32, &device)?;
        
        let modes = vec![
            DetectionMode::KpiKqi,
            DetectionMode::HardwareDegradation,
            DetectionMode::ThermalPower,
            DetectionMode::Combined,
        ];
        
        for mode in modes {
            let result = detector.detect(&data, mode, None)?;
            assert!(result.score >= 0.0 && result.score <= 1.0);
        }
        
        Ok(())
    }

    #[test]
    fn test_with_history() -> Result<()> {
        let detector = create_test_detector()?;
        let device = Device::Cpu;
        let data = create_test_data(3, 32, &device)?;
        let history = create_test_data(10, 32, &device)?;
        
        let result = detector.detect(&data, DetectionMode::Combined, Some(&history))?;
        
        assert!(result.score >= 0.0 && result.score <= 1.0);
        assert!(result.failure_probability.is_some());
        
        Ok(())
    }

    #[test]
    fn test_anomaly_type_classification() -> Result<()> {
        let detector = create_test_detector()?;
        let device = Device::Cpu;
        
        // Create high anomaly score data
        let anomaly_data = Tensor::randn(0.0, 5.0, &[3, 32], &device)?;
        
        let result = detector.detect(&anomaly_data, DetectionMode::Combined, None)?;
        
        if result.score > 0.7 {
            assert!(result.anomaly_type.is_some());
        }
        
        Ok(())
    }

    #[test]
    fn test_confidence_intervals() -> Result<()> {
        let detector = create_test_detector()?;
        let device = Device::Cpu;
        let data = create_test_data(3, 32, &device)?;
        
        let result = detector.detect(&data, DetectionMode::Combined, None)?;
        
        assert!(result.confidence.0 <= result.confidence.1);
        
        Ok(())
    }

    #[test]
    fn test_training() -> Result<()> {
        let mut detector = create_test_detector()?;
        let device = Device::Cpu;
        let normal_data = create_test_data(20, 32, &device)?;
        
        // This should not panic
        detector.train_on_normal(&normal_data, 5, 0.01)?;
        
        Ok(())
    }

    #[test]
    fn test_autoencoder_component() -> Result<()> {
        let device = Device::Cpu;
        let var_map = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &device);
        
        let autoencoder = autoencoder::AutoencoderDetector::new(32, 8, vb)?;
        let data = create_test_data(5, 32, &device)?;
        
        let encoded = autoencoder.encode(&data)?;
        assert_eq!(encoded.dims(), &[5, 8]);
        
        let decoded = autoencoder.decode(&encoded)?;
        assert_eq!(decoded.dims(), &[5, 32]);
        
        let score = autoencoder.anomaly_score(&data)?;
        assert!(score >= 0.0);
        
        Ok(())
    }

    #[test]
    fn test_vae_component() -> Result<()> {
        let device = Device::Cpu;
        let var_map = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &device);
        
        let vae = vae::VariationalDetector::new(32, 8, vb)?;
        let data = create_test_data(5, 32, &device)?;
        
        let (mu, logvar) = vae.encode(&data)?;
        assert_eq!(mu.dims(), &[5, 8]);
        assert_eq!(logvar.dims(), &[5, 8]);
        
        let z = vae.reparameterize(&mu, &logvar)?;
        assert_eq!(z.dims(), &[5, 8]);
        
        let decoded = vae.decode(&z)?;
        assert_eq!(decoded.dims(), &[5, 32]);
        
        let score = vae.anomaly_score(&data)?;
        assert!(score >= 0.0);
        
        Ok(())
    }

    #[test]
    fn test_ocsvm_component() -> Result<()> {
        let device = Device::Cpu;
        let var_map = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &device);
        
        let ocsvm = ocsvm::OneClassSVMDetector::new(32, 8, vb)?;
        let data = create_test_data(5, 32, &device)?;
        
        let features = ocsvm.extract_features(&data)?;
        assert_eq!(features.dims()[0], 5);
        assert_eq!(features.dims()[1], 8);
        
        let distance = ocsvm.distance_from_center(&data)?;
        assert_eq!(distance.dims()[0], 5);
        
        let score = ocsvm.anomaly_score(&data)?;
        assert!(score >= 0.0 && score <= 1.0);
        
        Ok(())
    }

    #[test]
    fn test_threshold_learner() -> Result<()> {
        let device = Device::Cpu;
        let var_map = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &device);
        
        let threshold_learner = threshold::DynamicThresholdLearner::new(vb)?;
        let score = 0.5;
        
        let threshold = threshold_learner.get_threshold(&score, DetectionMode::Combined)?;
        assert!(threshold > 0.0 && threshold <= 1.0);
        
        Ok(())
    }

    #[test]
    fn test_contrastive_learner() -> Result<()> {
        let device = Device::Cpu;
        let var_map = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &device);
        
        let contrastive = contrastive::ContrastiveLearner::new(8, vb)?;
        let data = create_test_data(5, 8, &device)?;
        
        let encoded = contrastive.encode(&data)?;
        assert_eq!(encoded.dims(), &[5, 8]);
        
        let projected = contrastive.project(&encoded)?;
        assert_eq!(projected.dims()[0], 5);
        
        let (aug1, aug2) = contrastive.augment(&data)?;
        assert_eq!(aug1.dims(), data.dims());
        assert_eq!(aug2.dims(), data.dims());
        
        Ok(())
    }

    #[test]
    fn test_failure_predictor() -> Result<()> {
        let device = Device::Cpu;
        let var_map = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &device);
        
        let predictor = predictor::FailurePredictor::new(32, vb)?;
        let current_data = create_test_data(3, 32, &device)?;
        let history = create_test_data(10, 32, &device)?;
        
        let failure_prob = predictor.predict_failure(&current_data, &history, 48)?;
        assert!(failure_prob >= 0.0 && failure_prob <= 1.0);
        
        Ok(())
    }

    #[test]
    fn test_edge_cases() -> Result<()> {
        let detector = create_test_detector()?;
        let device = Device::Cpu;
        
        // Test with very small data
        let small_data = Tensor::zeros(&[1, 32], &device)?;
        let result = detector.detect(&small_data, DetectionMode::Combined, None)?;
        assert!(result.score >= 0.0 && result.score <= 1.0);
        
        // Test with constant data
        let constant_data = Tensor::ones(&[3, 32], &device)?;
        let result = detector.detect(&constant_data, DetectionMode::Combined, None)?;
        assert!(result.score >= 0.0 && result.score <= 1.0);
        
        Ok(())
    }

    #[test]
    fn test_score_consistency() -> Result<()> {
        let detector = create_test_detector()?;
        let device = Device::Cpu;
        let data = create_test_data(5, 32, &device)?;
        
        // Multiple runs should give similar results
        let result1 = detector.detect(&data, DetectionMode::Combined, None)?;
        let result2 = detector.detect(&data, DetectionMode::Combined, None)?;
        
        let score_diff = (result1.score - result2.score).abs();
        assert!(score_diff < 0.1); // Should be reasonably consistent
        
        Ok(())
    }

    #[test]
    fn test_failure_mode_detection() -> Result<()> {
        let device = Device::Cpu;
        let var_map = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &device);
        
        let predictor = predictor::FailurePredictor::new(32, vb)?;
        let current_data = create_test_data(3, 32, &device)?;
        let history = create_test_data(10, 32, &device)?;
        
        let failure_modes = predictor.predict_failure_modes(&current_data, &history)?;
        
        assert_eq!(failure_modes.len(), 4);
        assert!(failure_modes.iter().any(|(mode, _)| mode == "Thermal"));
        assert!(failure_modes.iter().any(|(mode, _)| mode == "Mechanical"));
        assert!(failure_modes.iter().any(|(mode, _)| mode == "Electrical"));
        assert!(failure_modes.iter().any(|(mode, _)| mode == "Software"));
        
        for (_, prob) in failure_modes {
            assert!(prob >= 0.0 && prob <= 1.0);
        }
        
        Ok(())
    }
}