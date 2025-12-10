# Climate Prediction System - Testing Strategy

## 🎯 Testing Philosophy

**Goal**: Achieve > 90% code coverage with comprehensive unit, integration, E2E, and performance testing integrated with ReasoningBank for continuous quality improvement.

**Principles**:
- **Test-First Development**: Write tests before implementation (TDD)
- **Continuous Testing**: Run tests on every commit
- **Automated Validation**: No manual testing for regression
- **Performance Testing**: Every release must pass benchmarks
- **Learning Integration**: ReasoningBank learns from test failures

---

## 🏗️ Test Pyramid

```
           /\
          /  \         E2E Tests (5%)
         /    \        - Full system integration
        /      \       - User journey validation
       /--------\
      /          \     Integration Tests (25%)
     /            \    - Component interaction
    /              \   - API testing
   /----------------\
  /                  \ Unit Tests (70%)
 /                    \- Individual functions
/______________________\- Pure logic testing
```

### Test Distribution
- **Unit Tests**: 70% - Fast, isolated, comprehensive
- **Integration Tests**: 25% - Component interaction, API validation
- **E2E Tests**: 5% - Critical user journeys, full system validation

---

## 🧪 Unit Testing Strategy

### Rust Core Engine Tests

#### 1. Prediction Engine Tests
```rust
// crates/climate-core/tests/prediction_tests.rs
use climate_core::prediction::*;
use approx::assert_relative_eq;

#[cfg(test)]
mod prediction_tests {
    use super::*;

    #[tokio::test]
    async fn test_prediction_accuracy() {
        // Arrange
        let config = Config::test_config();
        let engine = PredictionEngine::new(config).await.unwrap();
        let input = PredictionInput::sample_data();

        // Act
        let output = engine.predict(input).await.unwrap();

        // Assert
        assert!(output.confidence > 0.8, "Confidence too low: {}", output.confidence);
        assert!(output.values.len() > 0, "No predictions returned");
        assert_relative_eq!(output.temperature, 20.0, epsilon = 5.0);

        // Store result in ReasoningBank for learning
        reasoning_bank::store("test/predictions/accuracy", &output).await.ok();
    }

    #[tokio::test]
    async fn test_simd_optimization_performance() {
        let engine = PredictionEngine::new(Config::default()).await.unwrap();
        let large_input = PredictionInput::large_sample(10000);

        let start = std::time::Instant::now();
        let _ = engine.preprocess_simd(large_input).unwrap();
        let duration = start.elapsed();

        assert!(duration.as_millis() < 100, "SIMD processing too slow: {:?}", duration);
    }

    #[tokio::test]
    async fn test_concurrent_predictions() {
        let engine = Arc::new(PredictionEngine::new(Config::default()).await.unwrap());
        let mut handles = vec![];

        for i in 0..1000 {
            let engine_clone = engine.clone();
            handles.push(tokio::spawn(async move {
                let input = PredictionInput::sample_with_id(i);
                engine_clone.predict(input).await
            }));
        }

        let results = futures::future::join_all(handles).await;
        let successful = results.iter().filter(|r| r.is_ok()).count();

        assert_eq!(successful, 1000, "Not all concurrent predictions succeeded");
    }

    #[tokio::test]
    async fn test_error_handling() {
        let engine = PredictionEngine::new(Config::default()).await.unwrap();
        let invalid_input = PredictionInput::invalid();

        let result = engine.predict(invalid_input).await;

        assert!(result.is_err(), "Should reject invalid input");

        // Log error pattern in ReasoningBank
        if let Err(e) = result {
            reasoning_bank::store("test/errors/prediction", &e.to_string()).await.ok();
        }
    }

    #[tokio::test]
    async fn test_memory_usage() {
        let engine = PredictionEngine::new(Config::default()).await.unwrap();
        let initial_memory = get_memory_usage();

        // Run 1000 predictions
        for _ in 0..1000 {
            let input = PredictionInput::sample();
            let _ = engine.predict(input).await.unwrap();
        }

        let final_memory = get_memory_usage();
        let memory_increase = final_memory - initial_memory;

        assert!(memory_increase < 10_000_000, "Memory leak detected: {} bytes", memory_increase);
    }

    #[test]
    fn test_data_serialization() {
        let input = PredictionInput::sample();
        let serialized = serde_json::to_string(&input).unwrap();
        let deserialized: PredictionInput = serde_json::from_str(&serialized).unwrap();

        assert_eq!(input, deserialized);
    }
}
```

#### 2. Data Pipeline Tests
```rust
// crates/climate-data/tests/pipeline_tests.rs
#[cfg(test)]
mod pipeline_tests {
    use climate_data::pipeline::*;

    #[tokio::test]
    async fn test_streaming_throughput() {
        let pipeline = DataPipeline::new(1000).await;
        let start = std::time::Instant::now();

        // Send 10,000 data points
        for i in 0..10000 {
            pipeline.ingest(RawData::sample(i)).await.unwrap();
        }

        let duration = start.elapsed();
        let throughput = 10000.0 / duration.as_secs_f64();

        assert!(throughput > 1000.0, "Throughput too low: {:.0} events/sec", throughput);
    }

    #[tokio::test]
    async fn test_data_validation() {
        let validator = Validator::new();

        let valid_data = WeatherData::valid_sample();
        assert!(validator.validate(&valid_data).await.is_ok());

        let invalid_temp = WeatherData { temperature: 150.0, ..WeatherData::default() };
        assert!(validator.validate(&invalid_temp).await.is_err());

        let invalid_humidity = WeatherData { humidity: 150.0, ..WeatherData::default() };
        assert!(validator.validate(&invalid_humidity).await.is_err());
    }

    #[tokio::test]
    async fn test_anomaly_detection() {
        let processor = StreamProcessor::new();
        let normal_data = vec![20.0, 21.0, 22.0, 20.5, 21.5];
        let anomaly_data = vec![20.0, 21.0, 100.0, 20.5, 21.5]; // 100.0 is anomaly

        let normal_result = processor.detect_anomalies(&normal_data).await.unwrap();
        assert!(normal_result.is_empty());

        let anomaly_result = processor.detect_anomalies(&anomaly_data).await.unwrap();
        assert_eq!(anomaly_result.len(), 1);
        assert_eq!(anomaly_result[0].index, 2);
    }

    #[tokio::test]
    async fn test_parquet_storage() {
        let storage = StorageManager::new().await;
        let batch = ProcessedBatch::sample();

        // Store as Parquet
        storage.store_parquet(&batch).await.unwrap();

        // Retrieve and verify
        let retrieved = storage.load_parquet(&batch.id).await.unwrap();
        assert_eq!(batch.data, retrieved.data);

        // Check compression ratio
        let original_size = bincode::serialize(&batch).unwrap().len();
        let compressed_size = storage.get_parquet_size(&batch.id).await.unwrap();
        let ratio = original_size as f64 / compressed_size as f64;

        assert!(ratio > 5.0, "Compression ratio too low: {:.2}", ratio);
    }
}
```

### Node.js API Tests

#### 1. REST API Tests
```typescript
// packages/api/tests/routes/predictions.test.ts
import request from 'supertest';
import { app } from '../../src/app';
import { createTestToken } from '../helpers/auth';

describe('Predictions API', () => {
  let authToken: string;

  beforeAll(async () => {
    authToken = await createTestToken({ id: 'test-user', role: 'user' });
  });

  describe('POST /predictions', () => {
    it('should create prediction successfully', async () => {
      const response = await request(app)
        .post('/api/v1/predictions')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          location: { latitude: 37.7749, longitude: -122.4194 },
          timeRange: { start: '2025-10-15T00:00:00Z', end: '2025-10-16T00:00:00Z' },
          variables: ['temperature', 'humidity'],
          resolution: 'hourly',
        })
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.data.predictions).toHaveLength(24); // 24 hours
      expect(response.body.metadata.confidence).toBeGreaterThan(0.8);

      // Store successful test in ReasoningBank
      await reasoningBank.store('test/api/predictions/success', response.body);
    });

    it('should reject invalid coordinates', async () => {
      const response = await request(app)
        .post('/api/v1/predictions')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          location: { latitude: 200, longitude: -122.4194 }, // Invalid latitude
          timeRange: { start: '2025-10-15T00:00:00Z', end: '2025-10-16T00:00:00Z' },
          variables: ['temperature'],
          resolution: 'hourly',
        })
        .expect(400);

      expect(response.body.error).toContain('latitude');
    });

    it('should enforce rate limiting', async () => {
      // Make 101 requests (limit is 100 per minute)
      const promises = Array(101).fill(null).map(() =>
        request(app)
          .post('/api/v1/predictions')
          .set('Authorization', `Bearer ${authToken}`)
          .send({
            location: { latitude: 37.7749, longitude: -122.4194 },
            timeRange: { start: '2025-10-15T00:00:00Z', end: '2025-10-16T00:00:00Z' },
            variables: ['temperature'],
            resolution: 'hourly',
          })
      );

      const results = await Promise.all(promises);
      const rateLimited = results.filter(r => r.status === 429);

      expect(rateLimited.length).toBeGreaterThan(0);
    });

    it('should cache predictions', async () => {
      const request1Start = Date.now();
      const response1 = await request(app)
        .post('/api/v1/predictions')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          location: { latitude: 37.7749, longitude: -122.4194 },
          timeRange: { start: '2025-10-15T00:00:00Z', end: '2025-10-16T00:00:00Z' },
          variables: ['temperature'],
          resolution: 'hourly',
        });
      const request1Time = Date.now() - request1Start;

      // Second identical request should be faster (cached)
      const request2Start = Date.now();
      const response2 = await request(app)
        .post('/api/v1/predictions')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          location: { latitude: 37.7749, longitude: -122.4194 },
          timeRange: { start: '2025-10-15T00:00:00Z', end: '2025-10-16T00:00:00Z' },
          variables: ['temperature'],
          resolution: 'hourly',
        });
      const request2Time = Date.now() - request2Start;

      expect(response1.body).toEqual(response2.body);
      expect(request2Time).toBeLessThan(request1Time / 2);
    });
  });

  describe('GET /predictions/:id', () => {
    it('should retrieve prediction by ID', async () => {
      // Create prediction first
      const createResponse = await request(app)
        .post('/api/v1/predictions')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          location: { latitude: 37.7749, longitude: -122.4194 },
          timeRange: { start: '2025-10-15T00:00:00Z', end: '2025-10-16T00:00:00Z' },
          variables: ['temperature'],
          resolution: 'hourly',
        });

      const predictionId = createResponse.body.data.id;

      // Retrieve prediction
      const response = await request(app)
        .get(`/api/v1/predictions/${predictionId}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body.id).toBe(predictionId);
    });

    it('should return 404 for non-existent prediction', async () => {
      await request(app)
        .get('/api/v1/predictions/non-existent-id')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(404);
    });
  });
});
```

#### 2. GraphQL API Tests
```typescript
// packages/api/tests/graphql/predictions.test.ts
import { ApolloServer } from 'apollo-server-express';
import { createTestClient } from 'apollo-server-testing';
import { schema } from '../../src/graphql/schema';

describe('GraphQL Predictions', () => {
  let server: ApolloServer;
  let query: any, mutate: any;

  beforeAll(() => {
    server = new ApolloServer({ schema });
    ({ query, mutate } = createTestClient(server));
  });

  it('should query prediction', async () => {
    const PREDICT_QUERY = `
      query Predict($location: String!, $time: String!) {
        predict(location: $location, time: $time) {
          id
          timestamp
          temperature
          confidence
          modelVersion
        }
      }
    `;

    const response = await query({
      query: PREDICT_QUERY,
      variables: {
        location: '37.7749,-122.4194',
        time: '2025-10-15T12:00:00Z',
      },
    });

    expect(response.data.predict).toBeDefined();
    expect(response.data.predict.confidence).toBeGreaterThan(0.8);
  });

  it('should submit feedback via mutation', async () => {
    const FEEDBACK_MUTATION = `
      mutation ProvideFeedback($predictionId: String!, $actualValue: Float!, $rating: Float!) {
        provideFeedback(predictionId: $predictionId, actualValue: $actualValue, rating: $rating)
      }
    `;

    const response = await mutate({
      mutation: FEEDBACK_MUTATION,
      variables: {
        predictionId: 'test-prediction-id',
        actualValue: 22.5,
        rating: 4.5,
      },
    });

    expect(response.data.provideFeedback).toBe('Feedback recorded successfully');
  });

  it('should subscribe to prediction updates', async () => {
    // Test subscription functionality
    const PREDICTION_SUBSCRIPTION = `
      subscription PredictionCreated {
        predictionCreated {
          id
          temperature
          confidence
        }
      }
    `;

    // Create subscription
    const subscription = server.executeOperation({
      query: PREDICTION_SUBSCRIPTION,
    });

    // Trigger prediction creation
    await mutate({
      mutation: `
        mutation CreatePrediction($input: PredictionInput!) {
          createPrediction(input: $input) { id }
        }
      `,
      variables: {
        input: { location: '37.7749,-122.4194', time: '2025-10-15T12:00:00Z' },
      },
    });

    // Verify subscription received update
    // (Implementation depends on subscription testing library)
  });
});
```

#### 3. WebSocket Tests
```typescript
// packages/api/tests/websocket/server.test.ts
import WebSocket from 'ws';
import { WebSocketServer } from '../../src/websocket/server';

describe('WebSocket Server', () => {
  let server: WebSocketServer;
  let client: WebSocket;

  beforeAll((done) => {
    server = new WebSocketServer(httpServer);
    client = new WebSocket('ws://localhost:3000');
    client.on('open', done);
  });

  afterAll(() => {
    client.close();
  });

  it('should connect successfully', (done) => {
    expect(client.readyState).toBe(WebSocket.OPEN);
    done();
  });

  it('should receive prediction updates', (done) => {
    client.on('message', (data: WebSocket.Data) => {
      const message = JSON.parse(data.toString());

      expect(message.type).toBe('prediction_update');
      expect(message.data.temperature).toBeDefined();
      done();
    });

    // Subscribe to location
    client.send(JSON.stringify({
      type: 'subscribe',
      location: { latitude: 37.7749, longitude: -122.4194 },
    }));
  });

  it('should handle disconnections gracefully', (done) => {
    client.on('close', (code, reason) => {
      expect(code).toBe(1000); // Normal closure
      done();
    });

    client.close();
  });
});
```

### Python ML Tests

#### 1. Model Training Tests
```python
# ml-research/tests/test_training.py
import pytest
import torch
import numpy as np
from training.trainer import ClimateTrainer
from models.climate_model import ClimateNet

class TestModelTraining:
    @pytest.fixture
    def trainer(self):
        model = ClimateNet(input_dim=10, hidden_dim=64, output_dim=1)
        return ClimateTrainer(model, learning_rate=0.001)

    def test_training_convergence(self, trainer):
        """Test that model converges during training"""
        X_train = torch.randn(1000, 10)
        y_train = torch.randn(1000, 1)

        initial_loss = trainer.evaluate(X_train, y_train)

        # Train for 100 epochs
        trainer.train(X_train, y_train, epochs=100)

        final_loss = trainer.evaluate(X_train, y_train)

        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"
        assert final_loss < 0.1, f"Final loss too high: {final_loss}"

    def test_overfitting_prevention(self, trainer):
        """Test that regularization prevents overfitting"""
        X_train = torch.randn(100, 10)
        y_train = torch.randn(100, 1)
        X_val = torch.randn(50, 10)
        y_val = torch.randn(50, 1)

        trainer.train(X_train, y_train, X_val, y_val, epochs=200, early_stopping=True)

        train_loss = trainer.evaluate(X_train, y_train)
        val_loss = trainer.evaluate(X_val, y_val)

        # Validation loss should not be much worse than training loss
        assert val_loss < train_loss * 1.5, f"Overfitting detected: train={train_loss}, val={val_loss}"

    def test_hyperparameter_tuning(self, trainer):
        """Test hyperparameter optimization"""
        from training.hyperparameter_tuner import HyperparameterTuner

        X_train = torch.randn(1000, 10)
        y_train = torch.randn(1000, 1)

        tuner = HyperparameterTuner()
        best_params = tuner.tune(X_train, y_train, n_trials=10)

        assert 'learning_rate' in best_params
        assert 'hidden_dim' in best_params
        assert best_params['learning_rate'] > 0

        # Store best params in ReasoningBank
        import reasoning_bank
        reasoning_bank.store('ml/hyperparameters/best', str(best_params))

    def test_model_export_onnx(self, trainer):
        """Test ONNX model export"""
        X_sample = torch.randn(1, 10)

        # Export to ONNX
        onnx_path = 'test_model.onnx'
        trainer.export_onnx(onnx_path, X_sample)

        # Verify ONNX file exists and is valid
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)

        # Clean up
        os.remove(onnx_path)
```

---

## 🔗 Integration Testing Strategy

### Component Integration Tests

#### 1. Rust Engine + Node.js API Integration
```typescript
// tests/integration/engine-api.test.ts
import { ClimateEngine } from '../../packages/api/src/engine';
import { PredictionInput } from '../../crates/climate-core/bindings';

describe('Rust Engine + Node.js API Integration', () => {
  let engine: ClimateEngine;

  beforeAll(async () => {
    engine = new ClimateEngine();
    await engine.initialize();
  });

  it('should make prediction via FFI', async () => {
    const input: PredictionInput = {
      location: { latitude: 37.7749, longitude: -122.4194 },
      timestamp: Date.now(),
      variables: ['temperature', 'humidity'],
    };

    const start = Date.now();
    const result = await engine.predict(input);
    const duration = Date.now() - start;

    expect(result.temperature).toBeDefined();
    expect(result.confidence).toBeGreaterThan(0.8);
    expect(duration).toBeLessThan(100); // < 100ms
  });

  it('should handle concurrent predictions', async () => {
    const promises = Array(100).fill(null).map((_, i) => {
      const input: PredictionInput = {
        location: { latitude: 37.7749, longitude: -122.4194 + i * 0.01 },
        timestamp: Date.now(),
        variables: ['temperature'],
      };
      return engine.predict(input);
    });

    const results = await Promise.all(promises);

    expect(results).toHaveLength(100);
    expect(results.every(r => r.confidence > 0)).toBe(true);
  });
});
```

#### 2. Data Pipeline + Storage Integration
```rust
// tests/integration/pipeline_storage.rs
#[tokio::test]
async fn test_end_to_end_data_flow() {
    // Initialize components
    let pipeline = DataPipeline::new(1000).await;
    let storage = StorageManager::new().await;

    // Ingest data
    let raw_data = generate_test_data(1000);
    for data in raw_data {
        pipeline.ingest(data).await.unwrap();
    }

    // Wait for processing
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Verify storage
    let stored_batches = storage.list_batches().await.unwrap();
    assert!(stored_batches.len() > 0, "No batches stored");

    // Verify data integrity
    for batch in stored_batches {
        let retrieved = storage.load_parquet(&batch.id).await.unwrap();
        assert_eq!(batch.record_count, retrieved.len());
    }
}
```

#### 3. ML Model + Inference Engine Integration
```python
# tests/integration/test_ml_inference.py
def test_pytorch_to_onnx_to_rust():
    """Test complete ML pipeline: PyTorch -> ONNX -> Rust inference"""
    # Train PyTorch model
    model = ClimateNet(input_dim=10, hidden_dim=64, output_dim=1)
    trainer = ClimateTrainer(model)
    X_train = torch.randn(1000, 10)
    y_train = torch.randn(1000, 1)
    trainer.train(X_train, y_train, epochs=50)

    # Export to ONNX
    onnx_path = 'climate_model.onnx'
    trainer.export_onnx(onnx_path, torch.randn(1, 10))

    # Load in Rust engine (via FFI)
    from rust_bindings import ClimateEngine
    engine = ClimateEngine(onnx_path)

    # Test prediction
    input_data = np.random.randn(1, 10).astype(np.float32)

    # PyTorch prediction
    with torch.no_grad():
        pytorch_pred = model(torch.from_numpy(input_data)).numpy()

    # Rust prediction
    rust_pred = engine.predict(input_data)

    # Results should be very close (within numerical precision)
    assert np.allclose(pytorch_pred, rust_pred, rtol=1e-3), \
        f"Predictions differ: PyTorch={pytorch_pred}, Rust={rust_pred}"
```

#### 4. ReasoningBank Learning Loop Integration
```typescript
// tests/integration/reasoning-bank.test.ts
describe('ReasoningBank Learning Loop', () => {
  it('should learn from prediction feedback', async () => {
    const engine = new ClimateEngine();

    // Make initial prediction
    const prediction1 = await engine.predict({
      location: { latitude: 37.7749, longitude: -122.4194 },
      timestamp: Date.now(),
      variables: ['temperature'],
    });

    const initialAccuracy = prediction1.metadata.accuracy;

    // Provide feedback for 100 predictions
    for (let i = 0; i < 100; i++) {
      const pred = await engine.predict({
        location: { latitude: 37.7749, longitude: -122.4194 },
        timestamp: Date.now() + i * 3600000,
        variables: ['temperature'],
      });

      // Simulate feedback with actual values
      await engine.provideFeedback({
        predictionId: pred.id,
        actualValue: pred.temperature + (Math.random() - 0.5) * 2,
        rating: 4.0,
      });
    }

    // Trigger learning update
    await reasoningBank.learn('climate/predictions');

    // Make new prediction
    const prediction2 = await engine.predict({
      location: { latitude: 37.7749, longitude: -122.4194 },
      timestamp: Date.now(),
      variables: ['temperature'],
    });

    const finalAccuracy = prediction2.metadata.accuracy;

    // Accuracy should improve
    expect(finalAccuracy).toBeGreaterThan(initialAccuracy);
  });
});
```

---

## 🌐 End-to-End Testing Strategy

### Critical User Journeys

#### 1. Complete Prediction Workflow
```typescript
// tests/e2e/prediction-workflow.test.ts
import { test, expect } from '@playwright/test';

test.describe('Prediction Workflow', () => {
  test('user can create and view prediction', async ({ page }) => {
    // Login
    await page.goto('http://localhost:3000/login');
    await page.fill('[data-testid="email"]', 'test@example.com');
    await page.fill('[data-testid="password"]', 'password123');
    await page.click('[data-testid="login-button"]');

    // Navigate to predictions
    await page.goto('http://localhost:3000/predictions');
    await page.waitForSelector('[data-testid="prediction-form"]');

    // Fill prediction form
    await page.fill('[data-testid="latitude"]', '37.7749');
    await page.fill('[data-testid="longitude"]', '-122.4194');
    await page.selectOption('[data-testid="resolution"]', 'hourly');
    await page.check('[data-testid="variable-temperature"]');

    // Submit prediction
    await page.click('[data-testid="predict-button"]');

    // Wait for results
    await page.waitForSelector('[data-testid="prediction-results"]', { timeout: 10000 });

    // Verify results
    const temperature = await page.textContent('[data-testid="predicted-temperature"]');
    expect(parseFloat(temperature!)).toBeGreaterThan(-50);
    expect(parseFloat(temperature!)).toBeLessThan(50);

    const confidence = await page.textContent('[data-testid="confidence"]');
    expect(parseFloat(confidence!)).toBeGreaterThan(0.8);
  });

  test('user can provide feedback', async ({ page }) => {
    // ... (login and create prediction as above)

    // Provide feedback
    await page.click('[data-testid="provide-feedback"]');
    await page.fill('[data-testid="actual-value"]', '22.5');
    await page.click('[data-testid="rating-4-stars"]');
    await page.click('[data-testid="submit-feedback"]');

    // Verify feedback submitted
    await page.waitForSelector('[data-testid="feedback-success"]');
    expect(await page.textContent('[data-testid="feedback-success"]'))
      .toContain('Thank you for your feedback');
  });
});
```

#### 2. Real-Time Updates via WebSocket
```typescript
// tests/e2e/realtime-updates.test.ts
test('user receives real-time prediction updates', async ({ page }) => {
  await page.goto('http://localhost:3000/realtime');

  // Subscribe to location
  await page.fill('[data-testid="latitude"]', '37.7749');
  await page.fill('[data-testid="longitude"]', '-122.4194');
  await page.click('[data-testid="subscribe-button"]');

  // Wait for first update
  await page.waitForSelector('[data-testid="prediction-update"]', { timeout: 10000 });

  // Verify updates continue
  let updateCount = 0;
  page.on('domcontentloaded', () => {
    const updates = page.locator('[data-testid="prediction-update"]');
    updateCount = updates.count();
  });

  await page.waitForTimeout(30000); // Wait 30 seconds
  expect(updateCount).toBeGreaterThan(5); // Should receive multiple updates
});
```

---

## ⚡ Performance Testing Strategy

### Load Testing with k6

#### 1. API Load Test
```javascript
// tests/performance/api-load.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

const errorRate = new Rate('errors');

export const options = {
  stages: [
    { duration: '2m', target: 100 },   // Ramp up to 100 users
    { duration: '5m', target: 100 },   // Stay at 100 users
    { duration: '2m', target: 1000 },  // Ramp up to 1000 users
    { duration: '5m', target: 1000 },  // Stay at 1000 users
    { duration: '2m', target: 0 },     // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<200', 'p(99)<500'], // 95% < 200ms, 99% < 500ms
    http_req_failed: ['rate<0.01'],                 // Error rate < 1%
    errors: ['rate<0.05'],
  },
};

export default function () {
  const payload = JSON.stringify({
    location: { latitude: 37.7749, longitude: -122.4194 },
    timeRange: { start: '2025-10-15T00:00:00Z', end: '2025-10-16T00:00:00Z' },
    variables: ['temperature', 'humidity'],
    resolution: 'hourly',
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer test-token',
    },
  };

  const response = http.post('http://localhost:3000/api/v1/predictions', payload, params);

  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 200ms': (r) => r.timings.duration < 200,
    'confidence > 0.8': (r) => JSON.parse(r.body).metadata.confidence > 0.8,
  }) || errorRate.add(1);

  sleep(1);
}
```

#### 2. Rust Engine Benchmark
```rust
// crates/climate-core/benches/prediction_bench.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use climate_core::prediction::*;

fn prediction_benchmark(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let engine = rt.block_on(async {
        PredictionEngine::new(Config::default()).await.unwrap()
    });

    let mut group = c.benchmark_group("predictions");

    for size in [10, 100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let input = PredictionInput::large_sample(size);
            b.to_async(&rt).iter(|| async {
                black_box(engine.predict(input.clone()).await.unwrap())
            });
        });
    }

    group.finish();
}

criterion_group!(benches, prediction_benchmark);
criterion_main!(benches);
```

### Performance Acceptance Criteria

| Metric | Target | Critical |
|--------|--------|----------|
| API Response Time (p95) | < 200ms | < 500ms |
| API Response Time (p99) | < 500ms | < 1000ms |
| Throughput | > 10K req/s | > 5K req/s |
| Error Rate | < 0.1% | < 1% |
| CPU Usage (avg) | < 50% | < 80% |
| Memory Usage | < 2GB | < 4GB |
| Database Query Time | < 100ms | < 500ms |

---

## 🔒 Security Testing Strategy

### 1. Authentication & Authorization Tests
```typescript
// tests/security/auth.test.ts
describe('Security: Authentication & Authorization', () => {
  it('should reject requests without token', async () => {
    const response = await request(app)
      .post('/api/v1/predictions')
      .send({ /* data */ })
      .expect(401);

    expect(response.body.error).toContain('Authentication required');
  });

  it('should reject expired tokens', async () => {
    const expiredToken = createExpiredToken();

    const response = await request(app)
      .post('/api/v1/predictions')
      .set('Authorization', `Bearer ${expiredToken}`)
      .send({ /* data */ })
      .expect(401);
  });

  it('should enforce role-based access control', async () => {
    const userToken = createToken({ role: 'user' });

    const response = await request(app)
      .post('/api/v1/admin/models')
      .set('Authorization', `Bearer ${userToken}`)
      .send({ /* data */ })
      .expect(403);
  });
});
```

### 2. SQL Injection Tests
```typescript
// tests/security/sql-injection.test.ts
describe('Security: SQL Injection', () => {
  it('should prevent SQL injection in prediction queries', async () => {
    const maliciousInput = "'; DROP TABLE predictions; --";

    const response = await request(app)
      .get(`/api/v1/predictions?location=${maliciousInput}`)
      .set('Authorization', `Bearer ${validToken}`)
      .expect(400);

    // Verify table still exists
    const tableExists = await db.$queryRaw`SELECT EXISTS (
      SELECT FROM pg_tables WHERE tablename = 'predictions'
    )`;
    expect(tableExists).toBe(true);
  });
});
```

### 3. Rate Limiting Tests
```typescript
// tests/security/rate-limiting.test.ts
describe('Security: Rate Limiting', () => {
  it('should enforce rate limits per user', async () => {
    const token = createToken({ id: 'test-user' });

    // Make 101 requests (limit is 100)
    const promises = Array(101).fill(null).map(() =>
      request(app)
        .post('/api/v1/predictions')
        .set('Authorization', `Bearer ${token}`)
        .send({ /* data */ })
    );

    const responses = await Promise.all(promises);
    const rateLimited = responses.filter(r => r.status === 429);

    expect(rateLimited.length).toBeGreaterThan(0);
  });
});
```

---

## 📊 Test Coverage Requirements

### Coverage Targets
- **Unit Tests**: > 90% line coverage
- **Integration Tests**: > 80% component coverage
- **E2E Tests**: 100% critical path coverage
- **Security Tests**: 100% attack surface coverage

### Coverage Tracking
```bash
# Rust coverage
cargo tarpaulin --out Html --output-dir coverage

# Node.js coverage
npm run test:coverage

# Python coverage
pytest --cov=ml_research --cov-report=html
```

---

## 🔄 Continuous Integration

### CI Pipeline
```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]

jobs:
  rust-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions-rs/toolchain@v1
      - run: cargo test --all-features
      - run: cargo tarpaulin --out Xml
      - uses: codecov/codecov-action@v3

  node-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
      - run: npm ci
      - run: npm run test:coverage
      - uses: codecov/codecov-action@v3

  python-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
      - run: pip install -r requirements.txt
      - run: pytest --cov --cov-report=xml
      - uses: codecov/codecov-action@v3

  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker-compose up -d
      - run: npm run test:e2e
      - run: docker-compose down

  performance-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker-compose up -d
      - run: k6 run tests/performance/api-load.js
```

---

## 📈 Test Metrics & Reporting

### ReasoningBank Integration
```bash
# Store test results in ReasoningBank
npx claude-flow@alpha hooks post-task --task-id "test-run-$(date +%s)"

# Analyze test patterns
npx claude-flow@alpha hooks neural-train --pattern-type "test-failures" --training-data "tests/results.json"

# Generate test report
npx claude-flow@alpha hooks session-end --export-metrics true
```

### Test Report Dashboard
```typescript
// Generate HTML test report
import { generateReport } from './reporting';

const testResults = {
  unit: { passed: 450, failed: 2, coverage: 92 },
  integration: { passed: 85, failed: 0, coverage: 83 },
  e2e: { passed: 25, failed: 1, coverage: 100 },
  performance: { p95: 180, p99: 420, throughput: 12000 },
};

generateReport(testResults, 'test-report.html');
```

---

*Generated by Claude Code with SPARC methodology*
*ReasoningBank enabled for continuous test quality improvement*
