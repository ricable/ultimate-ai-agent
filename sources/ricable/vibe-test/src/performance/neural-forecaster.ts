/**
 * Neural Forecaster - LSTM and N-BEATS based traffic prediction
 * Optimized for WASM runtime with memory-efficient implementations
 */

import {
  TimeSeries,
  TrafficForecast,
  PredictionPoint,
  CellMetrics,
} from '../core/types.js';
import { exponentialMovingAverage, percentile } from '../utils/math.js';
import { createLogger } from '../utils/logger.js';

const logger = createLogger('NeuralForecaster');

/**
 * Neural Forecaster Configuration
 */
export interface ForecastConfig {
  horizonMs: number;          // How far to predict (default: 15 minutes)
  resolutionMs: number;       // Prediction granularity (default: 1 minute)
  minHistoryPoints: number;   // Minimum data points needed
  modelType: 'lstm' | 'nbeats' | 'ensemble';
  confidenceLevel: number;    // For prediction intervals (0.95)
  seasonalityPeriod: number;  // For diurnal patterns (24 hours in ms)
}

const DEFAULT_CONFIG: ForecastConfig = {
  horizonMs: 15 * 60 * 1000,        // 15 minutes
  resolutionMs: 60 * 1000,          // 1 minute
  minHistoryPoints: 60,
  modelType: 'ensemble',
  confidenceLevel: 0.95,
  seasonalityPeriod: 24 * 60 * 60 * 1000, // 24 hours
};

/**
 * LSTM Cell - Simplified implementation for edge inference
 */
class LSTMCell {
  private hiddenSize: number;
  private inputSize: number;

  // Weight matrices (simplified - in production these would be loaded from trained model)
  private Wf: Float32Array; // Forget gate
  private Wi: Float32Array; // Input gate
  private Wc: Float32Array; // Cell state
  private Wo: Float32Array; // Output gate

  // Biases
  private bf: Float32Array;
  private bi: Float32Array;
  private bc: Float32Array;
  private bo: Float32Array;

  // States
  private h: Float32Array;  // Hidden state
  private c: Float32Array;  // Cell state

  constructor(inputSize: number, hiddenSize: number) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;

    const totalSize = inputSize + hiddenSize;

    // Initialize weights with Xavier initialization
    this.Wf = this.initWeights(hiddenSize, totalSize);
    this.Wi = this.initWeights(hiddenSize, totalSize);
    this.Wc = this.initWeights(hiddenSize, totalSize);
    this.Wo = this.initWeights(hiddenSize, totalSize);

    this.bf = new Float32Array(hiddenSize).fill(1); // Forget bias starts at 1
    this.bi = new Float32Array(hiddenSize);
    this.bc = new Float32Array(hiddenSize);
    this.bo = new Float32Array(hiddenSize);

    this.h = new Float32Array(hiddenSize);
    this.c = new Float32Array(hiddenSize);
  }

  private initWeights(rows: number, cols: number): Float32Array {
    const scale = Math.sqrt(2 / (rows + cols));
    const weights = new Float32Array(rows * cols);
    for (let i = 0; i < weights.length; i++) {
      weights[i] = (Math.random() * 2 - 1) * scale;
    }
    return weights;
  }

  forward(input: Float32Array): Float32Array {
    // Concatenate input and hidden state
    const concat = new Float32Array(this.inputSize + this.hiddenSize);
    concat.set(input);
    concat.set(this.h, this.inputSize);

    // Gates
    const ft = this.sigmoid(this.matVecAdd(this.Wf, concat, this.bf));
    const it = this.sigmoid(this.matVecAdd(this.Wi, concat, this.bi));
    const ct = this.tanh(this.matVecAdd(this.Wc, concat, this.bc));
    const ot = this.sigmoid(this.matVecAdd(this.Wo, concat, this.bo));

    // Update cell state
    for (let i = 0; i < this.hiddenSize; i++) {
      this.c[i] = ft[i] * this.c[i] + it[i] * ct[i];
    }

    // Update hidden state
    for (let i = 0; i < this.hiddenSize; i++) {
      this.h[i] = ot[i] * Math.tanh(this.c[i]);
    }

    return new Float32Array(this.h);
  }

  reset(): void {
    this.h.fill(0);
    this.c.fill(0);
  }

  private matVecAdd(W: Float32Array, x: Float32Array, b: Float32Array): Float32Array {
    const result = new Float32Array(this.hiddenSize);
    for (let i = 0; i < this.hiddenSize; i++) {
      result[i] = b[i];
      for (let j = 0; j < x.length; j++) {
        result[i] += W[i * x.length + j] * x[j];
      }
    }
    return result;
  }

  private sigmoid(x: Float32Array): Float32Array {
    return x.map((v) => 1 / (1 + Math.exp(-v)));
  }

  private tanh(x: Float32Array): Float32Array {
    return x.map((v) => Math.tanh(v));
  }
}

/**
 * N-BEATS Block - Neural Basis Expansion Analysis
 */
class NBeatsBlock {
  private inputSize: number;
  private thetaSize: number;
  private layers: number;
  private weights: Float32Array[];

  constructor(inputSize: number, hiddenSize: number = 32, thetaSize: number = 4) {
    this.inputSize = inputSize;
    this.thetaSize = thetaSize;
    this.layers = 3;

    // Initialize simple MLP weights
    this.weights = [];
    let prevSize = inputSize;
    for (let i = 0; i < this.layers; i++) {
      const w = new Float32Array(hiddenSize * prevSize);
      for (let j = 0; j < w.length; j++) {
        w[j] = (Math.random() * 2 - 1) * Math.sqrt(2 / prevSize);
      }
      this.weights.push(w);
      prevSize = hiddenSize;
    }
  }

  forward(input: Float32Array, horizonLength: number): { backcast: Float32Array; forecast: Float32Array } {
    // Simple feedforward through layers
    let x = input;
    const hiddenSize = 32;

    for (const w of this.weights) {
      const newX = new Float32Array(hiddenSize);
      for (let i = 0; i < hiddenSize; i++) {
        for (let j = 0; j < x.length; j++) {
          newX[i] += w[i * x.length + j] * x[j];
        }
        newX[i] = Math.max(0, newX[i]); // ReLU
      }
      x = newX;
    }

    // Generate basis expansion coefficients
    const theta = new Float32Array(this.thetaSize);
    for (let i = 0; i < this.thetaSize; i++) {
      theta[i] = x[i % x.length] * 0.1;
    }

    // Create backcast and forecast using polynomial basis
    const backcast = new Float32Array(this.inputSize);
    const forecast = new Float32Array(horizonLength);

    for (let t = 0; t < this.inputSize; t++) {
      for (let i = 0; i < this.thetaSize; i++) {
        backcast[t] += theta[i] * Math.pow(t / this.inputSize, i);
      }
    }

    for (let t = 0; t < horizonLength; t++) {
      for (let i = 0; i < this.thetaSize; i++) {
        forecast[t] += theta[i] * Math.pow((this.inputSize + t) / this.inputSize, i);
      }
    }

    return { backcast, forecast };
  }
}

/**
 * Neural Forecaster - Main class for traffic prediction
 */
export class NeuralForecaster {
  private config: ForecastConfig;
  private lstmModel: LSTMCell;
  private nbeatsBlocks: NBeatsBlock[];
  private forecastCache: Map<string, { forecast: TrafficForecast; timestamp: number }>;

  constructor(config: Partial<ForecastConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };

    // Initialize models
    this.lstmModel = new LSTMCell(1, 32);
    this.nbeatsBlocks = [
      new NBeatsBlock(32),
      new NBeatsBlock(32),
    ];

    this.forecastCache = new Map();

    logger.info('Neural forecaster initialized', { modelType: this.config.modelType });
  }

  /**
   * Generate traffic forecast for a cell
   */
  async forecast(timeSeries: TimeSeries): Promise<TrafficForecast> {
    const cacheKey = timeSeries.cellId;
    const cached = this.forecastCache.get(cacheKey);

    // Return cached forecast if fresh enough
    if (cached && Date.now() - cached.timestamp < 60000) {
      return cached.forecast;
    }

    const values = timeSeries.values;

    if (values.length < this.config.minHistoryPoints) {
      logger.warn('Insufficient data for forecasting', {
        cellId: timeSeries.cellId,
        dataPoints: values.length,
      });
      return this.createDefaultForecast(timeSeries.cellId);
    }

    let predictions: PredictionPoint[];
    let confidence: number;

    switch (this.config.modelType) {
      case 'lstm':
        ({ predictions, confidence } = this.forecastLSTM(values));
        break;
      case 'nbeats':
        ({ predictions, confidence } = this.forecastNBeats(values));
        break;
      case 'ensemble':
      default:
        ({ predictions, confidence } = this.forecastEnsemble(values));
    }

    const forecast: TrafficForecast = {
      cellId: timeSeries.cellId,
      predictions,
      confidence,
      model: this.config.modelType,
    };

    // Cache the forecast
    this.forecastCache.set(cacheKey, { forecast, timestamp: Date.now() });

    logger.debug('Forecast generated', {
      cellId: timeSeries.cellId,
      horizonPoints: predictions.length,
      confidence: confidence.toFixed(3),
    });

    return forecast;
  }

  /**
   * LSTM-based forecasting
   */
  private forecastLSTM(values: number[]): { predictions: PredictionPoint[]; confidence: number } {
    this.lstmModel.reset();

    // Normalize values
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const std = Math.sqrt(values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length) || 1;
    const normalized = values.map((v) => (v - mean) / std);

    // Feed history through LSTM
    let lastOutput = new Float32Array(32);
    for (const value of normalized) {
      const input = new Float32Array([value]);
      lastOutput = this.lstmModel.forward(input);
    }

    // Generate predictions
    const horizonPoints = Math.ceil(this.config.horizonMs / this.config.resolutionMs);
    const predictions: PredictionPoint[] = [];
    const now = Date.now();

    let currentOutput = lastOutput;
    for (let i = 0; i < horizonPoints; i++) {
      // Use last hidden state to predict
      const predictedNorm = currentOutput[0] * 0.1;
      const predictedValue = predictedNorm * std + mean;

      // Calculate prediction interval
      const uncertainty = std * (1 + i * 0.1);

      predictions.push({
        timestamp: now + (i + 1) * this.config.resolutionMs,
        value: Math.max(0, predictedValue),
        lowerBound: Math.max(0, predictedValue - 1.96 * uncertainty),
        upperBound: predictedValue + 1.96 * uncertainty,
      });

      // Feed prediction back
      const input = new Float32Array([predictedNorm]);
      currentOutput = this.lstmModel.forward(input);
    }

    // Confidence based on prediction variance
    const confidence = Math.max(0.5, 1 - std / mean * 0.5);

    return { predictions, confidence };
  }

  /**
   * N-BEATS forecasting
   */
  private forecastNBeats(values: number[]): { predictions: PredictionPoint[]; confidence: number } {
    const horizonPoints = Math.ceil(this.config.horizonMs / this.config.resolutionMs);

    // Use recent window
    const windowSize = 32;
    const recentValues = values.slice(-windowSize);
    const input = new Float32Array(recentValues);

    // Normalize
    const mean = recentValues.reduce((a, b) => a + b, 0) / recentValues.length;
    const std = Math.sqrt(recentValues.reduce((sum, v) => sum + (v - mean) ** 2, 0) / recentValues.length) || 1;

    for (let i = 0; i < input.length; i++) {
      input[i] = (input[i] - mean) / std;
    }

    // Stack N-BEATS blocks
    let residual = input;
    let totalForecast = new Float32Array(horizonPoints);

    for (const block of this.nbeatsBlocks) {
      const { backcast, forecast } = block.forward(residual, horizonPoints);

      // Subtract backcast from residual
      for (let i = 0; i < residual.length; i++) {
        residual[i] -= backcast[i];
      }

      // Add forecast
      for (let i = 0; i < horizonPoints; i++) {
        totalForecast[i] += forecast[i];
      }
    }

    // Denormalize and create predictions
    const predictions: PredictionPoint[] = [];
    const now = Date.now();

    for (let i = 0; i < horizonPoints; i++) {
      const value = totalForecast[i] * std + mean;
      const uncertainty = std * (1 + i * 0.05);

      predictions.push({
        timestamp: now + (i + 1) * this.config.resolutionMs,
        value: Math.max(0, value),
        lowerBound: Math.max(0, value - 1.96 * uncertainty),
        upperBound: value + 1.96 * uncertainty,
      });
    }

    const confidence = Math.max(0.6, 1 - std / mean * 0.4);

    return { predictions, confidence };
  }

  /**
   * Ensemble forecasting combining LSTM and N-BEATS
   */
  private forecastEnsemble(values: number[]): { predictions: PredictionPoint[]; confidence: number } {
    const lstmResult = this.forecastLSTM(values);
    const nbeatsResult = this.forecastNBeats(values);

    // Weighted average based on confidence
    const totalConfidence = lstmResult.confidence + nbeatsResult.confidence;
    const lstmWeight = lstmResult.confidence / totalConfidence;
    const nbeatsWeight = nbeatsResult.confidence / totalConfidence;

    const predictions: PredictionPoint[] = [];
    const count = Math.min(lstmResult.predictions.length, nbeatsResult.predictions.length);

    for (let i = 0; i < count; i++) {
      const lstm = lstmResult.predictions[i];
      const nbeats = nbeatsResult.predictions[i];

      predictions.push({
        timestamp: lstm.timestamp,
        value: lstm.value * lstmWeight + nbeats.value * nbeatsWeight,
        lowerBound: Math.min(lstm.lowerBound, nbeats.lowerBound),
        upperBound: Math.max(lstm.upperBound, nbeats.upperBound),
      });
    }

    const confidence = (lstmResult.confidence + nbeatsResult.confidence) / 2;

    return { predictions, confidence };
  }

  /**
   * Predict energy saving opportunities
   */
  predictEnergySavingWindow(
    forecast: TrafficForecast,
    threshold: number = 0.3
  ): EnergySavingWindow[] {
    const windows: EnergySavingWindow[] = [];
    let currentWindow: EnergySavingWindow | null = null;

    // Find periods where predicted load is below threshold
    const maxPrediction = Math.max(...forecast.predictions.map((p) => p.value));

    for (const prediction of forecast.predictions) {
      const normalizedLoad = prediction.value / maxPrediction;

      if (normalizedLoad < threshold) {
        if (!currentWindow) {
          currentWindow = {
            startTime: prediction.timestamp,
            endTime: prediction.timestamp,
            averageLoad: normalizedLoad,
            confidence: forecast.confidence,
            recommendedSleepRatio: Math.min(0.8, 1 - normalizedLoad),
          };
        } else {
          currentWindow.endTime = prediction.timestamp;
          currentWindow.averageLoad = (currentWindow.averageLoad + normalizedLoad) / 2;
        }
      } else if (currentWindow) {
        windows.push(currentWindow);
        currentWindow = null;
      }
    }

    if (currentWindow) {
      windows.push(currentWindow);
    }

    return windows;
  }

  /**
   * Detect diurnal patterns
   */
  detectDiurnalPattern(values: number[], resolution: number): DiurnalPattern {
    // Need at least 24 hours of data
    const pointsPerDay = Math.floor(this.config.seasonalityPeriod / resolution);
    if (values.length < pointsPerDay) {
      return { hasDiurnalPattern: false, peakHour: -1, troughHour: -1, amplitude: 0 };
    }

    // Calculate hourly averages
    const hoursPerDay = 24;
    const pointsPerHour = pointsPerDay / hoursPerDay;
    const hourlyAverages = new Array(hoursPerDay).fill(0);
    const hourlyCounts = new Array(hoursPerDay).fill(0);

    for (let i = 0; i < values.length; i++) {
      const hour = Math.floor((i % pointsPerDay) / pointsPerHour) % hoursPerDay;
      hourlyAverages[hour] += values[i];
      hourlyCounts[hour]++;
    }

    for (let h = 0; h < hoursPerDay; h++) {
      hourlyAverages[h] /= hourlyCounts[h] || 1;
    }

    // Find peak and trough
    let peakHour = 0;
    let troughHour = 0;
    let maxVal = hourlyAverages[0];
    let minVal = hourlyAverages[0];

    for (let h = 1; h < hoursPerDay; h++) {
      if (hourlyAverages[h] > maxVal) {
        maxVal = hourlyAverages[h];
        peakHour = h;
      }
      if (hourlyAverages[h] < minVal) {
        minVal = hourlyAverages[h];
        troughHour = h;
      }
    }

    const amplitude = (maxVal - minVal) / ((maxVal + minVal) / 2);
    const hasDiurnalPattern = amplitude > 0.3;

    return {
      hasDiurnalPattern,
      peakHour,
      troughHour,
      amplitude,
      hourlyAverages,
    };
  }

  /**
   * Create default forecast when data is insufficient
   */
  private createDefaultForecast(cellId: string): TrafficForecast {
    const horizonPoints = Math.ceil(this.config.horizonMs / this.config.resolutionMs);
    const now = Date.now();

    const predictions: PredictionPoint[] = [];
    for (let i = 0; i < horizonPoints; i++) {
      predictions.push({
        timestamp: now + (i + 1) * this.config.resolutionMs,
        value: 0,
        lowerBound: 0,
        upperBound: 0,
      });
    }

    return {
      cellId,
      predictions,
      confidence: 0,
      model: this.config.modelType,
    };
  }

  /**
   * Clear forecast cache
   */
  clearCache(): void {
    this.forecastCache.clear();
  }
}

/**
 * Energy saving window
 */
export interface EnergySavingWindow {
  startTime: number;
  endTime: number;
  averageLoad: number;
  confidence: number;
  recommendedSleepRatio: number;
}

/**
 * Diurnal pattern analysis
 */
export interface DiurnalPattern {
  hasDiurnalPattern: boolean;
  peakHour: number;
  troughHour: number;
  amplitude: number;
  hourlyAverages?: number[];
}

/**
 * Create a configured neural forecaster instance
 */
export function createNeuralForecaster(config?: Partial<ForecastConfig>): NeuralForecaster {
  return new NeuralForecaster(config);
}
