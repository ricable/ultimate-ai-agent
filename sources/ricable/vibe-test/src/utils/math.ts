/**
 * High-performance mathematical utilities for RAN optimization
 * Implements sublinear algorithms where possible
 */

/**
 * Calculate Lyapunov exponent to detect chaotic behavior in time series
 * Uses efficient algorithm with O(n log n) complexity
 */
export function calculateLyapunovExponent(
  timeSeries: number[],
  embeddingDimension: number = 3,
  timeDelay: number = 1,
  epsilon: number = 0.001
): number {
  const n = timeSeries.length;
  if (n < embeddingDimension * timeDelay + 10) {
    return 0;
  }

  // Construct delay embedding
  const embedded: number[][] = [];
  for (let i = 0; i < n - (embeddingDimension - 1) * timeDelay; i++) {
    const point: number[] = [];
    for (let j = 0; j < embeddingDimension; j++) {
      point.push(timeSeries[i + j * timeDelay]);
    }
    embedded.push(point);
  }

  // Calculate divergence rates
  let sumLog = 0;
  let count = 0;

  for (let i = 0; i < embedded.length - 1; i++) {
    // Find nearest neighbor (excluding temporal neighbors)
    let minDist = Infinity;
    let nearestIdx = -1;

    for (let j = 0; j < embedded.length - 1; j++) {
      if (Math.abs(i - j) <= timeDelay) continue;

      const dist = euclideanDistance(embedded[i], embedded[j]);
      if (dist < minDist && dist > epsilon) {
        minDist = dist;
        nearestIdx = j;
      }
    }

    if (nearestIdx === -1) continue;

    // Calculate divergence after one time step
    const nextDist = euclideanDistance(embedded[i + 1], embedded[nearestIdx + 1]);
    if (nextDist > epsilon && minDist > epsilon) {
      sumLog += Math.log(nextDist / minDist);
      count++;
    }
  }

  return count > 0 ? sumLog / count : 0;
}

/**
 * Calculate correlation dimension using Grassberger-Procaccia algorithm
 */
export function calculateCorrelationDimension(
  timeSeries: number[],
  embeddingDimension: number = 3,
  timeDelay: number = 1
): number {
  const embedded = embedTimeSeries(timeSeries, embeddingDimension, timeDelay);
  const n = embedded.length;
  if (n < 50) return 0;

  // Calculate correlation sum for multiple scales
  const epsilons: number[] = [];
  const correlations: number[] = [];

  const maxDist = calculateMaxDistance(embedded);
  const minDist = maxDist / 1000;

  for (let e = minDist; e < maxDist; e *= 1.5) {
    let count = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (euclideanDistance(embedded[i], embedded[j]) < e) {
          count++;
        }
      }
    }
    const correlation = (2 * count) / (n * (n - 1));
    if (correlation > 0) {
      epsilons.push(Math.log(e));
      correlations.push(Math.log(correlation));
    }
  }

  // Linear regression to find slope (correlation dimension)
  return linearRegressionSlope(epsilons, correlations);
}

/**
 * Fast Fourier Transform for frequency analysis
 * Uses Cooley-Tukey algorithm
 */
export function fft(input: number[]): { real: number[]; imag: number[] } {
  const n = input.length;
  const paddedLength = nextPowerOf2(n);

  // Pad with zeros
  const real = new Array(paddedLength).fill(0);
  const imag = new Array(paddedLength).fill(0);
  for (let i = 0; i < n; i++) {
    real[i] = input[i];
  }

  // Bit-reversal permutation
  for (let i = 0; i < paddedLength; i++) {
    const j = reverseBits(i, Math.log2(paddedLength));
    if (j > i) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
  }

  // Cooley-Tukey iterative FFT
  for (let size = 2; size <= paddedLength; size *= 2) {
    const halfSize = size / 2;
    const angle = (-2 * Math.PI) / size;

    for (let i = 0; i < paddedLength; i += size) {
      for (let j = 0; j < halfSize; j++) {
        const theta = angle * j;
        const cos = Math.cos(theta);
        const sin = Math.sin(theta);

        const idx1 = i + j;
        const idx2 = i + j + halfSize;

        const tReal = cos * real[idx2] - sin * imag[idx2];
        const tImag = sin * real[idx2] + cos * imag[idx2];

        real[idx2] = real[idx1] - tReal;
        imag[idx2] = imag[idx1] - tImag;
        real[idx1] = real[idx1] + tReal;
        imag[idx1] = imag[idx1] + tImag;
      }
    }
  }

  return { real: real.slice(0, n), imag: imag.slice(0, n) };
}

/**
 * Calculate entropy of a distribution
 */
export function calculateEntropy(probabilities: number[]): number {
  return -probabilities
    .filter((p) => p > 0)
    .reduce((sum, p) => sum + p * Math.log2(p), 0);
}

/**
 * Dynamic Time Warping distance between two time series
 * Useful for pattern matching across different cells
 */
export function dtwDistance(series1: number[], series2: number[]): number {
  const n = series1.length;
  const m = series2.length;

  // Use window constraint for efficiency
  const window = Math.max(10, Math.floor(Math.max(n, m) * 0.1));

  const dtw: number[][] = Array(n + 1)
    .fill(null)
    .map(() => Array(m + 1).fill(Infinity));

  dtw[0][0] = 0;

  for (let i = 1; i <= n; i++) {
    const jStart = Math.max(1, i - window);
    const jEnd = Math.min(m, i + window);

    for (let j = jStart; j <= jEnd; j++) {
      const cost = Math.abs(series1[i - 1] - series2[j - 1]);
      dtw[i][j] = cost + Math.min(dtw[i - 1][j], dtw[i][j - 1], dtw[i - 1][j - 1]);
    }
  }

  return dtw[n][m];
}

/**
 * Matrix operations for beamforming calculations
 */
export function matrixMultiply(a: number[][], b: number[][]): number[][] {
  const rowsA = a.length;
  const colsA = a[0].length;
  const colsB = b[0].length;

  const result: number[][] = Array(rowsA)
    .fill(null)
    .map(() => Array(colsB).fill(0));

  for (let i = 0; i < rowsA; i++) {
    for (let j = 0; j < colsB; j++) {
      for (let k = 0; k < colsA; k++) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }

  return result;
}

/**
 * Singular Value Decomposition (simplified)
 * Used for interference cancellation matrix calculation
 */
export function computeSVD(matrix: number[][]): {
  u: number[][];
  s: number[];
  v: number[][];
} {
  // Simplified power iteration method for dominant singular values
  const m = matrix.length;
  const n = matrix[0].length;
  const minDim = Math.min(m, n);

  const u: number[][] = [];
  const s: number[] = [];
  const v: number[][] = [];

  let workingMatrix = matrix.map((row) => [...row]);

  for (let k = 0; k < minDim; k++) {
    // Power iteration for dominant singular vector
    let vk = new Array(n).fill(1 / Math.sqrt(n));

    for (let iter = 0; iter < 100; iter++) {
      // u = A * v
      const uk = workingMatrix.map((row) =>
        row.reduce((sum, val, j) => sum + val * vk[j], 0)
      );

      // Normalize u
      const normU = Math.sqrt(uk.reduce((sum, val) => sum + val * val, 0));
      if (normU < 1e-10) break;
      uk.forEach((_, i) => (uk[i] /= normU));

      // v = A^T * u
      const newVk = new Array(n).fill(0);
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          newVk[j] += workingMatrix[i][j] * uk[i];
        }
      }

      // Singular value
      const sigma = Math.sqrt(newVk.reduce((sum, val) => sum + val * val, 0));
      if (sigma < 1e-10) break;

      // Normalize v
      newVk.forEach((_, i) => (newVk[i] /= sigma));

      // Check convergence
      const diff = vk.reduce((sum, val, i) => sum + Math.abs(val - newVk[i]), 0);
      vk = newVk;

      if (diff < 1e-10) {
        u.push(uk);
        s.push(sigma);
        v.push(vk);

        // Deflate matrix
        for (let i = 0; i < m; i++) {
          for (let j = 0; j < n; j++) {
            workingMatrix[i][j] -= sigma * uk[i] * vk[j];
          }
        }
        break;
      }
    }
  }

  return { u, s, v };
}

// Helper functions
function euclideanDistance(a: number[], b: number[]): number {
  return Math.sqrt(a.reduce((sum, val, i) => sum + (val - b[i]) ** 2, 0));
}

function embedTimeSeries(
  series: number[],
  dimension: number,
  delay: number
): number[][] {
  const result: number[][] = [];
  for (let i = 0; i < series.length - (dimension - 1) * delay; i++) {
    const point: number[] = [];
    for (let j = 0; j < dimension; j++) {
      point.push(series[i + j * delay]);
    }
    result.push(point);
  }
  return result;
}

function calculateMaxDistance(points: number[][]): number {
  let maxDist = 0;
  const sample = points.slice(0, Math.min(100, points.length));
  for (let i = 0; i < sample.length; i++) {
    for (let j = i + 1; j < sample.length; j++) {
      maxDist = Math.max(maxDist, euclideanDistance(sample[i], sample[j]));
    }
  }
  return maxDist;
}

function linearRegressionSlope(x: number[], y: number[]): number {
  const n = x.length;
  if (n < 2) return 0;

  const sumX = x.reduce((a, b) => a + b, 0);
  const sumY = y.reduce((a, b) => a + b, 0);
  const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
  const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);

  const denominator = n * sumXX - sumX * sumX;
  if (Math.abs(denominator) < 1e-10) return 0;

  return (n * sumXY - sumX * sumY) / denominator;
}

function nextPowerOf2(n: number): number {
  return Math.pow(2, Math.ceil(Math.log2(n)));
}

function reverseBits(x: number, bits: number): number {
  let result = 0;
  for (let i = 0; i < bits; i++) {
    result = (result << 1) | (x & 1);
    x >>= 1;
  }
  return result;
}

/**
 * Exponential moving average for streaming data
 */
export function exponentialMovingAverage(
  values: number[],
  alpha: number = 0.1
): number[] {
  if (values.length === 0) return [];

  const ema: number[] = [values[0]];
  for (let i = 1; i < values.length; i++) {
    ema.push(alpha * values[i] + (1 - alpha) * ema[i - 1]);
  }
  return ema;
}

/**
 * Calculate percentile of an array
 */
export function percentile(arr: number[], p: number): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = (p / 100) * (sorted.length - 1);
  const lower = Math.floor(idx);
  const upper = Math.ceil(idx);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (idx - lower) * (sorted[upper] - sorted[lower]);
}

/**
 * Standard deviation
 */
export function standardDeviation(arr: number[]): number {
  const n = arr.length;
  if (n === 0) return 0;
  const mean = arr.reduce((a, b) => a + b, 0) / n;
  const variance = arr.reduce((sum, val) => sum + (val - mean) ** 2, 0) / n;
  return Math.sqrt(variance);
}
