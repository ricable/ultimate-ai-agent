// k6 load testing script for API service
// Run with: k6 run --vus 100 --duration 5m load_testing_k6.js

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const jobCreationDuration = new Trend('job_creation_duration');
const jobStatusDuration = new Trend('job_status_duration');
const resultRetrievalDuration = new Trend('result_retrieval_duration');
const successfulJobs = new Counter('successful_jobs');

// Configuration
const BASE_URL = __ENV.API_URL || 'http://localhost:8080';
const AUTH_TOKEN = __ENV.AUTH_TOKEN || 'test-token-123';

// Load test scenarios
export const options = {
  stages: [
    // Ramp-up: 0 to 50 users over 1 minute
    { duration: '1m', target: 50 },
    // Stay at 50 users for 2 minutes
    { duration: '2m', target: 50 },
    // Ramp-up: 50 to 100 users over 1 minute
    { duration: '1m', target: 100 },
    // Stay at 100 users for 3 minutes (peak load)
    { duration: '3m', target: 100 },
    // Spike test: 100 to 200 users over 30 seconds
    { duration: '30s', target: 200 },
    // Stay at 200 users for 1 minute
    { duration: '1m', target: 200 },
    // Ramp-down: 200 to 0 users over 1 minute
    { duration: '1m', target: 0 },
  ],
  thresholds: {
    // 99% of requests must complete within 2 seconds
    http_req_duration: ['p(99)<2000'],
    // Error rate must be below 1%
    errors: ['rate<0.01'],
    // 95% of requests must succeed
    http_req_failed: ['rate<0.05'],
  },
};

// Helper function to generate random job payload
function generateJobPayload() {
  const randomId = Math.floor(Math.random() * 1000000);
  return JSON.stringify({
    job_type: 'alignment',
    input_files: [
      { name: `sample_${randomId}_R1.fastq`, size: 1024000 + randomId },
      { name: `sample_${randomId}_R2.fastq`, size: 1024000 + randomId },
    ],
    parameters: {
      reference_genome: 'macaca_mulatta',
      quality_threshold: 20 + (randomId % 20),
      alignment_mode: randomId % 2 === 0 ? 'end-to-end' : 'local',
    },
    metadata: {
      experiment_id: `EXP-${randomId}`,
      sample_type: 'RNA-seq',
      tissue: ['blood', 'liver', 'spleen'][randomId % 3],
    },
  });
}

// Main test scenario
export default function () {
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${AUTH_TOKEN}`,
  };

  // Test 1: Create analysis job
  const jobPayload = generateJobPayload();
  let createResponse = http.post(
    `${BASE_URL}/api/v1/jobs`,
    jobPayload,
    { headers, tags: { name: 'CreateJob' } }
  );

  jobCreationDuration.add(createResponse.timings.duration);

  const createSuccess = check(createResponse, {
    'job creation status is 201': (r) => r.status === 201,
    'job creation returns job_id': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.job_id !== undefined;
      } catch {
        return false;
      }
    },
  });

  if (!createSuccess) {
    errorRate.add(1);
    return; // Skip subsequent tests if job creation fails
  }

  errorRate.add(0);
  successfulJobs.add(1);

  // Extract job ID for subsequent requests
  const jobId = JSON.parse(createResponse.body).job_id;

  // Wait before polling status
  sleep(1);

  // Test 2: Check job status
  let statusResponse = http.get(
    `${BASE_URL}/api/v1/jobs/${jobId}`,
    { headers, tags: { name: 'GetJobStatus' } }
  );

  jobStatusDuration.add(statusResponse.timings.duration);

  check(statusResponse, {
    'job status query is 200': (r) => r.status === 200,
    'job status has valid state': (r) => {
      try {
        const body = JSON.parse(r.body);
        return ['pending', 'running', 'completed', 'failed'].includes(body.status);
      } catch {
        return false;
      }
    },
  });

  // Test 3: List jobs with pagination
  let listResponse = http.get(
    `${BASE_URL}/api/v1/jobs?page=1&limit=20`,
    { headers, tags: { name: 'ListJobs' } }
  );

  check(listResponse, {
    'job list query is 200': (r) => r.status === 200,
    'job list has pagination': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.total !== undefined && body.jobs !== undefined;
      } catch {
        return false;
      }
    },
  });

  // Simulate varying user behavior
  const behavior = Math.random();

  if (behavior < 0.3) {
    // 30% of users wait and check results
    sleep(2);

    // Test 4: Retrieve results (simulate completed job)
    let resultsResponse = http.get(
      `${BASE_URL}/api/v1/jobs/${jobId}/alignments`,
      { headers, tags: { name: 'GetResults' } }
    );

    resultRetrievalDuration.add(resultsResponse.timings.duration);

    check(resultsResponse, {
      'results query returns 200 or 404': (r) => [200, 404].includes(r.status),
    });
  } else if (behavior < 0.6) {
    // 30% of users poll status multiple times
    for (let i = 0; i < 3; i++) {
      sleep(0.5);
      http.get(`${BASE_URL}/api/v1/jobs/${jobId}`, { headers });
    }
  } else {
    // 40% of users just submit and leave
    sleep(0.5);
  }

  // Random think time between requests (0.5-2 seconds)
  sleep(Math.random() * 1.5 + 0.5);
}

// Scenario 2: Stress test with concurrent connections
export function stressTest() {
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${AUTH_TOKEN}`,
  };

  // Simulate burst of job submissions
  const jobs = [];
  for (let i = 0; i < 10; i++) {
    const jobPayload = generateJobPayload();
    const response = http.post(
      `${BASE_URL}/api/v1/jobs`,
      jobPayload,
      { headers }
    );

    if (response.status === 201) {
      jobs.push(JSON.parse(response.body).job_id);
    }
  }

  // Query all submitted jobs
  jobs.forEach((jobId) => {
    http.get(`${BASE_URL}/api/v1/jobs/${jobId}`, { headers });
  });
}

// Scenario 3: Spike test
export function spikeTest() {
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${AUTH_TOKEN}`,
  };

  // Simulate sudden spike in traffic
  const responses = [];
  for (let i = 0; i < 50; i++) {
    responses.push(
      http.post(
        `${BASE_URL}/api/v1/jobs`,
        generateJobPayload(),
        { headers }
      )
    );
  }

  const successCount = responses.filter((r) => r.status === 201).length;
  console.log(`Spike test: ${successCount}/50 requests succeeded`);
}

// Custom summary for detailed reporting
export function handleSummary(data) {
  return {
    'load-test-summary.json': JSON.stringify(data),
    stdout: textSummary(data, { indent: ' ', enableColors: true }),
  };
}

function textSummary(data) {
  const summary = [];

  summary.push('='.repeat(80));
  summary.push('LOAD TEST SUMMARY');
  summary.push('='.repeat(80));

  summary.push(`\nTest Duration: ${data.state.testRunDurationMs / 1000}s`);
  summary.push(`Total VUs: ${data.metrics.vus?.values?.max || 0}`);
  summary.push(`Total Iterations: ${data.metrics.iterations?.values?.count || 0}`);

  summary.push('\n--- HTTP Metrics ---');
  summary.push(`Requests: ${data.metrics.http_reqs?.values?.count || 0}`);
  summary.push(`Failed Requests: ${data.metrics.http_req_failed?.values?.rate * 100}%`);
  summary.push(`Request Duration (avg): ${data.metrics.http_req_duration?.values?.avg}ms`);
  summary.push(`Request Duration (p95): ${data.metrics.http_req_duration?.values['p(95)']}ms`);
  summary.push(`Request Duration (p99): ${data.metrics.http_req_duration?.values['p(99)']}ms`);

  summary.push('\n--- Custom Metrics ---');
  summary.push(`Error Rate: ${(data.metrics.errors?.values?.rate || 0) * 100}%`);
  summary.push(`Successful Jobs: ${data.metrics.successful_jobs?.values?.count || 0}`);
  summary.push(`Job Creation (avg): ${data.metrics.job_creation_duration?.values?.avg}ms`);
  summary.push(`Job Status Query (avg): ${data.metrics.job_status_duration?.values?.avg}ms`);

  summary.push('\n' + '='.repeat(80));

  return summary.join('\n');
}
