"""
Locust load testing script for CRISPR-Cas13 API service
Run with: locust -f load_testing_locust.py --host=http://localhost:8080 --users 100 --spawn-rate 10
"""

import json
import random
import time
from locust import HttpUser, task, between, events
from locust.contrib.fasthttp import FastHttpUser

# Configuration
AUTH_TOKEN = "test-token-123"


class CrisprPipelineUser(FastHttpUser):
    """Simulates a user interacting with the CRISPR pipeline API"""

    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    def on_start(self):
        """Called when a user starts - setup authentication"""
        self.client.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {AUTH_TOKEN}'
        }
        self.submitted_jobs = []

    @task(5)
    def create_alignment_job(self):
        """Submit an alignment job (highest weight - most common operation)"""
        job_payload = self._generate_job_payload('alignment')

        with self.client.post(
            '/api/v1/jobs',
            json=job_payload,
            catch_response=True,
            name='CreateJob:Alignment'
        ) as response:
            if response.status_code == 201:
                try:
                    job_data = response.json()
                    self.submitted_jobs.append(job_data['job_id'])
                    response.success()
                except Exception as e:
                    response.failure(f'Failed to parse job response: {e}')
            elif response.status_code == 429:
                response.failure('Rate limited')
            else:
                response.failure(f'Unexpected status code: {response.status_code}')

    @task(3)
    def check_job_status(self):
        """Check status of a previously submitted job"""
        if not self.submitted_jobs:
            return  # Skip if no jobs submitted yet

        job_id = random.choice(self.submitted_jobs)

        with self.client.get(
            f'/api/v1/jobs/{job_id}',
            catch_response=True,
            name='GetJobStatus'
        ) as response:
            if response.status_code == 200:
                try:
                    job_data = response.json()
                    if job_data['status'] not in ['pending', 'running', 'completed', 'failed']:
                        response.failure(f'Invalid job status: {job_data["status"]}')
                    else:
                        response.success()
                except Exception as e:
                    response.failure(f'Failed to parse status response: {e}')
            elif response.status_code == 404:
                response.failure('Job not found')
            else:
                response.failure(f'Unexpected status code: {response.status_code}')

    @task(2)
    def list_jobs(self):
        """List jobs with pagination"""
        page = random.randint(1, 10)
        limit = random.choice([10, 20, 50])

        with self.client.get(
            f'/api/v1/jobs?page={page}&limit={limit}',
            catch_response=True,
            name='ListJobs'
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'jobs' in data and 'total' in data:
                        response.success()
                    else:
                        response.failure('Missing pagination fields')
                except Exception as e:
                    response.failure(f'Failed to parse list response: {e}')
            else:
                response.failure(f'Unexpected status code: {response.status_code}')

    @task(1)
    def retrieve_alignment_results(self):
        """Retrieve alignment results for a job"""
        if not self.submitted_jobs:
            return

        job_id = random.choice(self.submitted_jobs)

        with self.client.get(
            f'/api/v1/jobs/{job_id}/alignments',
            catch_response=True,
            name='GetAlignmentResults'
        ) as response:
            if response.status_code in [200, 404]:  # 404 is OK if job not completed yet
                response.success()
            else:
                response.failure(f'Unexpected status code: {response.status_code}')

    @task(1)
    def retrieve_offtarget_results(self):
        """Retrieve off-target prediction results"""
        if not self.submitted_jobs:
            return

        job_id = random.choice(self.submitted_jobs)

        with self.client.get(
            f'/api/v1/jobs/{job_id}/offtargets',
            catch_response=True,
            name='GetOffTargetResults'
        ) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f'Unexpected status code: {response.status_code}')

    @task(1)
    def export_results_csv(self):
        """Export results as CSV"""
        if not self.submitted_jobs:
            return

        job_id = random.choice(self.submitted_jobs)

        with self.client.get(
            f'/api/v1/jobs/{job_id}/results/csv',
            headers={'Accept': 'text/csv'},
            catch_response=True,
            name='ExportResultsCSV'
        ) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f'Unexpected status code: {response.status_code}')

    def _generate_job_payload(self, job_type='alignment'):
        """Generate random job submission payload"""
        random_id = random.randint(1, 1000000)

        return {
            'job_type': job_type,
            'input_files': [
                {
                    'name': f'sample_{random_id}_R1.fastq',
                    'size': 1024000 + random_id
                },
                {
                    'name': f'sample_{random_id}_R2.fastq',
                    'size': 1024000 + random_id
                }
            ],
            'parameters': {
                'reference_genome': 'macaca_mulatta',
                'quality_threshold': random.randint(20, 40),
                'alignment_mode': random.choice(['end-to-end', 'local'])
            },
            'metadata': {
                'experiment_id': f'EXP-{random_id}',
                'sample_type': 'RNA-seq',
                'tissue': random.choice(['blood', 'liver', 'spleen', 'brain'])
            }
        }


class PowerUser(CrisprPipelineUser):
    """Simulates a power user who submits many jobs"""

    weight = 1  # 10% of users are power users
    wait_time = between(0.5, 1)  # Less wait time

    @task(10)
    def batch_job_submission(self):
        """Submit multiple jobs in quick succession"""
        for _ in range(5):
            self.create_alignment_job()
            time.sleep(0.1)


class ReadOnlyUser(FastHttpUser):
    """Simulates a read-only user who only checks results"""

    weight = 2  # 20% of users are read-only
    wait_time = between(2, 5)

    def on_start(self):
        self.client.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {AUTH_TOKEN}'
        }

    @task(5)
    def list_jobs(self):
        """List all jobs"""
        page = random.randint(1, 5)
        self.client.get(f'/api/v1/jobs?page={page}&limit=20', name='ListJobs:ReadOnly')

    @task(3)
    def check_job_status(self):
        """Check random job status"""
        # Simulate checking a known job ID
        job_id = f'job-{random.randint(1, 1000)}'
        self.client.get(f'/api/v1/jobs/{job_id}', name='GetJobStatus:ReadOnly')

    @task(1)
    def health_check(self):
        """Check API health"""
        self.client.get('/health', name='HealthCheck')


# Custom event handlers for detailed metrics
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Log slow requests"""
    if response_time > 2000:  # Log requests slower than 2 seconds
        print(f'SLOW REQUEST: {name} took {response_time}ms')


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts"""
    print('Starting load test...')
    print(f'Target host: {environment.host}')


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops - generate summary report"""
    print('\n' + '='*80)
    print('LOAD TEST SUMMARY')
    print('='*80)

    stats = environment.stats

    print(f'\nTotal Requests: {stats.total.num_requests}')
    print(f'Total Failures: {stats.total.num_failures}')
    print(f'Failure Rate: {stats.total.fail_ratio * 100:.2f}%')
    print(f'Average Response Time: {stats.total.avg_response_time:.2f}ms')
    print(f'Median Response Time: {stats.total.median_response_time:.2f}ms')
    print(f'95th Percentile: {stats.total.get_response_time_percentile(0.95):.2f}ms')
    print(f'99th Percentile: {stats.total.get_response_time_percentile(0.99):.2f}ms')
    print(f'Requests per Second: {stats.total.total_rps:.2f}')

    print('\n--- Top 10 Slowest Endpoints ---')
    sorted_stats = sorted(stats.entries.values(), key=lambda x: x.avg_response_time, reverse=True)
    for i, stat in enumerate(sorted_stats[:10], 1):
        print(f'{i}. {stat.name}: {stat.avg_response_time:.2f}ms (avg)')

    print('\n--- Top 10 Most Failed Endpoints ---')
    sorted_failures = sorted(stats.entries.values(), key=lambda x: x.num_failures, reverse=True)
    for i, stat in enumerate(sorted_failures[:10], 1):
        if stat.num_failures > 0:
            print(f'{i}. {stat.name}: {stat.num_failures} failures')

    print('\n' + '='*80)

    # Save detailed report to file
    with open('locust_report.json', 'w') as f:
        json.dump({
            'total_requests': stats.total.num_requests,
            'total_failures': stats.total.num_failures,
            'failure_rate': stats.total.fail_ratio,
            'avg_response_time': stats.total.avg_response_time,
            'median_response_time': stats.total.median_response_time,
            'p95_response_time': stats.total.get_response_time_percentile(0.95),
            'p99_response_time': stats.total.get_response_time_percentile(0.99),
            'rps': stats.total.total_rps,
        }, f, indent=2)

    print('Detailed report saved to locust_report.json')
