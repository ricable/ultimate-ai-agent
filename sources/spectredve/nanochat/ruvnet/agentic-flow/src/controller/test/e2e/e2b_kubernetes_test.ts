/**
 * E2B Kubernetes Deployment Tests
 *
 * Tests deployment patterns using E2B sandbox environments
 * with Kubernetes simulation
 */

import { Sandbox } from '@e2b/code-interpreter';

interface TestResult {
  name: string;
  passed: boolean;
  duration: number;
  error?: string;
  logs?: string[];
}

/**
 * Test controller binary compilation
 */
async function testControllerBuild(sandbox: Sandbox): Promise<TestResult> {
  const startTime = Date.now();
  const logs: string[] = [];

  try {
    logs.push('Installing Go dependencies...');
    await sandbox.commands.run('cd /workspace/controller && go mod download');

    logs.push('Building controller...');
    const buildResult = await sandbox.commands.run(
      'cd /workspace/controller && make build'
    );

    if (buildResult.exitCode !== 0) {
      return {
        name: 'Controller Build',
        passed: false,
        duration: Date.now() - startTime,
        error: buildResult.stderr,
        logs,
      };
    }

    logs.push('Verifying binary...');
    const verifyResult = await sandbox.commands.run(
      'test -f /workspace/controller/bin/manager && echo "Binary exists"'
    );

    return {
      name: 'Controller Build',
      passed: verifyResult.exitCode === 0,
      duration: Date.now() - startTime,
      logs,
    };
  } catch (error) {
    return {
      name: 'Controller Build',
      passed: false,
      duration: Date.now() - startTime,
      error: error instanceof Error ? error.message : String(error),
      logs,
    };
  }
}

/**
 * Test CRD generation
 */
async function testCRDGeneration(sandbox: Sandbox): Promise<TestResult> {
  const startTime = Date.now();
  const logs: string[] = [];

  try {
    logs.push('Generating CRDs...');
    const result = await sandbox.commands.run(
      'cd /workspace/controller && make manifests'
    );

    if (result.exitCode !== 0) {
      return {
        name: 'CRD Generation',
        passed: false,
        duration: Date.now() - startTime,
        error: result.stderr,
        logs,
      };
    }

    logs.push('Verifying CRD files...');
    const verifyResult = await sandbox.commands.run(
      'ls -la /workspace/controller/config/crd/bases/'
    );

    // Check for expected CRD files
    const expectedCRDs = [
      'ajj.io_applications.yaml',
      'ajj.io_clusters.yaml',
    ];

    let allFound = true;
    for (const crd of expectedCRDs) {
      const checkResult = await sandbox.commands.run(
        `test -f /workspace/controller/config/crd/bases/${crd} && echo "Found"`
      );
      if (checkResult.exitCode !== 0) {
        logs.push(`Missing CRD: ${crd}`);
        allFound = false;
      } else {
        logs.push(`Found CRD: ${crd}`);
      }
    }

    return {
      name: 'CRD Generation',
      passed: allFound,
      duration: Date.now() - startTime,
      logs,
    };
  } catch (error) {
    return {
      name: 'CRD Generation',
      passed: false,
      duration: Date.now() - startTime,
      error: error instanceof Error ? error.message : String(error),
      logs,
    };
  }
}

/**
 * Test Jujutsu client integration
 */
async function testJujutsuClient(sandbox: Sandbox): Promise<TestResult> {
  const startTime = Date.now();
  const logs: string[] = [];

  try {
    logs.push('Installing Jujutsu...');
    await sandbox.commands.run(
      'wget https://github.com/martinvonz/jj/releases/download/v0.12.0/jj-v0.12.0-x86_64-unknown-linux-musl.tar.gz && ' +
      'tar -xzf jj-v0.12.0-x86_64-unknown-linux-musl.tar.gz && ' +
      'sudo mv jj /usr/local/bin/ && ' +
      'sudo chmod +x /usr/local/bin/jj'
    );

    logs.push('Verifying jj installation...');
    const versionResult = await sandbox.commands.run('jj --version');
    logs.push(`Jujutsu version: ${versionResult.stdout}`);

    logs.push('Initializing test repository...');
    await sandbox.commands.run(
      'mkdir -p /tmp/test-repo && cd /tmp/test-repo && jj init --git'
    );

    logs.push('Creating test files...');
    await sandbox.commands.run(
      'cd /tmp/test-repo && mkdir -p apps/test && ' +
      'cat > apps/test/deployment.yaml << EOF\n' +
      'apiVersion: apps/v1\n' +
      'kind: Deployment\n' +
      'metadata:\n' +
      '  name: test-app\n' +
      'spec:\n' +
      '  replicas: 2\n' +
      'EOF'
    );

    logs.push('Creating Jujutsu change...');
    const createResult = await sandbox.commands.run(
      'cd /tmp/test-repo && jj describe -m "Add test deployment"'
    );

    logs.push('Listing changes...');
    const listResult = await sandbox.commands.run(
      'cd /tmp/test-repo && jj log --no-graph'
    );
    logs.push(`Changes:\n${listResult.stdout}`);

    return {
      name: 'Jujutsu Client Integration',
      passed: createResult.exitCode === 0 && listResult.stdout.includes('test deployment'),
      duration: Date.now() - startTime,
      logs,
    };
  } catch (error) {
    return {
      name: 'Jujutsu Client Integration',
      passed: false,
      duration: Date.now() - startTime,
      error: error instanceof Error ? error.message : String(error),
      logs,
    };
  }
}

/**
 * Test Kubernetes deployment pattern simulation
 */
async function testKubernetesDeployment(sandbox: Sandbox): Promise<TestResult> {
  const startTime = Date.now();
  const logs: string[] = [];

  try {
    logs.push('Installing kubectl...');
    await sandbox.commands.run(
      'curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && ' +
      'sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl'
    );

    logs.push('Installing kind (Kubernetes in Docker)...');
    await sandbox.commands.run(
      'curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64 && ' +
      'sudo install -o root -g root -m 0755 kind /usr/local/bin/kind'
    );

    logs.push('Creating kind cluster...');
    const clusterResult = await sandbox.commands.run(
      'kind create cluster --name ajj-test --wait 5m'
    );

    if (clusterResult.exitCode !== 0) {
      logs.push('Kind cluster creation failed, skipping Kubernetes tests');
      return {
        name: 'Kubernetes Deployment',
        passed: false,
        duration: Date.now() - startTime,
        error: 'Could not create kind cluster in sandbox environment',
        logs,
      };
    }

    logs.push('Verifying cluster...');
    const verifyResult = await sandbox.commands.run('kubectl cluster-info');
    logs.push(`Cluster info:\n${verifyResult.stdout}`);

    logs.push('Creating test namespace...');
    await sandbox.commands.run('kubectl create namespace ajj-test');

    logs.push('Applying test deployment...');
    await sandbox.commands.run(
      'kubectl apply -f /tmp/test-repo/apps/test/deployment.yaml -n ajj-test'
    );

    logs.push('Checking deployment status...');
    const statusResult = await sandbox.commands.run(
      'kubectl get deployments -n ajj-test'
    );
    logs.push(`Deployments:\n${statusResult.stdout}`);

    return {
      name: 'Kubernetes Deployment',
      passed: statusResult.exitCode === 0 && statusResult.stdout.includes('test-app'),
      duration: Date.now() - startTime,
      logs,
    };
  } catch (error) {
    return {
      name: 'Kubernetes Deployment',
      passed: false,
      duration: Date.now() - startTime,
      error: error instanceof Error ? error.message : String(error),
      logs,
    };
  }
}

/**
 * Test Helm chart rendering
 */
async function testHelmChart(sandbox: Sandbox): Promise<TestResult> {
  const startTime = Date.now();
  const logs: string[] = [];

  try {
    logs.push('Installing Helm...');
    await sandbox.commands.run(
      'curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash'
    );

    logs.push('Verifying Helm installation...');
    const versionResult = await sandbox.commands.run('helm version');
    logs.push(`Helm version: ${versionResult.stdout}`);

    logs.push('Rendering Helm chart...');
    const renderResult = await sandbox.commands.run(
      'helm template ajj-controller /workspace/charts/jujutsu-gitops-controller ' +
      '--set image.tag=test ' +
      '--set repository.url=https://github.com/test/repo'
    );

    if (renderResult.exitCode !== 0) {
      return {
        name: 'Helm Chart Rendering',
        passed: false,
        duration: Date.now() - startTime,
        error: renderResult.stderr,
        logs,
      };
    }

    logs.push('Validating rendered manifests...');
    const output = renderResult.stdout;

    // Check for expected resources
    const expectedResources = [
      'Deployment',
      'ServiceAccount',
      'ClusterRole',
      'ClusterRoleBinding',
    ];

    let allFound = true;
    for (const resource of expectedResources) {
      if (!output.includes(`kind: ${resource}`)) {
        logs.push(`Missing resource: ${resource}`);
        allFound = false;
      } else {
        logs.push(`Found resource: ${resource}`);
      }
    }

    return {
      name: 'Helm Chart Rendering',
      passed: allFound && renderResult.exitCode === 0,
      duration: Date.now() - startTime,
      logs,
    };
  } catch (error) {
    return {
      name: 'Helm Chart Rendering',
      passed: false,
      duration: Date.now() - startTime,
      error: error instanceof Error ? error.message : String(error),
      logs,
    };
  }
}

/**
 * Main test runner
 */
async function runTests() {
  console.log('🚀 Starting E2B Kubernetes deployment tests...\n');

  const sandbox = await Sandbox.create({
    timeout: 600000, // 10 minutes
  });

  try {
    console.log('📦 Setting up test environment...');

    // Copy controller code to sandbox
    console.log('Copying controller code...');
    await sandbox.commands.run('mkdir -p /workspace');
    // Note: In real implementation, code would be uploaded to sandbox

    const tests = [
      testControllerBuild,
      testCRDGeneration,
      testJujutsuClient,
      testHelmChart,
      // testKubernetesDeployment, // Disabled by default, requires Docker in sandbox
    ];

    const results: TestResult[] = [];

    for (const test of tests) {
      console.log(`\n🧪 Running ${test.name}...`);
      const result = await test(sandbox);
      results.push(result);

      if (result.logs) {
        console.log('📝 Logs:');
        result.logs.forEach(log => console.log(`  ${log}`));
      }

      if (result.passed) {
        console.log(`✅ ${result.name} passed in ${result.duration}ms`);
      } else {
        console.log(`❌ ${result.name} failed in ${result.duration}ms`);
        if (result.error) {
          console.log(`   Error: ${result.error}`);
        }
      }
    }

    // Summary
    console.log('\n📊 Test Summary:');
    console.log('═══════════════════════════════════════');
    const passed = results.filter(r => r.passed).length;
    const failed = results.filter(r => !r.passed).length;
    const total = results.length;
    console.log(`Total: ${total} tests`);
    console.log(`✅ Passed: ${passed}`);
    console.log(`❌ Failed: ${failed}`);
    console.log(`Success Rate: ${((passed / total) * 100).toFixed(1)}%`);

    return results;
  } finally {
    await sandbox.close();
  }
}

// Run tests if executed directly
if (require.main === module) {
  runTests()
    .then((results) => {
      const failed = results.filter(r => !r.passed).length;
      process.exit(failed > 0 ? 1 : 0);
    })
    .catch((error) => {
      console.error('Fatal error:', error);
      process.exit(1);
    });
}

export { runTests, TestResult };
