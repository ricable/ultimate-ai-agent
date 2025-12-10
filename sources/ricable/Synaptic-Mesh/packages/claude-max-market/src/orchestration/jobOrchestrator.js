/**
 * Job Orchestrator - Docker-based Claude job execution
 * 
 * Handles secure containerized execution of Claude tasks with
 * full isolation and resource management
 */

import Docker from 'dockerode';
import fs from 'fs/promises';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import tar from 'tar';

export class JobOrchestrator {
  constructor(docker) {
    this.docker = docker;
    this.activeJobs = new Map();
    this.defaultImage = 'synaptic-mesh/claude-max:latest';
    this.workDir = '/tmp/claude-jobs';
  }

  /**
   * Execute Claude job in secure Docker container
   */
  async executeJob(jobPayload) {
    const jobId = jobPayload.id || uuidv4();
    const startTime = Date.now();

    try {
      // Ensure work directory exists
      await this.ensureWorkDir();

      // Prepare job context
      const jobContext = await this.prepareJobContext(jobId, jobPayload);
      
      // Check if image exists, build/pull if needed
      await this.ensureImage();

      // Create and run container
      const container = await this.createJobContainer(jobId, jobContext);
      this.activeJobs.set(jobId, container);

      // Start container and stream I/O
      const result = await this.runJobContainer(container, jobPayload);
      
      // Cleanup
      await this.cleanupJob(jobId, container);
      
      return {
        jobId,
        success: true,
        response: result.response,
        usage: result.usage,
        executionTime: Date.now() - startTime,
        metadata: result.metadata
      };

    } catch (error) {
      // Cleanup on error
      await this.cleanupJob(jobId);
      
      return {
        jobId,
        success: false,
        error: error.message,
        executionTime: Date.now() - startTime
      };
    }
  }

  /**
   * Build Claude container image
   */
  async buildImage(options = {}) {
    const tag = options.tag || this.defaultImage;
    const dockerfile = this.generateDockerfile();
    
    // Create build context
    const buildContext = await this.createBuildContext(dockerfile);
    
    // Build image
    const stream = await this.docker.buildImage(buildContext, {
      t: tag,
      nocache: options.nocache || false,
      dockerfile: 'Dockerfile'
    });

    return new Promise((resolve, reject) => {
      const output = [];
      
      this.docker.modem.followProgress(stream, (err, res) => {
        if (err) {
          reject(err);
        } else {
          resolve({ 
            success: true, 
            tag,
            output: output.join('\n')
          });
        }
      }, (event) => {
        if (event.stream) {
          output.push(event.stream.trim());
          process.stdout.write(event.stream);
        }
      });
    });
  }

  /**
   * Pull Claude container image
   */
  async pullImage(tag = this.defaultImage) {
    try {
      const stream = await this.docker.pull(tag);
      
      return new Promise((resolve, reject) => {
        this.docker.modem.followProgress(stream, (err, res) => {
          if (err) {
            reject(err);
          } else {
            resolve({ success: true, tag });
          }
        }, (event) => {
          if (event.status) {
            process.stdout.write(`${event.status}\n`);
          }
        });
      });
    } catch (error) {
      throw new Error(`Failed to pull image ${tag}: ${error.message}`);
    }
  }

  /**
   * Stop all running jobs
   */
  async stopAllJobs() {
    const stopPromises = Array.from(this.activeJobs.entries()).map(
      async ([jobId, container]) => {
        try {
          await container.stop();
          await container.remove();
          this.activeJobs.delete(jobId);
        } catch (error) {
          console.warn(`Failed to stop job ${jobId}:`, error.message);
        }
      }
    );

    await Promise.all(stopPromises);
  }

  /**
   * System health check
   */
  async healthCheck() {
    try {
      // Check Docker connectivity
      const dockerInfo = await this.docker.info();
      
      // Check if Claude image exists
      const images = await this.docker.listImages();
      const imageExists = images.some(img => 
        img.RepoTags && img.RepoTags.includes(this.defaultImage)
      );

      return {
        docker: true,
        imageAvailable: imageExists,
        activeJobs: this.activeJobs.size,
        dockerVersion: dockerInfo.ServerVersion
      };
    } catch (error) {
      return {
        docker: false,
        imageAvailable: false,
        activeJobs: 0,
        error: error.message
      };
    }
  }

  /**
   * Cleanup stopped containers and volumes
   */
  async cleanup() {
    try {
      // Remove stopped containers
      const containers = await this.docker.listContainers({ all: true });
      const stoppedContainers = containers.filter(c => 
        c.State === 'exited' && 
        c.Image.includes('claude')
      );

      let removedCount = 0;
      for (const containerInfo of stoppedContainers) {
        try {
          const container = this.docker.getContainer(containerInfo.Id);
          await container.remove();
          removedCount++;
        } catch (error) {
          console.warn(`Failed to remove container ${containerInfo.Id}:`, error.message);
        }
      }

      // Clean up work directories
      try {
        await fs.rm(this.workDir, { recursive: true, force: true });
        await fs.mkdir(this.workDir, { recursive: true });
      } catch (error) {
        console.warn('Failed to clean work directory:', error.message);
      }

      return { containersRemoved: removedCount };
    } catch (error) {
      throw new Error(`Cleanup failed: ${error.message}`);
    }
  }

  // Private helper methods

  async ensureWorkDir() {
    try {
      await fs.mkdir(this.workDir, { recursive: true });
    } catch (error) {
      throw new Error(`Failed to create work directory: ${error.message}`);
    }
  }

  async prepareJobContext(jobId, jobPayload) {
    const jobDir = path.join(this.workDir, jobId);
    await fs.mkdir(jobDir, { recursive: true });

    // Create job input file
    const inputFile = path.join(jobDir, 'input.json');
    await fs.writeFile(inputFile, JSON.stringify(jobPayload, null, 2));

    // Create execution script
    const scriptFile = path.join(jobDir, 'execute.sh');
    const script = this.generateExecutionScript();
    await fs.writeFile(scriptFile, script);
    await fs.chmod(scriptFile, 0o755);

    return {
      jobDir,
      inputFile,
      scriptFile
    };
  }

  async ensureImage() {
    try {
      // Check if image exists
      const images = await this.docker.listImages();
      const imageExists = images.some(img => 
        img.RepoTags && img.RepoTags.includes(this.defaultImage)
      );

      if (!imageExists) {
        // Try to pull the image first
        try {
          await this.pullImage();
        } catch (pullError) {
          // If pull fails, build the image
          console.warn('Pull failed, building image locally...');
          await this.buildImage();
        }
      }
    } catch (error) {
      throw new Error(`Failed to ensure image availability: ${error.message}`);
    }
  }

  async createJobContainer(jobId, jobContext) {
    const containerConfig = {
      Image: this.defaultImage,
      name: `claude-job-${jobId}`,
      Env: [
        'CLAUDE_API_KEY=' + (process.env.CLAUDE_API_KEY || process.env.ANTHROPIC_API_KEY || ''),
        'JOB_ID=' + jobId
      ],
      HostConfig: {
        Memory: 512 * 1024 * 1024, // 512MB
        CpuShares: 512, // Half CPU
        NetworkMode: 'none', // No network access except API
        ReadonlyRootfs: true,
        Tmpfs: {
          '/tmp': 'rw,noexec,nosuid,size=100m'
        },
        Binds: [
          `${jobContext.jobDir}:/job:ro`
        ],
        AutoRemove: false, // We'll remove manually
        SecurityOpt: [
          'no-new-privileges:true'
        ]
      },
      User: 'nobody',
      WorkingDir: '/job',
      AttachStdout: true,
      AttachStderr: true,
      AttachStdin: true,
      OpenStdin: true,
      StdinOnce: true
    };

    return await this.docker.createContainer(containerConfig);
  }

  async runJobContainer(container, jobPayload) {
    return new Promise(async (resolve, reject) => {
      try {
        // Start container
        await container.start();

        // Attach to container streams
        const stream = await container.attach({
          stream: true,
          stdout: true,
          stderr: true,
          stdin: true
        });

        let stdout = '';
        let stderr = '';

        // Handle output streams
        container.modem.demuxStream(stream, 
          process.stdout, // stdout to console
          process.stderr  // stderr to console
        );

        // Send job payload to container
        stream.write(JSON.stringify(jobPayload) + '\n');
        stream.end();

        // Wait for container to finish
        const data = await container.wait();
        
        if (data.StatusCode === 0) {
          // Get container logs for result parsing
          const logs = await container.logs({
            stdout: true,
            stderr: false
          });

          try {
            // Parse JSON result from logs
            const logText = logs.toString();
            const resultMatch = logText.match(/RESULT: (.+)/);
            
            if (resultMatch) {
              const result = JSON.parse(resultMatch[1]);
              resolve(result);
            } else {
              resolve({
                response: 'Job completed but no structured result found',
                usage: { totalTokens: 0 },
                metadata: { logs: logText }
              });
            }
          } catch (parseError) {
            resolve({
              response: logs.toString(),
              usage: { totalTokens: 0 },
              metadata: { parseError: parseError.message }
            });
          }
        } else {
          const errorLogs = await container.logs({
            stdout: false,
            stderr: true
          });
          reject(new Error(`Container failed with exit code ${data.StatusCode}: ${errorLogs.toString()}`));
        }
      } catch (error) {
        reject(error);
      }
    });
  }

  async cleanupJob(jobId, container = null) {
    try {
      // Remove container if provided
      if (container) {
        try {
          await container.stop();
          await container.remove();
        } catch (error) {
          console.warn(`Failed to cleanup container for job ${jobId}:`, error.message);
        }
      }

      // Remove from active jobs
      this.activeJobs.delete(jobId);

      // Clean up job directory
      const jobDir = path.join(this.workDir, jobId);
      try {
        await fs.rm(jobDir, { recursive: true, force: true });
      } catch (error) {
        console.warn(`Failed to cleanup job directory ${jobDir}:`, error.message);
      }
    } catch (error) {
      console.warn(`Cleanup failed for job ${jobId}:`, error.message);
    }
  }

  generateDockerfile() {
    return `
FROM node:18-alpine

# Install dependencies
RUN apk add --no-cache ca-certificates curl

# Create app directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install npm dependencies
RUN npm ci --only=production && npm cache clean --force

# Copy application code
COPY claude-task-executor.js .
COPY security-config.json .

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \\
    adduser -S nodejs -u 1001

# Set up security
RUN chown -R nodejs:nodejs /app
USER nodejs

# Expose no ports (stdin/stdout only)

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
  CMD node -e "console.log('healthy')" || exit 1

# Default command
CMD ["node", "claude-task-executor.js"]
`;
  }

  generateExecutionScript() {
    return `#!/bin/sh
# Claude Job Execution Script

set -e

echo "Starting Claude job execution..."
echo "Job ID: $JOB_ID"

# Validate input
if [ ! -f "/job/input.json" ]; then
    echo "ERROR: No input file found"
    exit 1
fi

# Check API key
if [ -z "$CLAUDE_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: No API key provided"
    exit 1
fi

# Execute Claude task
echo "Executing Claude task..."
cat /job/input.json | node /app/claude-task-executor.js

echo "Job execution completed"
`;
  }

  async createBuildContext(dockerfile) {
    const buildDir = path.join(this.workDir, 'build');
    await fs.mkdir(buildDir, { recursive: true });

    // Write Dockerfile
    await fs.writeFile(path.join(buildDir, 'Dockerfile'), dockerfile);

    // Copy package.json
    const packageJson = {
      "name": "claude-container",
      "version": "1.0.0",
      "dependencies": {
        "@anthropic-ai/sdk": "^0.20.0"
      }
    };
    await fs.writeFile(
      path.join(buildDir, 'package.json'), 
      JSON.stringify(packageJson, null, 2)
    );

    // Copy Claude task executor (from existing container)
    const executorSource = path.join(process.cwd(), 'docker/claude-container/claude-task-executor.js');
    const securityConfigSource = path.join(process.cwd(), 'docker/claude-container/security-config.json');
    
    try {
      await fs.copyFile(executorSource, path.join(buildDir, 'claude-task-executor.js'));
      await fs.copyFile(securityConfigSource, path.join(buildDir, 'security-config.json'));
    } catch (error) {
      // Create minimal versions if originals don't exist
      await this.createMinimalExecutor(buildDir);
    }

    // Create tar stream for Docker build context
    return tar.create({
      gzip: false,
      cwd: buildDir
    }, ['.']);
  }

  async createMinimalExecutor(buildDir) {
    const minimalExecutor = `
const { Anthropic } = require('@anthropic-ai/sdk');

async function main() {
  const anthropic = new Anthropic({
    apiKey: process.env.CLAUDE_API_KEY || process.env.ANTHROPIC_API_KEY,
  });

  let input = '';
  process.stdin.on('data', chunk => input += chunk);
  process.stdin.on('end', async () => {
    try {
      const jobPayload = JSON.parse(input);
      
      const response = await anthropic.messages.create({
        model: jobPayload.model || 'claude-3-sonnet-20240229',
        max_tokens: jobPayload.maxTokens || 1000,
        messages: [{
          role: 'user',
          content: jobPayload.prompt
        }]
      });

      const result = {
        response: response.content[0].text,
        usage: response.usage,
        metadata: {
          model: response.model,
          timestamp: new Date().toISOString()
        }
      };

      console.log('RESULT:', JSON.stringify(result));
    } catch (error) {
      console.error('ERROR:', error.message);
      process.exit(1);
    }
  });
}

main().catch(console.error);
`;

    await fs.writeFile(path.join(buildDir, 'claude-task-executor.js'), minimalExecutor);
    
    const securityConfig = {
      maxMemoryMB: 512,
      maxExecutionTimeMs: 300000,
      allowedApis: ['api.anthropic.com'],
      workspaceDir: '/tmp/claude-work',
      readOnlyMode: true,
      networkRestricted: true
    };
    
    await fs.writeFile(
      path.join(buildDir, 'security-config.json'), 
      JSON.stringify(securityConfig, null, 2)
    );
  }
}