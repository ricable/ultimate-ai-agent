import { Command } from 'commander';
import chalk from 'chalk';
import ora from 'ora';
import Table from 'cli-table3';
import { DAGClient } from '../core/dag-client.js';

export function dagCommand() {
  const cmd = new Command('dag');
  
  cmd
    .description('Manage Directed Acyclic Graph workflows')
    .option('-p, --port <port>', 'DAG service port', '7072')
    .option('-h, --host <host>', 'DAG service host', 'localhost');
  
  // List workflows
  cmd
    .command('list')
    .alias('ls')
    .description('List DAG workflows')
    .option('--status <status>', 'Filter by status (pending, running, completed, failed)')
    .action(async (options, command) => {
      const parentOpts = command.parent.opts();
      const spinner = ora('Fetching workflows...').start();
      
      try {
        const client = new DAGClient(parentOpts.host, parentOpts.port);
        const workflows = await client.getWorkflows(options.status);
        
        spinner.stop();
        
        if (workflows.length === 0) {
          console.log(chalk.yellow('No workflows found'));
          return;
        }
        
        const table = new Table({
          head: ['ID', 'Name', 'Status', 'Nodes', 'Progress', 'Created', 'Duration'],
          style: { head: ['cyan'] }
        });
        
        workflows.forEach(workflow => {
          const statusColor = {
            pending: 'yellow',
            running: 'blue',
            completed: 'green',
            failed: 'red'
          }[workflow.status] || 'gray';
          
          table.push([
            workflow.id.substring(0, 8),
            workflow.name,
            chalk[statusColor](workflow.status),
            workflow.nodeCount,
            `${workflow.progress}%`,
            new Date(workflow.created).toLocaleString(),
            formatDuration(workflow.duration)
          ]);
        });
        
        console.log(table.toString());
        console.log(chalk.gray(`Total workflows: ${workflows.length}`));
        
      } catch (error) {
        spinner.fail(chalk.red('Failed to fetch workflows: ' + error.message));
      }
    });
  
  // Create workflow
  cmd
    .command('create <name>')
    .description('Create a new DAG workflow')
    .option('-f, --file <file>', 'Workflow definition file (JSON/YAML)')
    .option('-d, --description <desc>', 'Workflow description')
    .action(async (name, options, command) => {
      const parentOpts = command.parent.opts();
      const spinner = ora('Creating workflow...').start();
      
      try {
        const client = new DAGClient(parentOpts.host, parentOpts.port);
        
        let definition;
        if (options.file) {
          const fs = await import('fs-extra');
          const content = await fs.readFile(options.file, 'utf8');
          
          if (options.file.endsWith('.yaml') || options.file.endsWith('.yml')) {
            const yaml = await import('js-yaml');
            definition = yaml.load(content);
          } else {
            definition = JSON.parse(content);
          }
        } else {
          // Create basic workflow template
          definition = {
            nodes: [
              { id: 'start', type: 'start', name: 'Start' },
              { id: 'end', type: 'end', name: 'End' }
            ],
            edges: [
              { from: 'start', to: 'end' }
            ]
          };
        }
        
        const workflow = await client.createWorkflow({
          name,
          description: options.description,
          definition
        });
        
        spinner.succeed(chalk.green(`Workflow created: ${workflow.id}`));
        
        console.log('\n' + chalk.cyan('Workflow Details:'));
        console.log(chalk.gray('  ID:') + ' ' + workflow.id);
        console.log(chalk.gray('  Name:') + ' ' + workflow.name);
        console.log(chalk.gray('  Nodes:') + ' ' + workflow.nodeCount);
        console.log(chalk.gray('  Edges:') + ' ' + workflow.edgeCount);
        
      } catch (error) {
        spinner.fail(chalk.red('Failed to create workflow: ' + error.message));
      }
    });
  
  // Run workflow
  cmd
    .command('run <workflowId>')
    .description('Execute a DAG workflow')
    .option('-i, --input <input>', 'Input data (JSON string or file)')
    .option('-w, --watch', 'Watch execution progress')
    .option('--async', 'Run asynchronously')
    .action(async (workflowId, options, command) => {
      const parentOpts = command.parent.opts();
      const spinner = ora('Starting workflow execution...').start();
      
      try {
        const client = new DAGClient(parentOpts.host, parentOpts.port);
        
        let inputData = {};
        if (options.input) {
          if (options.input.startsWith('{')) {
            inputData = JSON.parse(options.input);
          } else {
            const fs = await import('fs-extra');
            inputData = await fs.readJson(options.input);
          }
        }
        
        const execution = await client.runWorkflow(workflowId, {
          input: inputData,
          async: options.async
        });
        
        if (options.async) {
          spinner.succeed(chalk.green(`Workflow started: ${execution.id}`));
          console.log(chalk.gray('Run "synaptic dag status <executionId>" to check progress'));
        } else if (options.watch) {
          spinner.text = 'Workflow running...';
          
          // Watch execution progress
          const progressInterval = setInterval(async () => {
            const status = await client.getExecutionStatus(execution.id);
            
            spinner.text = `Progress: ${status.progress}% - ${status.currentNode || 'initializing'}`;
            
            if (status.status === 'completed') {
              clearInterval(progressInterval);
              spinner.succeed(chalk.green('Workflow completed successfully'));
              
              console.log('\n' + chalk.cyan('Execution Summary:'));
              console.log(chalk.gray('  Duration:') + ' ' + formatDuration(status.duration));
              console.log(chalk.gray('  Nodes executed:') + ' ' + status.nodesExecuted);
              
              if (status.output) {
                console.log('\n' + chalk.cyan('Output:'));
                console.log(JSON.stringify(status.output, null, 2));
              }
              
            } else if (status.status === 'failed') {
              clearInterval(progressInterval);
              spinner.fail(chalk.red('Workflow failed: ' + status.error));
              
              if (status.failedNode) {
                console.log(chalk.gray('Failed at node:') + ' ' + status.failedNode);
              }
            }
          }, 1000);
        } else {
          spinner.succeed(chalk.green('Workflow completed'));
          
          if (execution.output) {
            console.log('\n' + chalk.cyan('Output:'));
            console.log(JSON.stringify(execution.output, null, 2));
          }
        }
        
      } catch (error) {
        spinner.fail(chalk.red('Failed to run workflow: ' + error.message));
      }
    });
  
  // Show workflow status
  cmd
    .command('status <executionId>')
    .description('Check workflow execution status')
    .action(async (executionId, options, command) => {
      const parentOpts = command.parent.opts();
      const spinner = ora('Fetching execution status...').start();
      
      try {
        const client = new DAGClient(parentOpts.host, parentOpts.port);
        const status = await client.getExecutionStatus(executionId);
        
        spinner.stop();
        
        console.log(chalk.cyan('\nExecution Status:'));
        console.log(chalk.gray('  ID:') + ' ' + executionId);
        console.log(chalk.gray('  Status:') + ' ' + getStatusColor(status.status));
        console.log(chalk.gray('  Progress:') + ' ' + `${status.progress}%`);
        console.log(chalk.gray('  Current node:') + ' ' + (status.currentNode || 'N/A'));
        console.log(chalk.gray('  Duration:') + ' ' + formatDuration(status.duration));
        
        if (status.nodes) {
          console.log('\n' + chalk.cyan('Node Status:'));
          
          const table = new Table({
            head: ['Node', 'Status', 'Duration', 'Output'],
            style: { head: ['cyan'] }
          });
          
          Object.entries(status.nodes).forEach(([nodeId, nodeStatus]) => {
            table.push([
              nodeId,
              getStatusColor(nodeStatus.status),
              formatDuration(nodeStatus.duration),
              nodeStatus.output ? 'Yes' : 'No'
            ]);
          });
          
          console.log(table.toString());
        }
        
      } catch (error) {
        spinner.fail(chalk.red('Failed to fetch status: ' + error.message));
      }
    });
  
  // Visualize workflow
  cmd
    .command('visualize <workflowId>')
    .alias('viz')
    .description('Visualize DAG workflow structure')
    .option('-f, --format <format>', 'Output format (ascii, dot, mermaid)', 'ascii')
    .option('-o, --output <file>', 'Output to file')
    .action(async (workflowId, options, command) => {
      const parentOpts = command.parent.opts();
      const spinner = ora('Generating visualization...').start();
      
      try {
        const client = new DAGClient(parentOpts.host, parentOpts.port);
        const visualization = await client.visualizeWorkflow(workflowId, {
          format: options.format
        });
        
        spinner.stop();
        
        if (options.output) {
          const fs = await import('fs-extra');
          await fs.writeFile(options.output, visualization);
          console.log(chalk.green(`Visualization saved to: ${options.output}`));
        } else {
          console.log('\n' + chalk.cyan('Workflow Visualization:'));
          console.log(visualization);
        }
        
      } catch (error) {
        spinner.fail(chalk.red('Failed to visualize workflow: ' + error.message));
      }
    });
  
  // Delete workflow
  cmd
    .command('delete <workflowId>')
    .alias('rm')
    .description('Delete a DAG workflow')
    .option('-f, --force', 'Force deletion without confirmation')
    .action(async (workflowId, options, command) => {
      const parentOpts = command.parent.opts();
      const spinner = ora('Deleting workflow...').start();
      
      try {
        const client = new DAGClient(parentOpts.host, parentOpts.port);
        
        // TODO: Add confirmation prompt if not forced
        
        await client.deleteWorkflow(workflowId);
        spinner.succeed(chalk.green(`Workflow deleted: ${workflowId}`));
        
      } catch (error) {
        spinner.fail(chalk.red('Failed to delete workflow: ' + error.message));
      }
    });
    
  return cmd;
}

function formatDuration(ms) {
  if (!ms) return 'N/A';
  
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  
  if (hours > 0) {
    return `${hours}h ${minutes % 60}m`;
  } else if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`;
  } else {
    return `${seconds}s`;
  }
}

function getStatusColor(status) {
  const colors = {
    pending: chalk.yellow(status),
    running: chalk.blue(status),
    completed: chalk.green(status),
    failed: chalk.red(status),
    cancelled: chalk.gray(status)
  };
  
  return colors[status] || chalk.white(status);
}