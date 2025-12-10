import { Command } from 'commander';
import chalk from 'chalk';
import ora from 'ora';
import Table from 'cli-table3';
import { NeuralClient } from '../core/neural-client.js';

export function neuralCommand() {
  const cmd = new Command('neural');
  
  cmd
    .description('Manage neural network models and training')
    .option('-p, --port <port>', 'Neural service port', '7071')
    .option('-h, --host <host>', 'Neural service host', 'localhost');
  
  // List models
  cmd
    .command('list')
    .alias('ls')
    .description('List available neural models')
    .option('--type <type>', 'Filter by model type')
    .action(async (options, command) => {
      const parentOpts = command.parent.opts();
      const spinner = ora('Fetching neural models...').start();
      
      try {
        const client = new NeuralClient(parentOpts.host, parentOpts.port);
        const models = await client.getModels(options.type);
        
        spinner.stop();
        
        if (models.length === 0) {
          console.log(chalk.yellow('No neural models found'));
          return;
        }
        
        const table = new Table({
          head: ['ID', 'Name', 'Type', 'Architecture', 'Status', 'Accuracy', 'Size'],
          style: { head: ['cyan'] }
        });
        
        models.forEach(model => {
          table.push([
            model.id.substring(0, 8),
            model.name,
            model.type,
            model.architecture,
            model.status === 'trained' ? chalk.green(model.status) : chalk.yellow(model.status),
            model.accuracy ? `${(model.accuracy * 100).toFixed(2)}%` : 'N/A',
            formatSize(model.size)
          ]);
        });
        
        console.log(table.toString());
        console.log(chalk.gray(`Total models: ${models.length}`));
        
      } catch (error) {
        spinner.fail(chalk.red('Failed to fetch models: ' + error.message));
      }
    });
  
  // Create model
  cmd
    .command('create <name>')
    .description('Create a new neural model')
    .option('-t, --type <type>', 'Model type (classification, regression, generative)', 'classification')
    .option('-a, --architecture <arch>', 'Architecture (mlp, cnn, rnn, transformer)', 'mlp')
    .option('--layers <layers>', 'Layer configuration (comma-separated)', '128,64,32')
    .option('--activation <activation>', 'Activation function', 'relu')
    .action(async (name, options, command) => {
      const parentOpts = command.parent.opts();
      const spinner = ora('Creating neural model...').start();
      
      try {
        const client = new NeuralClient(parentOpts.host, parentOpts.port);
        
        const layers = options.layers.split(',').map(l => parseInt(l));
        
        const model = await client.createModel({
          name,
          type: options.type,
          architecture: options.architecture,
          layers,
          activation: options.activation
        });
        
        spinner.succeed(chalk.green(`Model created: ${model.id}`));
        
        console.log('\n' + chalk.cyan('Model Details:'));
        console.log(chalk.gray('  ID:') + ' ' + model.id);
        console.log(chalk.gray('  Name:') + ' ' + model.name);
        console.log(chalk.gray('  Type:') + ' ' + model.type);
        console.log(chalk.gray('  Architecture:') + ' ' + model.architecture);
        console.log(chalk.gray('  Layers:') + ' ' + model.layers.join(' → '));
        
      } catch (error) {
        spinner.fail(chalk.red('Failed to create model: ' + error.message));
      }
    });
  
  // Train model
  cmd
    .command('train <modelId>')
    .description('Train a neural model')
    .option('-d, --dataset <dataset>', 'Training dataset path or ID')
    .option('-e, --epochs <epochs>', 'Number of training epochs', '100')
    .option('-b, --batch-size <size>', 'Batch size', '32')
    .option('-l, --learning-rate <rate>', 'Learning rate', '0.001')
    .option('--validation-split <split>', 'Validation split ratio', '0.2')
    .action(async (modelId, options, command) => {
      const parentOpts = command.parent.opts();
      const spinner = ora('Starting training...').start();
      
      try {
        const client = new NeuralClient(parentOpts.host, parentOpts.port);
        
        const training = await client.trainModel(modelId, {
          dataset: options.dataset,
          epochs: parseInt(options.epochs),
          batchSize: parseInt(options.batchSize),
          learningRate: parseFloat(options.learningRate),
          validationSplit: parseFloat(options.validationSplit)
        });
        
        spinner.text = 'Training in progress...';
        
        // Monitor training progress
        const progressInterval = setInterval(async () => {
          const status = await client.getTrainingStatus(training.id);
          
          if (status.status === 'completed') {
            clearInterval(progressInterval);
            spinner.succeed(chalk.green('Training completed'));
            
            console.log('\n' + chalk.cyan('Training Results:'));
            console.log(chalk.gray('  Final loss:') + ' ' + status.finalLoss.toFixed(4));
            console.log(chalk.gray('  Final accuracy:') + ' ' + `${(status.finalAccuracy * 100).toFixed(2)}%`);
            console.log(chalk.gray('  Training time:') + ' ' + formatDuration(status.duration));
            
          } else if (status.status === 'failed') {
            clearInterval(progressInterval);
            spinner.fail(chalk.red('Training failed: ' + status.error));
            
          } else {
            spinner.text = `Training... Epoch ${status.currentEpoch}/${options.epochs} - Loss: ${status.currentLoss.toFixed(4)}`;
          }
        }, 1000);
        
      } catch (error) {
        spinner.fail(chalk.red('Failed to start training: ' + error.message));
      }
    });
  
  // Evaluate model
  cmd
    .command('evaluate <modelId>')
    .description('Evaluate a trained model')
    .option('-d, --dataset <dataset>', 'Test dataset path or ID')
    .option('--metrics <metrics>', 'Metrics to compute (comma-separated)', 'accuracy,precision,recall,f1')
    .action(async (modelId, options, command) => {
      const parentOpts = command.parent.opts();
      const spinner = ora('Evaluating model...').start();
      
      try {
        const client = new NeuralClient(parentOpts.host, parentOpts.port);
        
        const metrics = options.metrics.split(',');
        const results = await client.evaluateModel(modelId, {
          dataset: options.dataset,
          metrics
        });
        
        spinner.succeed(chalk.green('Evaluation completed'));
        
        console.log('\n' + chalk.cyan('Evaluation Results:'));
        
        const table = new Table({
          head: ['Metric', 'Value'],
          style: { head: ['cyan'] }
        });
        
        Object.entries(results.metrics).forEach(([metric, value]) => {
          table.push([
            metric,
            typeof value === 'number' ? value.toFixed(4) : value
          ]);
        });
        
        console.log(table.toString());
        
        if (results.confusionMatrix) {
          console.log('\n' + chalk.cyan('Confusion Matrix:'));
          displayConfusionMatrix(results.confusionMatrix);
        }
        
      } catch (error) {
        spinner.fail(chalk.red('Failed to evaluate model: ' + error.message));
      }
    });
  
  // Predict with model
  cmd
    .command('predict <modelId>')
    .description('Make predictions with a model')
    .option('-i, --input <input>', 'Input data (JSON string or file path)')
    .option('-o, --output <output>', 'Output file path')
    .option('--batch', 'Process as batch prediction')
    .action(async (modelId, options, command) => {
      const parentOpts = command.parent.opts();
      const spinner = ora('Making predictions...').start();
      
      try {
        const client = new NeuralClient(parentOpts.host, parentOpts.port);
        
        // Parse input
        let inputData;
        if (options.input.startsWith('{') || options.input.startsWith('[')) {
          inputData = JSON.parse(options.input);
        } else {
          // Load from file
          const fs = await import('fs-extra');
          inputData = await fs.readJson(options.input);
        }
        
        const predictions = await client.predict(modelId, {
          data: inputData,
          batch: options.batch
        });
        
        spinner.succeed(chalk.green('Predictions completed'));
        
        if (options.output) {
          const fs = await import('fs-extra');
          await fs.writeJson(options.output, predictions, { spaces: 2 });
          console.log(chalk.gray(`Results saved to: ${options.output}`));
        } else {
          console.log('\n' + chalk.cyan('Predictions:'));
          console.log(JSON.stringify(predictions, null, 2));
        }
        
      } catch (error) {
        spinner.fail(chalk.red('Failed to make predictions: ' + error.message));
      }
    });
  
  // Delete model
  cmd
    .command('delete <modelId>')
    .alias('rm')
    .description('Delete a neural model')
    .option('-f, --force', 'Force deletion without confirmation')
    .action(async (modelId, options, command) => {
      const parentOpts = command.parent.opts();
      const spinner = ora('Deleting model...').start();
      
      try {
        const client = new NeuralClient(parentOpts.host, parentOpts.port);
        
        // TODO: Add confirmation prompt if not forced
        
        await client.deleteModel(modelId);
        spinner.succeed(chalk.green(`Model deleted: ${modelId}`));
        
      } catch (error) {
        spinner.fail(chalk.red('Failed to delete model: ' + error.message));
      }
    });
  
  // Export model
  cmd
    .command('export <modelId>')
    .description('Export a neural model')
    .option('-f, --format <format>', 'Export format (onnx, tensorflow, pytorch)', 'onnx')
    .option('-o, --output <path>', 'Output file path')
    .action(async (modelId, options, command) => {
      const parentOpts = command.parent.opts();
      const spinner = ora('Exporting model...').start();
      
      try {
        const client = new NeuralClient(parentOpts.host, parentOpts.port);
        
        const exportPath = await client.exportModel(modelId, {
          format: options.format,
          outputPath: options.output
        });
        
        spinner.succeed(chalk.green('Model exported'));
        console.log(chalk.gray(`Exported to: ${exportPath}`));
        
      } catch (error) {
        spinner.fail(chalk.red('Failed to export model: ' + error.message));
      }
    });
    
  return cmd;
}

function formatSize(bytes) {
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = bytes;
  let unit = 0;
  
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit++;
  }
  
  return `${size.toFixed(2)} ${units[unit]}`;
}

function formatDuration(ms) {
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

function displayConfusionMatrix(matrix) {
  const table = new Table({
    head: ['', ...matrix.labels],
    style: { head: ['cyan'] }
  });
  
  matrix.data.forEach((row, i) => {
    table.push([
      chalk.cyan(matrix.labels[i]),
      ...row.map(val => val.toString())
    ]);
  });
  
  console.log(table.toString());
}