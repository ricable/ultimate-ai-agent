import fs from 'fs-extra';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export function validateConfig(config) {
  const errors = [];
  
  // Validate project section
  if (!config.project?.name) {
    errors.push('Project name is required');
  }
  
  if (!['basic', 'advanced', 'enterprise'].includes(config.project?.template)) {
    errors.push('Invalid project template');
  }
  
  // Validate mesh section
  if (!['mesh', 'hierarchical', 'ring', 'star'].includes(config.mesh?.topology)) {
    errors.push('Invalid mesh topology');
  }
  
  if (!config.mesh?.defaultAgents || config.mesh.defaultAgents < 1 || config.mesh.defaultAgents > 100) {
    errors.push('Default agents must be between 1 and 100');
  }
  
  // Validate ports
  const ports = [
    config.mesh?.coordinationPort,
    config.neural?.port,
    config.dag?.port,
    config.peer?.port,
    config.features?.mcpPort,
    config.features?.webuiPort
  ].filter(Boolean);
  
  for (const port of ports) {
    if (!isValidPort(port)) {
      errors.push(`Invalid port: ${port}`);
    }
  }
  
  // Check for port conflicts
  const uniquePorts = new Set(ports);
  if (uniquePorts.size !== ports.length) {
    errors.push('Port conflicts detected');
  }
  
  return {
    valid: errors.length === 0,
    errors
  };
}

export async function validateEnvironment() {
  const errors = [];
  const warnings = [];
  
  try {
    // Check Node.js version
    const nodeVersion = process.version;
    const majorVersion = parseInt(nodeVersion.slice(1).split('.')[0]);
    
    if (majorVersion < 20) {
      errors.push(`Node.js 20+ required, found ${nodeVersion}`);
    }
    
    // Check for required dependencies
    const requiredCommands = ['npm', 'node'];
    
    for (const cmd of requiredCommands) {
      try {
        await execAsync(`which ${cmd}`);
      } catch {
        try {
          await execAsync(`where ${cmd}`); // Windows
        } catch {
          errors.push(`Required command not found: ${cmd}`);
        }
      }
    }
    
    // Check disk space
    try {
      const stats = await fs.stat(process.cwd());
      // Minimum 100MB free space recommended
      // This is a simplified check - in production you'd want proper disk space checking
    } catch (error) {
      warnings.push('Could not check disk space');
    }
    
    // Check write permissions
    try {
      const testFile = '.synaptic-test-write';
      await fs.writeFile(testFile, 'test');
      await fs.remove(testFile);
    } catch {
      errors.push('No write permission in current directory');
    }
    
    // Check for conflicting processes (simplified)
    const defaultPorts = [7070, 7071, 7072, 7073, 3000];
    for (const port of defaultPorts) {
      if (await isPortInUse(port)) {
        warnings.push(`Port ${port} is already in use`);
      }
    }
    
  } catch (error) {
    errors.push(`Environment validation failed: ${error.message}`);
  }
  
  return {
    valid: errors.length === 0,
    errors,
    warnings
  };
}

function isValidPort(port) {
  return Number.isInteger(port) && port >= 1024 && port <= 65535;
}

async function isPortInUse(port) {
  try {
    const { exec } = await import('child_process');
    const { promisify } = await import('util');
    const execAsync = promisify(exec);
    
    // This is a simplified check - in production you'd want more robust port checking
    if (process.platform === 'win32') {
      const { stdout } = await execAsync(`netstat -an | findstr :${port}`);
      return stdout.length > 0;
    } else {
      const { stdout } = await execAsync(`lsof -i :${port}`);
      return stdout.length > 0;
    }
  } catch {
    // If we can't check, assume port is free
    return false;
  }
}

export function validateWorkflowDefinition(definition) {
  const errors = [];
  
  if (!definition.nodes || !Array.isArray(definition.nodes)) {
    errors.push('Workflow must have nodes array');
  } else {
    // Check for required start/end nodes
    const hasStart = definition.nodes.some(node => node.type === 'start');
    const hasEnd = definition.nodes.some(node => node.type === 'end');
    
    if (!hasStart) errors.push('Workflow must have a start node');
    if (!hasEnd) errors.push('Workflow must have an end node');
    
    // Validate node IDs are unique
    const nodeIds = definition.nodes.map(node => node.id);
    const uniqueIds = new Set(nodeIds);
    if (uniqueIds.size !== nodeIds.length) {
      errors.push('Node IDs must be unique');
    }
  }
  
  if (!definition.edges || !Array.isArray(definition.edges)) {
    errors.push('Workflow must have edges array');
  } else {
    // Validate edges reference existing nodes
    const nodeIds = new Set(definition.nodes?.map(node => node.id) || []);
    
    for (const edge of definition.edges) {
      if (!nodeIds.has(edge.from)) {
        errors.push(`Edge references unknown node: ${edge.from}`);
      }
      if (!nodeIds.has(edge.to)) {
        errors.push(`Edge references unknown node: ${edge.to}`);
      }
    }
  }
  
  return {
    valid: errors.length === 0,
    errors
  };
}

export function validateNeuralModelConfig(config) {
  const errors = [];
  
  if (!config.name || typeof config.name !== 'string') {
    errors.push('Model name is required');
  }
  
  if (!['classification', 'regression', 'generative'].includes(config.type)) {
    errors.push('Invalid model type');
  }
  
  if (!['mlp', 'cnn', 'rnn', 'transformer'].includes(config.architecture)) {
    errors.push('Invalid model architecture');
  }
  
  if (!config.layers || !Array.isArray(config.layers) || config.layers.length === 0) {
    errors.push('Model must have at least one layer');
  } else {
    for (const layer of config.layers) {
      if (!Number.isInteger(layer) || layer < 1) {
        errors.push('Layer sizes must be positive integers');
        break;
      }
    }
  }
  
  return {
    valid: errors.length === 0,
    errors
  };
}