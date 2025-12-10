import fs from 'fs-extra';
import path from 'path';

const CONFIG_DIR = '.synaptic';
const CONFIG_FILE = 'config.json';

export async function loadConfig(projectPath = process.cwd()) {
  try {
    const configPath = path.join(projectPath, CONFIG_DIR, CONFIG_FILE);
    
    if (!(await fs.pathExists(configPath))) {
      return null;
    }
    
    const config = await fs.readJson(configPath);
    return config;
  } catch (error) {
    throw new Error(`Failed to load configuration: ${error.message}`);
  }
}

export async function saveConfig(config, projectPath = process.cwd()) {
  try {
    const configDir = path.join(projectPath, CONFIG_DIR);
    const configPath = path.join(configDir, CONFIG_FILE);
    
    await fs.ensureDir(configDir);
    await fs.writeJson(configPath, config, { spaces: 2 });
  } catch (error) {
    throw new Error(`Failed to save configuration: ${error.message}`);
  }
}

export async function getConfigPath(projectPath = process.cwd()) {
  return path.join(projectPath, CONFIG_DIR);
}

export async function configExists(projectPath = process.cwd()) {
  const configPath = path.join(projectPath, CONFIG_DIR, CONFIG_FILE);
  return fs.pathExists(configPath);
}