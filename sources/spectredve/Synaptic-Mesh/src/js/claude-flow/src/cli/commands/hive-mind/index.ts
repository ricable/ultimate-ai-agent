#!/usr/bin/env node
/**
 * Hive Mind CLI Commands Index
 * 
 * Main entry point for all Hive Mind CLI commands
 */

import { Command } from 'commander';
import { initCommand } from './init.js';
import { spawnCommand } from './spawn.js';
import { statusCommand } from './status.js';
import { taskCommand } from './task.js';
import { wizardCommand } from './wizard.js';

export const hiveMindCommand = new Command('hive-mind')
  .description('Hive Mind collective intelligence swarm management')
  .addCommand(initCommand)
  .addCommand(spawnCommand)
  .addCommand(statusCommand)
  .addCommand(taskCommand)
  .addCommand(wizardCommand);

// Export individual commands for testing
export {
  initCommand,
  spawnCommand,
  statusCommand,
  taskCommand,
  wizardCommand
};