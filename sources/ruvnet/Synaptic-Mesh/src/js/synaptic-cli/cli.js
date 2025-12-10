#!/usr/bin/env node

/**
 * Synaptic Neural Mesh CLI
 * Revolutionary AI orchestration with neural mesh topology
 */

import('./src/cli/main.js').catch(err => {
  console.error('Failed to start Synaptic CLI:', err);
  process.exit(1);
});