/// <reference lib="deno.ns" />

import { stats } from "./types.ts";
import { SupabaseDeployer } from "../../../packages/edge/supabase-deploy.ts";

const supabaseDeployer = new SupabaseDeployer();

export async function toggleEdgeFunctions(enable: boolean) {
  console.log(`\n\x1b[38;5;117m⚡ ${enable ? 'Activating' : 'Deactivating'} edge computing system...\x1b[0m`);
  
  stats.edgeFunctionsEnabled = enable;
  if (!enable) {
    stats.activeEdgeFunctions = [];
    stats.deployedUrls = {};
    console.log('\x1b[38;5;209m↳ All edge functions terminated\x1b[0m');
  }
}

export async function viewEdgeFunctionStatus() {
  console.log('\n\x1b[38;5;51m▀▀▀ EDGE FUNCTION STATUS ▀▀▀\x1b[0m');
  if (stats.edgeFunctionsEnabled) {
    if (stats.activeEdgeFunctions.length === 0) {
      console.log('\x1b[38;5;209m! No active functions detected\x1b[0m');
    } else {
      stats.activeEdgeFunctions.forEach(func => {
        console.log(`\x1b[38;5;82m✓ ${func} [RUNNING]\x1b[0m`);
        if (stats.deployedUrls[func]) {
          console.log(`  \x1b[38;5;245m↳ Endpoint: \x1b[38;5;117m${stats.deployedUrls[func]}\x1b[0m`);
        }
      });
    }
  } else {
    console.log('\x1b[38;5;196m✗ Edge computing system offline\x1b[0m');
  }
}

export async function viewEdgeFunctionLogs() {
  console.log('\n\x1b[38;5;51m▀▀▀ SYSTEM LOGS ▀▀▀\x1b[0m');
  if (stats.edgeFunctionsEnabled && stats.activeEdgeFunctions.length > 0) {
    if (stats.selectedProvider === 'supabase') {
      for (const func of stats.activeEdgeFunctions) {
        console.log(`\n\x1b[38;5;117m⚡ ${func} Logs:\x1b[0m`);
        try {
          const logs = await supabaseDeployer.getFunctionLogs(func);
          if (logs.length === 0) {
            console.log('\x1b[38;5;245m  No log entries found\x1b[0m');
          } else {
            logs.forEach(log => {
              const timestamp = new Date().toISOString();
              console.log(`\x1b[38;5;245m[${timestamp}]\x1b[0m ${log}`);
            });
          }
        } catch (error) {
          console.log(`\x1b[38;5;196m  ✗ Error retrieving logs: ${error.message}\x1b[0m`);
        }
      }
    } else {
      console.log('\x1b[38;5;196m✗ Log access not supported for current provider\x1b[0m');
    }
  } else {
    console.log('\x1b[38;5;196m✗ No active functions to monitor\x1b[0m');
  }
}
