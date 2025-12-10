/// <reference lib="deno.ns" />

import { MCPServer } from "../../packages/core/server.ts";
import { ServerInfo } from "../../packages/core/types.ts";
import { SupabaseDeployer } from "../../packages/edge/supabase-deploy.ts";
import { readLines } from "https://deno.land/std@0.224.0/io/mod.ts";
import { 
  serverInfo, 
  stats, 
  edgeProviders 
} from "./lib/types.ts";
import { 
  checkEdgeProviders,
  configureProvider 
} from "./lib/edge-providers.ts";
import {
  toggleEdgeFunctions,
  viewEdgeFunctionStatus,
  viewEdgeFunctionLogs
} from "./lib/edge-functions.ts";
import { displayServerStatus } from "./lib/server-status.ts";

const server = new MCPServer(serverInfo);
const supabaseDeployer = new SupabaseDeployer();

function formatJSON(obj: any, indent = 0): string {
  const spaces = '  '.repeat(indent);
  const nextSpaces = '  '.repeat(indent + 1);
  const colors = {
    key: '\x1b[38;5;51m',     // Cyan for keys
    string: '\x1b[38;5;87m',  // Light cyan for strings
    number: '\x1b[38;5;207m', // Pink for numbers
    boolean: '\x1b[38;5;219m',// Light pink for booleans
    null: '\x1b[38;5;243m',   // Dark gray for null
    bracket: '\x1b[38;5;39m', // Blue for brackets
    comma: '\x1b[38;5;45m',   // Bright cyan for commas
    reset: '\x1b[0m'
  };
  
  if (Array.isArray(obj)) {
    if (obj.length === 0) return `${colors.bracket}[]${colors.reset}`;
    const items = obj.map((item, index) => {
      const comma = index < obj.length - 1 ? `${colors.comma},${colors.reset}` : '';
      if (typeof item === 'string') {
        return `${nextSpaces}${colors.string}"${item}"${colors.reset}${comma}`;
      }
      return `${formatJSON(item, indent + 1)}${comma}`;
    }).join('\n');
    return `${colors.bracket}[\n${colors.reset}${items}\n${spaces}${colors.bracket}]${colors.reset}`;
  }
  
  if (typeof obj === 'object' && obj !== null) {
    const entries = Object.entries(obj);
    if (entries.length === 0) return `${colors.bracket}{}${colors.reset}`;
    const formattedEntries = entries.map(([key, value], index) => {
      const comma = index < entries.length - 1 ? `${colors.comma},${colors.reset}` : '';
      let formattedValue;
      switch (typeof value) {
        case 'string':
          formattedValue = `${colors.string}"${value}"${colors.reset}`;
          break;
        case 'number':
          formattedValue = `${colors.number}${value}${colors.reset}`;
          break;
        case 'boolean':
          formattedValue = `${colors.boolean}${value}${colors.reset}`;
          break;
        case 'object':
          if (value === null) {
            formattedValue = `${colors.null}null${colors.reset}`;
          } else {
            formattedValue = formatJSON(value, indent + 1);
          }
          break;
        default:
          formattedValue = String(value);
      }
      return `${nextSpaces}${colors.key}"${key}"${colors.reset}: ${formattedValue}${comma}`;
    }).join('\n');
    return `${colors.bracket}{\n${colors.reset}${formattedEntries}\n${spaces}${colors.bracket}}${colors.reset}`;
  }
  
  if (typeof obj === 'string') {
    return `${colors.string}"${obj}"${colors.reset}`;
  }
  if (typeof obj === 'number') {
    return `${colors.number}${obj}${colors.reset}`;
  }
  if (typeof obj === 'boolean') {
    return `${colors.boolean}${obj}${colors.reset}`;
  }
  if (obj === null) {
    return `${colors.null}null${colors.reset}`;
  }
  
  return String(obj);
}

async function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function displayStartupSequence() {
  console.clear();
  const steps = [
    '⚡ Initializing AI Federation Network...',
    '🔌 Establishing network protocols...',
    '🤖 Loading ML models...',
    '⚙️  Configuring runtime environment...',
    '🌐 Scanning edge computing matrix...',
    '📡 Starting distributed system...'
  ];

  for (const step of steps) {
    console.log(`\x1b[38;5;117m${step}\x1b[0m`);
    await sleep(300);
  }
  console.log('\n\x1b[38;5;82m✓ System initialization complete\x1b[0m\n');
  await sleep(500);
}

async function handleKeypress() {
  for await (const line of readLines(Deno.stdin)) {
    const choice = line.trim();
    
    switch (choice) {
      case '1':
        console.log('\n\x1b[38;5;51m▀▀▀ NETWORK STATUS ▀▀▀\x1b[0m');
        console.log(`Active Connections: \x1b[38;5;117m${stats.connections}\x1b[0m`);
        break;
        
      case '2':
        console.log('\n\x1b[38;5;51m╔══════════════════════════════════╗\x1b[0m');
        console.log('\x1b[38;5;51m║     SYSTEM CORE INFORMATION      ║\x1b[0m');
        console.log('\x1b[38;5;51m╚══════════════════════════════════╝\x1b[0m');
        console.log('\x1b[38;5;117m<< INITIATING DATA STREAM >>\x1b[0m\n');
        console.log(formatJSON(serverInfo));
        console.log('\n\x1b[38;5;117m<< END TRANSMISSION >>\x1b[0m');
        break;
        
      case '3':
        console.log('\n\x1b[38;5;51m▀▀▀ SYSTEM CAPABILITIES ▀▀▀\x1b[0m');
        console.log(formatJSON(serverInfo.capabilities));
        break;
        
      case '4':
        await configureProvider();
        break;
        
      case '5':
        if (!stats.edgeFunctionsEnabled) {
          console.log('\n\x1b[38;5;196m✗ Edge computing system offline\x1b[0m');
          console.log('\x1b[38;5;245mUse option [4] to configure and enable edge functions\x1b[0m');
          break;
        }
        console.log('\n\x1b[38;5;51m▀▀▀ FUNCTION DEPLOYMENT ▀▀▀\x1b[0m');
        console.log('\x1b[38;5;239m┌─────────────────────┐\x1b[0m');
        console.log('\x1b[38;5;239m│\x1b[0m 1. intent-detection \x1b[38;5;239m│\x1b[0m');
        console.log('\x1b[38;5;239m│\x1b[0m 2. meeting-info     \x1b[38;5;239m│\x1b[0m');
        console.log('\x1b[38;5;239m│\x1b[0m 3. webhook-handler  \x1b[38;5;239m│\x1b[0m');
        console.log('\x1b[38;5;239m│\x1b[0m 4. Cancel          \x1b[38;5;239m│\x1b[0m');
        console.log('\x1b[38;5;239m└─────────────────────┘\x1b[0m');
        
        for await (const deployChoice of readLines(Deno.stdin)) {
          switch (deployChoice.trim()) {
            case '1':
              await supabaseDeployer.deployFunction('intent-detection');
              break;
            case '2':
              await supabaseDeployer.deployFunction('meeting-info');
              break;
            case '3':
              await supabaseDeployer.deployFunction('webhook-handler');
              break;
            case '4':
            default:
              break;
          }
          break;
        }
        break;
        
      case '6':
        await viewEdgeFunctionStatus();
        break;
        
      case '7':
        await viewEdgeFunctionLogs();
        break;
        
      case '8':
        if (stats.edgeFunctionsEnabled) {
          const functions = await supabaseDeployer.listDeployedFunctions();
          console.log('\n\x1b[38;5;51m▀▀▀ DEPLOYED FUNCTIONS ▀▀▀\x1b[0m');
          if (functions.length === 0) {
            console.log('\x1b[38;5;245mNo functions currently deployed\x1b[0m');
          } else {
            functions.forEach(f => console.log(`\x1b[38;5;82m✓ ${f}\x1b[0m`));
          }
        } else {
          console.log('\n\x1b[38;5;196m✗ Edge computing system offline\x1b[0m');
        }
        break;
        
      case '9':
        displayServerStatus();
        break;
        
      default:
        console.log('\n\x1b[38;5;196m✗ Invalid command\x1b[0m');
        break;
    }
    
    console.log('\n\x1b[38;5;51m[SYS]>\x1b[38;5;87m AWAITING INPUT\x1b[38;5;51m [1-9]:\x1b[0m');
  }
}

// Create request handler
async function handler(req: Request): Promise<Response> {
  return new Response("AI Federation Network Online", { status: 200 });
}

// Initialize everything in sequence
await displayStartupSequence();
await checkEdgeProviders();
displayServerStatus();

// Start the HTTP server with logging disabled
const httpServer = Deno.serve({ 
  port: 3000,
  onListen: ({ port }) => {
    console.log(`\x1b[38;5;245m🌐 Network interface active on port ${port}\x1b[0m`);
  }
}, handler);

// Start keyboard input handler
handleKeypress();

await httpServer.finished;
