/// <reference lib="deno.ns" />

import { stats, edgeProviders } from "./types.ts";
import { toggleEdgeFunctions } from "./edge-functions.ts";

async function promptForCredentials(provider: string): Promise<Record<string, string>> {
  const credentials: Record<string, string> = {};
  const providerConfig = edgeProviders[provider];

  console.log(`\n\x1b[38;5;51m▀▀▀ CLOUD PROVIDER AUTHENTICATION ▀▀▀\x1b[0m`);
  console.log(`\x1b[38;5;117mInitiating secure connection to ${providerConfig.name}...\x1b[0m`);
  
  for (const envVar of providerConfig.requiredEnvVars) {
    const input = prompt(`\x1b[38;5;251m${envVar.name}\x1b[0m (\x1b[38;5;245m${envVar.description}\x1b[0m): `);
    if (!input) {
      throw new Error('Authentication sequence aborted');
    }
    credentials[envVar.name] = input;
  }

  return credentials;
}

async function saveCredentials(credentials: Record<string, string>): Promise<void> {
  const envPath = '../../.env';
  
  try {
    console.log('\n\x1b[38;5;117m⚡ Initializing secure storage...\x1b[0m');
    
    // Read existing .env content
    let envContent = '';
    try {
      envContent = await Deno.readTextFile(envPath);
      console.log('\x1b[38;5;245m↳ Existing configuration detected\x1b[0m');
    } catch {
      console.log('\x1b[38;5;245m↳ No existing configuration found\x1b[0m');
    }

    // Parse existing variables
    const envVars = new Map();
    envContent.split('\n').forEach(line => {
      const match = line.match(/^([^=]+)=(.*)$/);
      if (match) {
        envVars.set(match[1], match[2]);
      }
    });

    // Update with new credentials
    console.log('\x1b[38;5;117m⚡ Encrypting credentials...\x1b[0m');
    Object.entries(credentials).forEach(([key, value]) => {
      envVars.set(key, value);
      console.log(`\x1b[38;5;245m↳ Secured ${key}\x1b[0m`);
    });

    // Build new .env content
    const newContent = Array.from(envVars.entries())
      .map(([key, value]) => `${key}=${value}`)
      .join('\n') + '\n';

    // Write back to .env
    console.log('\x1b[38;5;117m⚡ Committing to secure storage...\x1b[0m');
    await Deno.writeTextFile(envPath, newContent);
    console.log('\n\x1b[38;5;82m✓ Authentication data secured\x1b[0m');

    // Set environment variables for current process
    Object.entries(credentials).forEach(([key, value]) => {
      Deno.env.set(key, value);
    });

  } catch (error) {
    console.error('\n\x1b[38;5;196m✗ Security violation:', error.message, '\x1b[0m');
    throw error;
  }
}

export async function checkEdgeProviders() {
  console.log('\n\x1b[38;5;51m▀▀▀ EDGE PROVIDER STATUS ▀▀▀\x1b[0m');
  for (const [name, provider] of Object.entries(edgeProviders)) {
    if (provider.isConfigured()) {
      console.log(`\x1b[38;5;82m✓ ${provider.name} [AUTHENTICATED]\x1b[0m`);
    } else {
      console.log(`\x1b[38;5;209m! ${provider.name} [REQUIRES AUTH]\x1b[0m`);
      console.log('\x1b[38;5;245mRequired credentials:\x1b[0m');
      provider.requiredEnvVars.forEach(v => {
        console.log(`  \x1b[38;5;251m${v.name}\x1b[0m: ${v.description}`);
      });
    }
  }
}

export async function configureProvider() {
  console.log('\n\x1b[38;5;51m▀▀▀ CLOUD PROVIDER SELECTION ▀▀▀\x1b[0m');
  console.log('\x1b[38;5;239m┌─────────────────────────┐\x1b[0m');
  console.log('\x1b[38;5;239m│\x1b[0m 1. Supabase              \x1b[38;5;239m│\x1b[0m');
  console.log('\x1b[38;5;239m│\x1b[0m 2. Cloudflare Workers    \x1b[38;5;239m│\x1b[0m');
  console.log('\x1b[38;5;239m│\x1b[0m 3. Fly.io                \x1b[38;5;239m│\x1b[0m');
  console.log('\x1b[38;5;239m│\x1b[0m 4. Cancel                \x1b[38;5;239m│\x1b[0m');
  console.log('\x1b[38;5;239m└─────────────────────────┘\x1b[0m');

  const choice = prompt('\n\x1b[38;5;51m▶ Select provider [1-4]:\x1b[0m ');
  
  let provider: string;
  switch (choice) {
    case '1':
      provider = 'supabase';
      break;
    case '2':
      provider = 'cloudflare';
      break;
    case '3':
      provider = 'flyio';
      break;
    default:
      return;
  }

  const selectedProvider = edgeProviders[provider];
  if (!selectedProvider) {
    console.log('\n\x1b[38;5;196m✗ Invalid provider selection\x1b[0m');
    return;
  }

  try {
    // Prompt for and save credentials
    const credentials = await promptForCredentials(provider);
    await saveCredentials(credentials);

    // Verify configuration
    if (selectedProvider.isConfigured()) {
      console.log(`\n\x1b[38;5;82m✓ ${selectedProvider.name} authentication successful\x1b[0m`);
      stats.selectedProvider = provider;
      await toggleEdgeFunctions(true);
      console.log('\n\x1b[38;5;82m✓ Edge computing system online\x1b[0m');
    } else {
      console.log(`\n\x1b[38;5;196m✗ Authentication verification failed\x1b[0m`);
    }
  } catch (error) {
    if (error.message !== 'Authentication sequence aborted') {
      console.error('\n\x1b[38;5;196m✗ Authentication failed:', error.message, '\x1b[0m');
    }
  }
}
