import { serverInfo, stats } from "./types.ts";

const ASCII_LOGO = `
╔════════════════════════════════════════════════════════════╗
║    ___    ____   ______ ____ ___  _______________          ║
║   /   |  /  _/  / ____// __// _ \\/ ___/ __/ __/            ║
║  / /| |  / /   / /_   / _/ / // / /__/ _// _/              ║
║ / ___ |_/ /   / __/  / ___/ // / ___/ __/ __/              ║
║/_/  |_/___/  /_/    /_/  /___/_/  /_/ /_/                  ║
║                                                            ║
║                 AI FEDERATION NETWORK                      ║
║              Distributed Runtime System                    ║
╚════════════════════════════════════════════════════════════╝
`;

const PROCESS_SPINNER = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█', '▇', '▆', '▅', '▄', '▃', '▂'];
let spinnerInterval: number | undefined;

function startProcessMonitor(message: string) {
  let i = 0;
  console.log(''); // Add spacing
  spinnerInterval = setInterval(() => {
    console.log(`\r${PROCESS_SPINNER[i]} ${message}`);
    i = (i + 1) % PROCESS_SPINNER.length;
  }, 80);
}

function stopProcessMonitor() {
  if (spinnerInterval) {
    clearInterval(spinnerInterval);
    console.log('\r'); // Clear spinner line
  }
}

export function displayServerStatus() {
  console.clear(); // Clear the terminal for fresh display
  
  // Display Federation Logo
  console.log('\x1b[36m%s\x1b[0m', ASCII_LOGO);
  
  // Runtime Status
  console.log('\n\x1b[38;5;51m▀▀▀ RUNTIME STATUS ▀▀▀\x1b[0m');
  console.log(`⚡ Instance: \x1b[38;5;87m${serverInfo.name} [v${serverInfo.version}]\x1b[0m`);
  console.log(`🔌 Active Connections: \x1b[38;5;147m${stats.connections}\x1b[0m`);
  
  // Edge Computing Status
  console.log('\n\x1b[38;5;51m▀▀▀ EDGE COMPUTING STATUS ▀▀▀\x1b[0m');
  if (stats.edgeFunctionsEnabled) {
    console.log('⚛️  Service Status: \x1b[38;5;82mONLINE\x1b[0m');
    if (stats.selectedProvider) {
      console.log(`🌐 Cloud Provider: \x1b[38;5;117m${stats.selectedProvider}\x1b[0m`);
    }
    if (stats.activeEdgeFunctions.length > 0) {
      console.log('📡 Active Functions:');
      stats.activeEdgeFunctions.forEach(func => {
        console.log(`   ∟ \x1b[38;5;82m${func}\x1b[0m`);
        if (stats.deployedUrls[func]) {
          console.log(`     ⮡ \x1b[38;5;45m${stats.deployedUrls[func]}\x1b[0m`);
        }
      });
    }
  } else {
    console.log('💠 Service Status: \x1b[38;5;209mOFFLINE\x1b[0m');
  }

  // System Capabilities
  console.log('\n\x1b[38;5;51m▀▀▀ SYSTEM CAPABILITIES ▀▀▀\x1b[0m');
  if (serverInfo.capabilities.models?.length) {
    console.log('🤖 ML Models:', serverInfo.capabilities.models.map(m => `\x1b[38;5;117m${m}\x1b[0m`).join(', '));
  }
  if (serverInfo.capabilities.protocols?.length) {
    console.log('🌐 Network Protocols:', serverInfo.capabilities.protocols.map(p => `\x1b[38;5;117m${p}\x1b[0m`).join(', '));
  }
  if (serverInfo.capabilities.features?.length) {
    console.log('⚙️  Runtime Features:', serverInfo.capabilities.features.map(f => `\x1b[38;5;117m${f}\x1b[0m`).join(', '));
  }

  // Command Interface
  console.log('\n\x1b[38;5;51m▀▀▀ SYSTEM CONTROLS ▀▀▀\x1b[0m');
  console.log('\x1b[38;5;239m┌───────────────────────────────────────┐\x1b[0m');
  console.log('\x1b[38;5;239m│\x1b[0m 1️⃣  Monitor Network Connections        \x1b[38;5;239m│\x1b[0m');
  console.log('\x1b[38;5;239m│\x1b[0m 2️⃣  View System Information            \x1b[38;5;239m│\x1b[0m');
  console.log('\x1b[38;5;239m│\x1b[0m 3️⃣  List Runtime Capabilities          \x1b[38;5;239m│\x1b[0m');
  console.log('\x1b[38;5;239m│\x1b[0m 4️⃣  Configure Cloud Provider           \x1b[38;5;239m│\x1b[0m');
  console.log('\x1b[38;5;239m│\x1b[0m 5️⃣  Deploy Edge Function               \x1b[38;5;239m│\x1b[0m');
  console.log('\x1b[38;5;239m│\x1b[0m 6️⃣  Check Edge Function Status         \x1b[38;5;239m│\x1b[0m');
  console.log('\x1b[38;5;239m│\x1b[0m 7️⃣  View System Logs                   \x1b[38;5;239m│\x1b[0m');
  console.log('\x1b[38;5;239m│\x1b[0m 8️⃣  List Deployed Functions            \x1b[38;5;239m│\x1b[0m');
  console.log('\x1b[38;5;239m│\x1b[0m 9️⃣  Show Command Interface             \x1b[38;5;239m│\x1b[0m');
  console.log('\x1b[38;5;239m└───────────────────────────────────────┘\x1b[0m');
  
  console.log('\n\x1b[38;5;51m▶ Enter command [1-9]:\x1b[0m');
}
