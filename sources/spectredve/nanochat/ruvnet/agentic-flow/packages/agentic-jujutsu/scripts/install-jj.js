#!/usr/bin/env node
/**
 * Automatic jj installation script
 * Tries multiple methods to ensure jj is available
 */

const { execSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
};

function log(message, color = 'cyan') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function checkJJInstalled() {
  try {
    execSync('jj --version', { stdio: 'pipe' });
    return true;
  } catch (error) {
    return false;
  }
}

function checkCargoInstalled() {
  try {
    execSync('cargo --version', { stdio: 'pipe' });
    return true;
  } catch (error) {
    return false;
  }
}

function installWithCargo() {
  log('\n📦 Installing jj via Cargo...', 'cyan');
  log('This may take 5-10 minutes on first install.\n', 'yellow');

  try {
    // Use spawn for real-time output
    const install = spawn('cargo', [
      'install',
      '--git',
      'https://github.com/martinvonz/jj',
      'jj-cli',
      '--locked'
    ], {
      stdio: 'inherit'
    });

    return new Promise((resolve, reject) => {
      install.on('close', (code) => {
        if (code === 0) {
          log('\n✅ jj installed successfully via Cargo!', 'green');
          resolve(true);
        } else {
          reject(new Error(`Cargo install failed with code ${code}`));
        }
      });
    });
  } catch (error) {
    throw new Error(`Cargo installation failed: ${error.message}`);
  }
}

async function downloadPrebuiltBinary() {
  log('\n📥 Attempting to download prebuilt jj binary...', 'cyan');

  const platform = os.platform();
  const arch = os.arch();

  // Map to jj release platforms
  const platformMap = {
    'darwin': 'macos',
    'linux': 'linux',
    'win32': 'windows'
  };

  const archMap = {
    'x64': 'x86_64',
    'arm64': 'aarch64'
  };

  const jjPlatform = platformMap[platform];
  const jjArch = archMap[arch];

  if (!jjPlatform || !jjArch) {
    throw new Error(`Unsupported platform: ${platform} ${arch}`);
  }

  log(`Platform: ${jjPlatform}-${jjArch}`, 'blue');

  // For now, show instructions for manual installation
  // TODO: Implement actual binary download from GitHub releases
  throw new Error('Prebuilt binary download not yet implemented');
}

function showManualInstructions() {
  log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', 'yellow');
  log('⚠️  Manual Installation Required', 'yellow');
  log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n', 'yellow');

  log('Please install jj using one of these methods:\n', 'bright');

  log('1️⃣  Cargo (Recommended - works on all platforms):', 'cyan');
  log('   cargo install --git https://github.com/martinvonz/jj jj-cli\n');

  log('2️⃣  Homebrew (macOS/Linux):', 'cyan');
  log('   brew install jj\n');

  log('3️⃣  Nix (Linux/macOS):', 'cyan');
  log('   nix-env -iA nixpkgs.jujutsu\n');

  log('4️⃣  From source:', 'cyan');
  log('   git clone https://github.com/martinvonz/jj');
  log('   cd jj && cargo build --release\n');

  log('📖 More info: https://github.com/martinvonz/jj#installation\n', 'blue');

  log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n', 'yellow');
}

async function main() {
  log('\n🔍 Checking for jj installation...', 'cyan');

  if (checkJJInstalled()) {
    const version = execSync('jj --version', { encoding: 'utf8' }).trim();
    log(`✅ jj is already installed: ${version}`, 'green');
    log('✅ agentic-jujutsu is ready to use!\n', 'green');
    return;
  }

  log('❌ jj not found on system', 'yellow');

  // Skip automatic installation in CI environments
  if (process.env.CI === 'true' || process.env.SKIP_JJ_INSTALL === 'true') {
    log('⏭️  Skipping automatic installation (CI environment)', 'yellow');
    showManualInstructions();
    process.exit(0);
  }

  // Skip if user opted out
  if (process.env.AGENTIC_JUJUTSU_SKIP_INSTALL === 'true') {
    log('⏭️  Skipping installation (AGENTIC_JUJUTSU_SKIP_INSTALL=true)', 'yellow');
    showManualInstructions();
    process.exit(0);
  }

  log('\n💡 agentic-jujutsu can attempt to install jj automatically.', 'cyan');

  if (checkCargoInstalled()) {
    log('✅ Cargo detected - can install via Cargo\n', 'green');

    // Check if we should auto-install
    const autoInstall = process.env.AGENTIC_JUJUTSU_AUTO_INSTALL === 'true';

    if (autoInstall) {
      try {
        await installWithCargo();
        log('✅ Installation complete!\n', 'green');
        return;
      } catch (error) {
        log(`❌ Auto-installation failed: ${error.message}`, 'red');
        showManualInstructions();
        process.exit(1);
      }
    } else {
      log('To enable automatic installation, set:', 'yellow');
      log('  export AGENTIC_JUJUTSU_AUTO_INSTALL=true\n', 'yellow');
      showManualInstructions();
    }
  } else {
    log('❌ Cargo not found - cannot auto-install', 'yellow');
    log('\nTo enable auto-installation, first install Rust:', 'yellow');
    log('  curl --proto \'=https\' --tlsv1.2 -sSf https://sh.rustup.rs | sh\n', 'yellow');
    showManualInstructions();
  }

  // Exit with 0 to not break npm install
  // User can still use the package, they just need to install jj manually
  process.exit(0);
}

main().catch(error => {
  log(`\n❌ Error: ${error.message}`, 'red');
  showManualInstructions();
  process.exit(0); // Don't fail npm install
});
