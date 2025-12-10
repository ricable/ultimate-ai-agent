#!/usr/bin/env node

/**
 * QuDAG Native CLI
 * Command-line interface for quantum-resistant cryptography operations
 */

const qudag = require('../index.js');
const { program } = require('commander');
const fs = require('fs');
const path = require('path');

program
  .name('qudag-native')
  .description('QuDAG Native - Quantum-resistant cryptography CLI')
  .version(qudag.version());

// Info command
program
  .command('info')
  .description('Display module information')
  .action(() => {
    const info = qudag.getModuleInfo();
    console.log('\n📦 QuDAG Native Module');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log(`Name:        ${info.name}`);
    console.log(`Version:     ${info.version}`);
    console.log(`Description: ${info.description}`);
    console.log('\n✨ Features:');
    info.features.forEach(f => console.log(`   • ${f}`));
    console.log();
  });

// ML-KEM-768 commands
const mlkem = program.command('mlkem768')
  .description('ML-KEM-768 quantum-resistant encryption operations');

mlkem
  .command('keygen')
  .description('Generate ML-KEM-768 keypair')
  .option('-o, --output <dir>', 'Output directory for keys', '.')
  .option('-n, --name <name>', 'Key file name prefix', 'mlkem768')
  .action((options) => {
    try {
      const keypair = qudag.mlkem768GenerateKeypair();

      const pubKeyPath = path.join(options.output, `${options.name}.pub`);
      const secKeyPath = path.join(options.output, `${options.name}.key`);

      fs.writeFileSync(pubKeyPath, keypair.publicKey);
      fs.writeFileSync(secKeyPath, keypair.secretKey);

      console.log('\n✅ ML-KEM-768 Keypair Generated');
      console.log(`   Public key:  ${pubKeyPath} (1184 bytes)`);
      console.log(`   Secret key:  ${secKeyPath} (2400 bytes)`);
      console.log(`   Fingerprint: ${qudag.quantumFingerprint(keypair.publicKey)}\n`);
    } catch (err) {
      console.error('❌ Error:', err.message);
      process.exit(1);
    }
  });

mlkem
  .command('encrypt')
  .description('Encapsulate shared secret using public key')
  .requiredOption('-p, --public-key <file>', 'Path to public key file')
  .option('-o, --output <dir>', 'Output directory', '.')
  .action((options) => {
    try {
      const publicKey = fs.readFileSync(options.publicKey);
      const result = qudag.mlkem768Encapsulate(publicKey);

      const ctPath = path.join(options.output, 'ciphertext.bin');
      const ssPath = path.join(options.output, 'shared_secret.bin');

      fs.writeFileSync(ctPath, result.ciphertext);
      fs.writeFileSync(ssPath, result.sharedSecret);

      console.log('\n✅ Encryption Complete');
      console.log(`   Ciphertext:    ${ctPath} (1088 bytes)`);
      console.log(`   Shared secret: ${ssPath} (32 bytes)`);
      console.log(`   Secret hex:    ${qudag.bytesToHex(result.sharedSecret)}\n`);
    } catch (err) {
      console.error('❌ Error:', err.message);
      process.exit(1);
    }
  });

mlkem
  .command('decrypt')
  .description('Decapsulate shared secret using secret key')
  .requiredOption('-c, --ciphertext <file>', 'Path to ciphertext file')
  .requiredOption('-s, --secret-key <file>', 'Path to secret key file')
  .action((options) => {
    try {
      const ciphertext = fs.readFileSync(options.ciphertext);
      const secretKey = fs.readFileSync(options.secretKey);

      const sharedSecret = qudag.mlkem768Decapsulate(ciphertext, secretKey);

      console.log('\n✅ Decryption Complete');
      console.log(`   Shared secret: ${qudag.bytesToHex(sharedSecret)}\n`);
    } catch (err) {
      console.error('❌ Error:', err.message);
      process.exit(1);
    }
  });

// ML-DSA-65 commands
const mldsa = program.command('mldsa65')
  .description('ML-DSA-65 quantum-resistant digital signature operations');

mldsa
  .command('keygen')
  .description('Generate ML-DSA-65 keypair')
  .option('-o, --output <dir>', 'Output directory for keys', '.')
  .option('-n, --name <name>', 'Key file name prefix', 'mldsa65')
  .action((options) => {
    try {
      const keypair = qudag.mldsa65GenerateKeypair();

      const pubKeyPath = path.join(options.output, `${options.name}.pub`);
      const secKeyPath = path.join(options.output, `${options.name}.key`);

      fs.writeFileSync(pubKeyPath, keypair.publicKey);
      fs.writeFileSync(secKeyPath, keypair.secretKey);

      console.log('\n✅ ML-DSA-65 Keypair Generated');
      console.log(`   Public key:  ${pubKeyPath} (1952 bytes)`);
      console.log(`   Secret key:  ${secKeyPath} (4032 bytes)`);
      console.log(`   Fingerprint: ${qudag.quantumFingerprint(keypair.publicKey)}\n`);
    } catch (err) {
      console.error('❌ Error:', err.message);
      process.exit(1);
    }
  });

mldsa
  .command('sign')
  .description('Sign a message with ML-DSA-65')
  .requiredOption('-m, --message <file>', 'Path to message file')
  .requiredOption('-s, --secret-key <file>', 'Path to secret key file')
  .option('-o, --output <file>', 'Output signature file', 'signature.bin')
  .action((options) => {
    try {
      const message = fs.readFileSync(options.message);
      const secretKey = fs.readFileSync(options.secretKey);

      const signature = qudag.mldsa65Sign(message, secretKey);
      fs.writeFileSync(options.output, signature);

      console.log('\n✅ Message Signed');
      console.log(`   Signature: ${options.output} (3309 bytes)`);
      console.log(`   Message:   ${options.message} (${message.length} bytes)\n`);
    } catch (err) {
      console.error('❌ Error:', err.message);
      process.exit(1);
    }
  });

mldsa
  .command('verify')
  .description('Verify a signature with ML-DSA-65')
  .requiredOption('-m, --message <file>', 'Path to message file')
  .requiredOption('-g, --signature <file>', 'Path to signature file')
  .requiredOption('-p, --public-key <file>', 'Path to public key file')
  .action((options) => {
    try {
      const message = fs.readFileSync(options.message);
      const signature = fs.readFileSync(options.signature);
      const publicKey = fs.readFileSync(options.publicKey);

      const valid = qudag.mldsa65Verify(message, signature, publicKey);

      if (valid) {
        console.log('\n✅ Signature VALID\n');
        process.exit(0);
      } else {
        console.log('\n❌ Signature INVALID\n');
        process.exit(1);
      }
    } catch (err) {
      console.error('❌ Error:', err.message);
      process.exit(1);
    }
  });

// BLAKE3 commands
const blake3 = program.command('blake3')
  .description('BLAKE3 cryptographic hash operations');

blake3
  .command('hash')
  .description('Compute BLAKE3 hash of a file')
  .requiredOption('-f, --file <file>', 'Path to file')
  .option('--hex', 'Output as hex string')
  .action((options) => {
    try {
      const data = fs.readFileSync(options.file);

      if (options.hex) {
        const hash = qudag.blake3HashHex(data);
        console.log(`\n${hash}\n`);
      } else {
        const hash = qudag.blake3Hash(data);
        console.log(`\n${qudag.bytesToHex(hash)}\n`);
      }
    } catch (err) {
      console.error('❌ Error:', err.message);
      process.exit(1);
    }
  });

blake3
  .command('fingerprint')
  .description('Generate quantum-resistant fingerprint')
  .requiredOption('-f, --file <file>', 'Path to file')
  .action((options) => {
    try {
      const data = fs.readFileSync(options.file);
      const fingerprint = qudag.quantumFingerprint(data);
      console.log(`\n${fingerprint}\n`);
    } catch (err) {
      console.error('❌ Error:', err.message);
      process.exit(1);
    }
  });

// MCP Server command
program
  .command('mcp')
  .description('Start MCP server for quantum cryptography operations')
  .option('-t, --transport <type>', 'Transport type: stdio or sse', 'stdio')
  .option('-p, --port <port>', 'Port for SSE transport', '3000')
  .action((options) => {
    console.log(`\n🚀 Starting QuDAG MCP Server (${options.transport})\n`);

    if (options.transport === 'stdio') {
      require('../mcp-server/stdio.js')();
    } else if (options.transport === 'sse') {
      require('../mcp-server/sse.js')(parseInt(options.port));
    } else {
      console.error('❌ Unknown transport type:', options.transport);
      console.error('   Use: stdio or sse');
      process.exit(1);
    }
  });

// Utility commands
program
  .command('random')
  .description('Generate cryptographically secure random bytes')
  .requiredOption('-n, --bytes <number>', 'Number of bytes to generate')
  .option('-o, --output <file>', 'Output file (default: stdout as hex)')
  .action((options) => {
    try {
      const bytes = parseInt(options.bytes);
      const randomData = qudag.randomBytes(bytes);

      if (options.output) {
        fs.writeFileSync(options.output, randomData);
        console.log(`\n✅ Generated ${bytes} random bytes → ${options.output}\n`);
      } else {
        console.log(`\n${qudag.bytesToHex(randomData)}\n`);
      }
    } catch (err) {
      console.error('❌ Error:', err.message);
      process.exit(1);
    }
  });

program.parse();
