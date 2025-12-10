#!/usr/bin/env npx tsx
/**
 * RAG Query CLI
 *
 * Usage: npx tsx src/cli/query.ts "Your question about Ericsson RAN"
 *
 * Interactive query interface for the self-learning RAG system.
 */

import * as readline from 'readline';
import { RAGPipeline } from '../rag/rag-pipeline.js';
import { logger } from '../utils/logger.js';

async function main() {
  const args = process.argv.slice(2);

  console.log(`
╔══════════════════════════════════════════════════════════════════╗
║           RuVector RAG Query Interface                           ║
╠══════════════════════════════════════════════════════════════════╣
║  Self-Learning RAG for Ericsson RAN Technical Documentation      ║
╚══════════════════════════════════════════════════════════════════╝
`);

  // Initialize RAG pipeline
  console.log('Initializing RAG pipeline...');
  const ragPipeline = new RAGPipeline();
  await ragPipeline.initialize();

  const stats = ragPipeline.getStats();
  console.log(`Loaded ${stats.totalChunks} document chunks`);
  console.log('');

  // Single query mode
  if (args.length > 0) {
    const query = args.join(' ');
    await executeQuery(ragPipeline, query);
    return;
  }

  // Interactive mode
  console.log('Enter your questions (type "exit" to quit, "help" for commands)');
  console.log('━'.repeat(66));

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const prompt = () => {
    rl.question('\n> ', async (input) => {
      const trimmed = input.trim();

      if (!trimmed) {
        prompt();
        return;
      }

      if (trimmed.toLowerCase() === 'exit' || trimmed.toLowerCase() === 'quit') {
        console.log('\nGoodbye!');
        rl.close();
        return;
      }

      if (trimmed.toLowerCase() === 'help') {
        printHelp();
        prompt();
        return;
      }

      if (trimmed.toLowerCase() === 'stats') {
        const s = ragPipeline.getStats();
        console.log('\nVector Store Statistics:');
        console.log(`  Total Chunks: ${s.totalChunks}`);
        console.log(`  MOMs Loaded: ${s.momCount}`);
        console.log('  By Type:');
        for (const [type, count] of Object.entries(s.byDocumentType)) {
          console.log(`    ${type}: ${count}`);
        }
        prompt();
        return;
      }

      if (trimmed.toLowerCase().startsWith('param ')) {
        const paramName = trimmed.substring(6).trim();
        await queryParameter(ragPipeline, paramName);
        prompt();
        return;
      }

      await executeQuery(ragPipeline, trimmed);
      prompt();
    });
  };

  prompt();
}

async function executeQuery(ragPipeline: RAGPipeline, query: string): Promise<void> {
  console.log('\nSearching documentation...');

  const startTime = Date.now();
  const result = await ragPipeline.query(query, {
    topK: 5,
    minSimilarity: 0.4,
    includeMetadata: true,
  });
  const duration = Date.now() - startTime;

  console.log('');
  console.log('─'.repeat(66));
  console.log('');

  if (result.answer) {
    console.log(result.answer);
  } else {
    console.log('No answer could be generated.');
  }

  console.log('');
  console.log('─'.repeat(66));
  console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`);
  console.log(`Sources: ${result.sources.length}`);
  console.log(`Processing time: ${duration}ms`);

  if (result.sources.length > 0) {
    console.log('\nReferences:');
    result.sources.slice(0, 3).forEach((s, i) => {
      console.log(`  ${i + 1}. ${s}`);
    });
  }
}

async function queryParameter(ragPipeline: RAGPipeline, paramName: string): Promise<void> {
  console.log(`\nLooking up parameter: ${paramName}`);

  const result = await ragPipeline.query(
    `What is the ${paramName} parameter in Ericsson RAN? Include its type, valid range, default value, and 3GPP specification reference.`,
    {
      topK: 5,
      parameterNames: [paramName],
      minSimilarity: 0.3,
    }
  );

  console.log('');
  console.log('─'.repeat(66));
  console.log('');

  if (result.answer) {
    console.log(result.answer);
  } else {
    console.log(`Parameter "${paramName}" not found in documentation.`);
  }

  console.log('');
  console.log('─'.repeat(66));
  console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`);
}

function printHelp(): void {
  console.log(`
Commands:
  <query>        Ask any question about Ericsson RAN documentation
  param <name>   Look up a specific 3GPP parameter (e.g., "param pZeroNominalPusch")
  stats          Show vector store statistics
  help           Show this help message
  exit           Exit the query interface

Example Queries:
  What is the optimal alpha value for urban deployments?
  How does pZeroNominalPusch affect uplink SINR?
  Explain the Tuning Paradox in RAN optimization
  What are the 3GPP specifications for fractional power control?
  How to configure LTE uplink power control?
`);
}

main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
