#!/usr/bin/env node

/**
 * Nova Medicina - Diagnosis Verification Example
 * Demonstrates medical information verification
 *
 * @author ruv (github.com/ruvnet, ruv.io)
 */

import { Verifier } from '../src/index.js';

async function main() {
  console.log('🏥 Nova Medicina - Diagnosis Verification Example\n');

  // Create verifier instance
  const verifier = new Verifier({
    confidenceThreshold: 0.95,
    sources: ['pubmed', 'cochrane', 'uptodate']
  });

  // Example diagnosis verification
  const result = await verifier.verify({
    diagnosis: 'type 2 diabetes',
    evidence: {
      fastingGlucose: 140,
      hba1c: 7.2
    }
  });

  console.log('Verification Results:');
  console.log('Diagnosis:', result.diagnosis);
  console.log('Verified:', result.verified);
  console.log('Confidence:', result.confidence);
  console.log('Sources:', result.sources);
  console.log('Citations:', result.citations);

  if (result.contradictions.length > 0) {
    console.log('\n⚠️  Contradictions found:');
    result.contradictions.forEach(c => console.log('  -', c));
  }
}

main().catch(console.error);
