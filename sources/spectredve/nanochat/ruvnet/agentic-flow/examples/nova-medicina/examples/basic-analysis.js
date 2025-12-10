#!/usr/bin/env node

/**
 * Nova Medicina - Basic Analysis Example
 * Demonstrates basic symptom analysis
 *
 * @author ruv (github.com/ruvnet, ruv.io)
 */

import { Analyzer } from '../src/index.js';

async function main() {
  console.log('🏥 Nova Medicina - Basic Analysis Example\n');

  // Create analyzer instance
  const analyzer = new Analyzer({
    minConfidenceScore: 0.95,
    verificationLevel: 'moderate'
  });

  // Example symptom analysis
  const result = await analyzer.analyze({
    symptoms: 'fever, cough, fatigue',
    duration: '3 days',
    age: 35,
    gender: 'female'
  });

  console.log('Analysis Results:');
  console.log('Symptoms:', result.symptoms);
  console.log('Confidence:', result.confidence);
  console.log('Urgency:', result.urgency);
  console.log('Recommendations:', result.recommendations);
  console.log('\n⚠️ ', result.disclaimer);
}

main().catch(console.error);
