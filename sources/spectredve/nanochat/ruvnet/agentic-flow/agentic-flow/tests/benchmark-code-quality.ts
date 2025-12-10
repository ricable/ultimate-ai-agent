#!/usr/bin/env npx tsx
// Code Quality Benchmark for Alternative LLM Models

import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';

interface BenchmarkResult {
  model: string;
  task: string;
  code: string;
  compiles: boolean;
  hasTests: boolean;
  hasDocumentation: boolean;
  errorHandling: boolean;
  codeQuality: number; // 1-10
  latency: number;
  cost: number;
}

const CODING_TASKS = [
  {
    name: 'binary_search',
    prompt: 'Write a Python function that implements binary search with type hints, docstrings, and error handling for edge cases'
  },
  {
    name: 'rest_endpoint',
    prompt: 'Create a FastAPI endpoint that accepts JSON, validates input, handles errors, and returns proper HTTP status codes'
  },
  {
    name: 'async_processing',
    prompt: 'Write a Python async function that fetches data from multiple URLs concurrently using asyncio, with timeout and error handling'
  },
  {
    name: 'data_processing',
    prompt: 'Create a Python function that reads a CSV file, processes data with pandas, handles missing values, and exports to JSON with error handling'
  }
];

const TEST_MODELS = [
  { id: 'meta-llama/llama-3.1-8b-instruct', name: 'Llama 3.1 8B' },
  { id: 'deepseek/deepseek-chat-v3.1', name: 'DeepSeek V3.1' },
  { id: 'google/gemini-2.5-flash-preview-09-2025', name: 'Gemini 2.5 Flash' },
  { id: 'anthropic/claude-3-5-sonnet', name: 'Claude 3.5 Sonnet (baseline)' }
];

async function testModel(modelId: string, modelName: string, task: any): Promise<BenchmarkResult> {
  const startTime = Date.now();

  try {
    console.log(`   Testing ${modelName}...`);

    const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.OPENROUTER_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: modelId,
        messages: [{
          role: 'user',
          content: task.prompt
        }],
        max_tokens: 2000
      })
    });

    const latency = Date.now() - startTime;

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    const code = data.choices[0]?.message?.content || '';

    // Analyze code quality
    const analysis = analyzeCode(code);
    const cost = ((data.usage?.prompt_tokens || 0) * 0.00001) + ((data.usage?.completion_tokens || 0) * 0.00003);

    return {
      model: modelName,
      task: task.name,
      code,
      ...analysis,
      latency,
      cost
    };
  } catch (error: any) {
    console.log(`   ❌ Error: ${error.message}`);
    return {
      model: modelName,
      task: task.name,
      code: '',
      compiles: false,
      hasTests: false,
      hasDocumentation: false,
      errorHandling: false,
      codeQuality: 0,
      latency: Date.now() - startTime,
      cost: 0
    };
  }
}

function analyzeCode(code: string): {
  compiles: boolean;
  hasTests: boolean;
  hasDocumentation: boolean;
  errorHandling: boolean;
  codeQuality: number;
} {
  // Check for Python syntax (basic check)
  const hasDef = code.includes('def ') || code.includes('async def ');
  const hasImports = code.includes('import ') || code.includes('from ');

  // Check for documentation
  const hasDocstring = code.includes('"""') || code.includes("'''");
  const hasTypeHints = code.includes(': ') && code.includes('->');
  const hasComments = code.includes('#');

  // Check for error handling
  const hasTryExcept = code.includes('try:') && code.includes('except');
  const hasRaises = code.includes('raise ');
  const hasAssert = code.includes('assert ');

  // Check for tests
  const hasTests = code.includes('def test_') || code.includes('import pytest') || code.includes('import unittest');

  // Calculate quality score
  let quality = 0;
  if (hasDef) quality += 2;
  if (hasDocstring) quality += 2;
  if (hasTypeHints) quality += 2;
  if (hasTryExcept || hasRaises) quality += 2;
  if (hasImports) quality += 1;
  if (hasComments) quality += 1;

  return {
    compiles: hasDef && hasImports,
    hasTests,
    hasDocumentation: hasDocstring || hasComments,
    errorHandling: hasTryExcept || hasRaises || hasAssert,
    codeQuality: Math.min(quality, 10)
  };
}

async function runBenchmark() {
  console.log('╔═══════════════════════════════════════════════════╗');
  console.log('║  Code Quality Benchmark - Alternative LLM Models ║');
  console.log('╚═══════════════════════════════════════════════════╝\n');

  const results: BenchmarkResult[] = [];
  const outputDir = '/tmp/code-benchmark';

  // Create output directory
  try {
    mkdirSync(outputDir, { recursive: true });
  } catch (e) {}

  for (const task of CODING_TASKS) {
    console.log(`\n📝 Task: ${task.name}`);
    console.log('─'.repeat(60));

    for (const model of TEST_MODELS) {
      const result = await testModel(model.id, model.name, task);
      results.push(result);

      // Save generated code
      if (result.code) {
        const filename = join(outputDir, `${task.name}_${model.name.replace(/\s+/g, '_')}.py`);
        writeFileSync(filename, result.code);
      }

      // Print result
      const quality = '★'.repeat(Math.floor(result.codeQuality)) + '☆'.repeat(10 - Math.floor(result.codeQuality));
      console.log(`   ${result.compiles ? '✅' : '❌'} ${model.name}: ${result.latency}ms | Quality: ${quality} (${result.codeQuality}/10)`);
    }
  }

  // Generate summary report
  console.log('\n' + '='.repeat(80));
  console.log('📊 BENCHMARK SUMMARY');
  console.log('='.repeat(80) + '\n');

  // Group results by model
  const byModel = results.reduce((acc, r) => {
    if (!acc[r.model]) acc[r.model] = [];
    acc[r.model].push(r);
    return acc;
  }, {} as Record<string, BenchmarkResult[]>);

  for (const [model, modelResults] of Object.entries(byModel)) {
    const avgQuality = modelResults.reduce((sum, r) => sum + r.codeQuality, 0) / modelResults.length;
    const avgLatency = modelResults.reduce((sum, r) => sum + r.latency, 0) / modelResults.length;
    const totalCost = modelResults.reduce((sum, r) => sum + r.cost, 0);
    const successRate = modelResults.filter(r => r.compiles).length / modelResults.length * 100;
    const hasDocsRate = modelResults.filter(r => r.hasDocumentation).length / modelResults.length * 100;
    const errorHandlingRate = modelResults.filter(r => r.errorHandling).length / modelResults.length * 100;

    console.log(`\n${model}`);
    console.log('─'.repeat(60));
    console.log(`  Success Rate:      ${successRate.toFixed(0)}%`);
    console.log(`  Avg Quality Score: ${avgQuality.toFixed(1)}/10`);
    console.log(`  Documentation:     ${hasDocsRate.toFixed(0)}%`);
    console.log(`  Error Handling:    ${errorHandlingRate.toFixed(0)}%`);
    console.log(`  Avg Latency:       ${avgLatency.toFixed(0)}ms`);
    console.log(`  Total Cost:        $${totalCost.toFixed(6)}`);
  }

  // Ranking
  console.log('\n' + '='.repeat(80));
  console.log('🏆 RANKINGS');
  console.log('='.repeat(80) + '\n');

  const rankings = Object.entries(byModel).map(([model, modelResults]) => ({
    model,
    avgQuality: modelResults.reduce((sum, r) => sum + r.codeQuality, 0) / modelResults.length,
    avgLatency: modelResults.reduce((sum, r) => sum + r.latency, 0) / modelResults.length,
    totalCost: modelResults.reduce((sum, r) => sum + r.cost, 0),
    successRate: modelResults.filter(r => r.compiles).length / modelResults.length * 100
  })).sort((a, b) => b.avgQuality - a.avgQuality);

  rankings.forEach((r, i) => {
    const medal = i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : '  ';
    console.log(`${medal} #${i + 1}: ${r.model}`);
    console.log(`     Quality: ${r.avgQuality.toFixed(1)}/10 | Latency: ${r.avgLatency.toFixed(0)}ms | Cost: $${r.totalCost.toFixed(6)}\n`);
  });

  console.log('='.repeat(80));
  console.log('💡 RECOMMENDATIONS');
  console.log('='.repeat(80) + '\n');

  const best = rankings[0];
  const fastest = rankings.sort((a, b) => a.avgLatency - b.avgLatency)[0];
  const cheapest = rankings.sort((a, b) => a.totalCost - b.totalCost)[0];

  console.log(`Best Quality:  ${best.model} (${best.avgQuality.toFixed(1)}/10)`);
  console.log(`Fastest:       ${fastest.model} (${fastest.avgLatency.toFixed(0)}ms avg)`);
  console.log(`Most Affordable: ${cheapest.model} ($${cheapest.totalCost.toFixed(6)} total)`);

  console.log(`\n✅ Generated code samples saved to: ${outputDir}/\n`);
  console.log('✨ Benchmark complete!\n');
}

runBenchmark().catch(console.error);
