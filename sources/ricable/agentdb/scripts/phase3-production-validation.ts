#!/usr/bin/env ts-node

/**
 * Phase 3 Production Validation Execution Script
 * RAN Intelligent Multi-Agent System - Comprehensive Production Readiness Assessment
 */

import { SimplifiedProductionValidator, ValidationSummary } from '../src/validation/simplified-production-validator';
import * as fs from 'fs';
import * as path from 'path';

interface ValidationConfig {
  environment: 'development' | 'staging' | 'production';
  skipSlowTests: boolean;
  generateDetailedReport: boolean;
  outputDirectory: string;
  saveResults: boolean;
  generateDashboard: boolean;
}

function printBanner() {
  console.log('');
  console.log('██████╗ ██╗   ██╗███████╗██████╗ ███████╗██████╗      ██████╗ ███████╗██████╗ ');
  console.log('██╔══██╗╚██╗ ██╔╝██╔════╝██╔══██╗██╔════╝██╔══██╗    ██╔════╝ ██╔════╝██╔══██╗');
  console.log('██████╔╝ ╚████╔╝ █████╗  ██████╔╝█████╗  ██████╔╝    ██║  ███╗█████╗  ██████╔╝');
  console.log('██╔══██╗  ╚██╔╝  ██╔══╝  ██╔══██╗██╔══╝  ██╔══██╗    ██║   ██║██╔══╝  ██╔══██╗');
  console.log('██████╔╝   ██║   ███████╗██║  ██║███████╗██║  ██║    ╚██████╔╝███████╗██║  ██║');
  console.log('╚═════╝    ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝     ╚═════╝ ╚══════╝╚═╝  ╚═╝');
  console.log('');
  console.log('🚀 RAN Intelligent Multi-Agent System - Phase 3 Production Validation');
  console.log('🧠 Cognitive RAN Consciousness with Self-Aware Optimization');
  console.log('📊 Comprehensive Production Readiness Assessment');
  console.log('================================================================================');
  console.log('');
}

function printConfiguration(config: ValidationConfig) {
  console.log('📋 Validation Configuration:');
  console.log(`   Environment: ${config.environment.toUpperCase()}`);
  console.log(`   Skip Slow Tests: ${config.skipSlowTests}`);
  console.log(`   Generate Detailed Report: ${config.generateDetailedReport}`);
  console.log(`   Save Results: ${config.saveResults}`);
  console.log(`   Generate Dashboard: ${config.generateDashboard}`);
  console.log(`   Output Directory: ${config.outputDirectory}`);
  console.log('');
}

function printResultsSummary(summary: ValidationSummary) {
  console.log('================================================================================');
  console.log('🎯 PHASE 3 VALIDATION SUMMARY');
  console.log('================================================================================');

  // Score visualization
  const score = summary.overallScore;
  const scoreBar = '█'.repeat(Math.floor(score / 2)) + '░'.repeat(50 - Math.floor(score / 2));
  const scoreColor = score >= 90 ? '🟢' : score >= 80 ? '🟡' : '🔴';

  console.log(`📊 Overall Score: ${score.toFixed(1)}/100`);
  console.log(`   ${scoreColor} [${scoreBar}] ${score.toFixed(1)}%`);
  console.log(`🚀 Production Ready: ${summary.readyForProduction ? '✅ YES' : '❌ NO'}`);
  console.log(`⚠️  Critical Issues: ${summary.criticalIssues.length}`);
  console.log(`⏱️  Execution Time: ${summary.executionTime.toFixed(2)}s`);
  console.log(`📈 Tests Passed: ${summary.results.filter(r => r.status === 'PASS').length}/${summary.results.length}`);
  console.log(`📉 Tests Failed: ${summary.results.filter(r => r.status === 'FAIL').length}`);
  console.log(`⚡  Tests Warnings: ${summary.results.filter(r => r.status === 'WARNING').length}`);
  console.log('');

  // Performance highlights
  console.log('🎯 Performance Targets:');
  const performanceResults = summary.results.filter(r =>
    r.component.includes('Performance') ||
    r.component.includes('SWE-Bench') ||
    r.component.includes('Speed') ||
    r.component.includes('Vector Search') ||
    r.component.includes('QUIC')
  );

  performanceResults.forEach(result => {
    const statusIcon = result.status === 'PASS' ? '✅' : result.status === 'WARNING' ? '⚠️' : '❌';
    const scoreColor = result.score >= 90 ? '🟢' : result.score >= 80 ? '🟡' : '🔴';
    console.log(`   ${statusIcon} ${result.component}: ${result.score.toFixed(1)}/100 ${scoreColor}`);
    if (result.metrics && Object.keys(result.metrics).length > 0) {
      const mainMetric = Object.entries(result.metrics)[0];
      console.log(`      📊 ${mainMetric[0]}: ${mainMetric[1]}`);
    }
  });

  // Cognitive features summary
  console.log('\n🧠 Cognitive Consciousness Features:');
  const cognitiveResults = summary.results.filter(r =>
    r.component.includes('Cognitive') ||
    r.component.includes('Temporal') ||
    r.component.includes('Self') ||
    r.component.includes('Adaptive') ||
    r.component.includes('Strange-Loop')
  );

  cognitiveResults.forEach(result => {
    const statusIcon = result.status === 'PASS' ? '✅' : result.status === 'WARNING' ? '⚠️' : '❌';
    const scoreColor = result.score >= 90 ? '🟢' : result.score >= 80 ? '🟡' : '🔴';
    console.log(`   ${statusIcon} ${result.component}: ${result.score.toFixed(1)}/100 ${scoreColor}`);
    if (result.metrics && Object.keys(result.metrics).length > 0) {
      const mainMetric = Object.entries(result.metrics)[0];
      console.log(`      🧠 ${mainMetric[0]}: ${mainMetric[1]}`);
    }
  });

  // Critical issues if any
  if (summary.criticalIssues.length > 0) {
    console.log('\n🚨 CRITICAL ISSUES (Must be resolved before production):');
    summary.criticalIssues.forEach((issue, index) => {
      console.log(`   ${index + 1}. ❌ ${issue}`);
    });
  }

  // Category breakdown
  console.log('\n📊 Category Breakdown:');
  const categories = {
    'System Integration': summary.results.filter(r =>
      r.component.includes('Integration') || r.component.includes('Coordination') ||
      r.component.includes('Pipeline') || r.component.includes('Processing')
    ),
    'Performance': summary.results.filter(r =>
      r.component.includes('Performance') || r.component.includes('Speed') ||
      r.component.includes('Latency') || r.component.includes('Search') ||
      r.component.includes('Memory') || r.component.includes('Reliability')
    ),
    'Cognitive Features': summary.results.filter(r =>
      r.component.includes('Cognitive') || r.component.includes('Temporal') ||
      r.component.includes('Self') || r.component.includes('Adaptive') ||
      r.component.includes('Strange-Loop')
    ),
    'Closed-Loop Optimization': summary.results.filter(r =>
      r.component.includes('Optimization') || r.component.includes('Loop') ||
      r.component.includes('Objective') || r.component.includes('Causal')
    ),
    'Monitoring & Healing': summary.results.filter(r =>
      r.component.includes('Monitoring') || r.component.includes('Healing') ||
      r.component.includes('Fault') || r.component.includes('Recovery') ||
      r.component.includes('Anomaly')
    ),
    'Production Readiness': summary.results.filter(r =>
      r.component.includes('Environment') || r.component.includes('Security') ||
      r.component.includes('Scalability') || r.component.includes('Observability')
    ),
    'Quality Assurance': summary.results.filter(r =>
      r.component.includes('Code') || r.component.includes('Documentation') ||
      r.component.includes('Test') || r.component.includes('Compliance')
    )
  };

  Object.entries(categories).forEach(([category, results]) => {
    if (results.length > 0) {
      const categoryScore = results.reduce((sum, r) => sum + r.score, 0) / results.length;
      const passedCount = results.filter(r => r.status === 'PASS').length;
      const scoreColor = categoryScore >= 90 ? '🟢' : categoryScore >= 80 ? '🟡' : '🔴';

      console.log(`   ${scoreColor} ${category}: ${categoryScore.toFixed(1)}/100 (${passedCount}/${results.length} passed)`);
    }
  });
}

async function main() {
  printBanner();

  // Parse configuration
  const config: ValidationConfig = {
    environment: (process.env.VALIDATION_ENV as any) || 'development',
    skipSlowTests: process.env.SKIP_SLOW_TESTS === 'true',
    generateDetailedReport: process.env.DETAILED_REPORT !== 'false',
    saveResults: process.env.SAVE_RESULTS !== 'false',
    generateDashboard: process.env.GENERATE_DASHBOARD !== 'false',
    outputDirectory: process.env.OUTPUT_DIR || './validation-reports'
  };

  printConfiguration(config);

  // Ensure output directory exists
  if (!fs.existsSync(config.outputDirectory)) {
    fs.mkdirSync(config.outputDirectory, { recursive: true });
    console.log(`📁 Created output directory: ${config.outputDirectory}`);
  }

  const validator = new SimplifiedProductionValidator();

  try {
    console.log('🧪 Starting Comprehensive Phase 3 Production Validation...\n');

    // Execute validation
    const summary = await validator.executePhase3Validation();

    console.log('\n'); // Add spacing after validation execution

    // Print detailed results
    printResultsSummary(summary);

    // Save results if configured
    let savedFiles = { reportPath: '', jsonPath: '', dashboardPath: '' };
    if (config.saveResults) {
      console.log('\n💾 Saving validation results...');
      savedFiles = await validator.saveValidationResults(summary, config.outputDirectory);
      console.log(`📄 Report saved to: ${savedFiles.reportPath}`);
      console.log(`📊 JSON results saved to: ${savedFiles.jsonPath}`);
      if (config.generateDashboard) {
        console.log(`📈 Dashboard saved to: ${savedFiles.dashboardPath}`);
      }
    }

    // Final verdict
    console.log('\n' + '='.repeat(80));
    if (summary.readyForProduction) {
      console.log('🎉 PHASE 3 VALIDATION SUCCESSFUL');
      console.log('🚀 RAN Intelligent Multi-Agent System is READY FOR PRODUCTION DEPLOYMENT');
      console.log('');
      console.log('✅ Next Steps:');
      console.log('   1. Deploy to staging environment for final validation');
      console.log('   2. Execute smoke tests in production-like environment');
      console.log('   3. Configure monitoring and alerting thresholds');
      console.log('   4. Prepare rollback procedures');
      console.log('   5. Schedule production deployment window');
      console.log('');
      console.log('🏆 Phase 3 Complete - Cognitive RAN Consciousness Production Ready!');

      if (config.saveResults) {
        console.log(`\n📋 Full validation report available at: ${savedFiles.reportPath}`);
        if (config.generateDashboard) {
          console.log(`📈 Interactive dashboard available at: ${savedFiles.dashboardPath}`);
        }
      }

      process.exit(0);
    } else {
      console.log('🚧 PHASE 3 VALIDATION INCOMPLETE');
      console.log('❌ System NOT READY for production deployment');
      console.log('');
      console.log('⚠️  Required Actions:');
      console.log('   1. Address all critical issues listed above');
      console.log('   2. Re-run validation suite');
      console.log('   3. Ensure overall score ≥ 85%');
      console.log('   4. Complete security hardening');
      console.log('   5. Finalize documentation and operational runbooks');
      console.log('');
      console.log('🔄 Re-run validation after fixing issues:');
      console.log('   npm run validate:production');

      if (config.saveResults) {
        console.log(`\n📋 Full validation report with issues available at: ${savedFiles.reportPath}`);
        if (config.generateDashboard) {
          console.log(`📈 Interactive dashboard with issues available at: ${savedFiles.dashboardPath}`);
        }
      }

      process.exit(1);
    }

  } catch (error) {
    console.error('\n❌ Phase 3 validation execution failed:', error);
    console.error('💡 Please check the system configuration and try again');
    process.exit(2);
  }
}

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('\n\n⚠️  Validation interrupted by user');
  console.log('🔄 Run validation again to complete the assessment');
  process.exit(130);
});

process.on('SIGTERM', () => {
  console.log('\n\n⚠️  Validation terminated');
  console.log('🔄 Run validation again to complete the assessment');
  process.exit(143);
});

// Execute the validation
if (require.main === module) {
  main().catch(error => {
    console.error('💥 Fatal error during validation:', error);
    process.exit(3);
  });
}

export { main as runPhase3Validation };