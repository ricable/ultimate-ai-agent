#!/usr/bin/env ts-node

/**
 * Comprehensive Production Validation Script
 * Executes full Phase 3 production readiness validation
 */

import { ProductionValidationSystem } from '../src/validation/production-validation-system';
import * as fs from 'fs';
import * as path from 'path';

interface ValidationConfig {
  environment: 'development' | 'staging' | 'production';
  skipSlowTests: boolean;
  generateDetailedReport: boolean;
  outputDirectory: string;
}

async function main() {
  console.log('🚀 RAN Intelligent Multi-Agent System - Phase 3 Production Validation');
  console.log('=======================================================================');

  const config: ValidationConfig = {
    environment: (process.env.VALIDATION_ENV as any) || 'development',
    skipSlowTests: process.env.SKIP_SLOW_TESTS === 'true',
    generateDetailedReport: process.env.DETAILED_REPORT !== 'false',
    outputDirectory: process.env.OUTPUT_DIR || './validation-reports'
  };

  console.log(`📋 Configuration:`);
  console.log(`   Environment: ${config.environment}`);
  console.log(`   Skip Slow Tests: ${config.skipSlowTests}`);
  console.log(`   Generate Detailed Report: ${config.generateDetailedReport}`);
  console.log(`   Output Directory: ${config.outputDirectory}`);

  // Ensure output directory exists
  if (!fs.existsSync(config.outputDirectory)) {
    fs.mkdirSync(config.outputDirectory, { recursive: true });
  }

  const validationSystem = new ProductionValidationSystem();

  try {
    console.log('\n🧪 Starting Comprehensive Production Validation...\n');

    // Execute full validation
    const validationResult = await validationSystem.executeFullValidation();

    // Generate and save report
    const report = validationSystem.generateValidationReport();

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const reportFileName = `phase3-production-validation-${timestamp}.md`;
    const reportPath = path.join(config.outputDirectory, reportFileName);

    fs.writeFileSync(reportPath, report);
    console.log(`\n📄 Validation report saved to: ${reportPath}`);

    // Save JSON results for programmatic access
    const jsonResults = {
      timestamp: new Date().toISOString(),
      configuration: config,
      results: validationResult,
      summary: {
        overallScore: validationResult.overallScore,
        readyForProduction: validationResult.readyForProduction,
        criticalIssuesCount: validationResult.criticalIssues.length,
        totalTests: validationResult.results.length,
        passedTests: validationResult.results.filter(r => r.status === 'PASS').length,
        failedTests: validationResult.results.filter(r => r.status === 'FAIL').length,
        warningTests: validationResult.results.filter(r => r.status === 'WARNING').length
      }
    };

    const jsonFileName = `phase3-validation-results-${timestamp}.json`;
    const jsonPath = path.join(config.outputDirectory, jsonFileName);
    fs.writeFileSync(jsonPath, JSON.stringify(jsonResults, null, 2));
    console.log(`📊 JSON results saved to: ${jsonPath}`);

    // Generate summary dashboard HTML
    const dashboardHtml = generateDashboardHtml(validationResult, jsonResults);
    const dashboardFileName = `validation-dashboard-${timestamp}.html`;
    const dashboardPath = path.join(config.outputDirectory, dashboardFileName);
    fs.writeFileSync(dashboardPath, dashboardHtml);
    console.log(`📈 Dashboard saved to: ${dashboardPath}`);

    // Final summary
    console.log('\n' + '='.repeat(70));
    console.log('🎯 VALIDATION SUMMARY');
    console.log('='.repeat(70));
    console.log(`📊 Overall Score: ${validationResult.overallScore.toFixed(1)}/100`);
    console.log(`🚀 Production Ready: ${validationResult.readyForProduction ? '✅ YES' : '❌ NO'}`);
    console.log(`⚠️  Critical Issues: ${validationResult.criticalIssues.length}`);
    console.log(`📈 Tests Passed: ${jsonResults.summary.passedTests}/${jsonResults.summary.totalTests}`);
    console.log(`📉 Tests Failed: ${jsonResults.summary.failedTests}/${jsonResults.summary.totalTests}`);
    console.log(`⚡  Tests Warnings: ${jsonResults.summary.warningTests}/${jsonResults.summary.totalTests}`);

    if (validationResult.criticalIssues.length > 0) {
      console.log('\n🚨 CRITICAL ISSUES (Must be resolved before production):');
      validationResult.criticalIssues.forEach((issue, index) => {
        console.log(`   ${index + 1}. ${issue}`);
      });
    }

    // Performance targets summary
    console.log('\n🎯 PERFORMANCE TARGETS:');
    const performanceResults = validationResult.results.filter(r =>
      r.component.includes('Performance') ||
      r.component.includes('SWE-Bench') ||
      r.component.includes('Speed') ||
      r.component.includes('Vector Search') ||
      r.component.includes('QUIC')
    );

    performanceResults.forEach(result => {
      const statusIcon = result.status === 'PASS' ? '✅' : result.status === 'WARNING' ? '⚠️' : '❌';
      console.log(`   ${statusIcon} ${result.component}: ${result.score.toFixed(1)}/100`);
      if (result.metrics && Object.keys(result.metrics).length > 0) {
        const mainMetric = Object.entries(result.metrics)[0];
        console.log(`      ${mainMetric[0]}: ${mainMetric[1]}`);
      }
    });

    console.log('\n🧠 COGNITIVE FEATURES:');
    const cognitiveResults = validationResult.results.filter(r =>
      r.component.includes('Cognitive') ||
      r.component.includes('Temporal') ||
      r.component.includes('Self') ||
      r.component.includes('Adaptive')
    );

    cognitiveResults.forEach(result => {
      const statusIcon = result.status === 'PASS' ? '✅' : result.status === 'WARNING' ? '⚠️' : '❌';
      console.log(`   ${statusIcon} ${result.component}: ${result.score.toFixed(1)}/100`);
    });

    // Exit with appropriate code
    if (validationResult.readyForProduction) {
      console.log('\n🎉 SYSTEM READY FOR PRODUCTION DEPLOYMENT!');
      process.exit(0);
    } else {
      console.log('\n🚧 SYSTEM NOT READY - Address critical issues before deployment');
      process.exit(1);
    }

  } catch (error) {
    console.error('\n❌ Validation execution failed:', error);
    process.exit(2);
  }
}

function generateDashboardHtml(validationResult: any, jsonResults: any): string {
  return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAN Intelligent Multi-Agent System - Production Validation Dashboard</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #fafafa;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .score-excellent { color: #28a745; }
        .score-good { color: #ffc107; }
        .score-poor { color: #dc3545; }
        .status-ready { background: #28a745; }
        .status-not-ready { background: #dc3545; }
        .content {
            padding: 30px;
        }
        .section {
            margin-bottom: 40px;
        }
        .section h2 {
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .test-results {
            display: grid;
            gap: 15px;
        }
        .test-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #ddd;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .test-item.pass { border-left-color: #28a745; }
        .test-item.warning { border-left-color: #ffc107; }
        .test-item.fail { border-left-color: #dc3545; }
        .test-name {
            font-weight: 500;
        }
        .test-score {
            font-weight: bold;
            font-size: 1.1em;
        }
        .critical-issues {
            background: #fff5f5;
            border: 1px solid #fed7d7;
            border-radius: 6px;
            padding: 20px;
            margin-top: 20px;
        }
        .critical-issues h3 {
            color: #c53030;
            margin-top: 0;
        }
        .issue-list {
            list-style: none;
            padding: 0;
        }
        .issue-list li {
            padding: 8px 0;
            border-bottom: 1px solid #fed7d7;
        }
        .issue-list li:last-child {
            border-bottom: none;
        }
        .timestamp {
            text-align: center;
            color: #666;
            font-size: 0.9em;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997);
            transition: width 0.3s ease;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>RAN Intelligent Multi-Agent System</h1>
            <p>Phase 3 Production Validation Dashboard</p>
        </div>

        <div class="summary">
            <div class="metric-card">
                <div class="metric-label">Overall Score</div>
                <div class="metric-value ${validationResult.overallScore >= 90 ? 'score-excellent' : validationResult.overallScore >= 80 ? 'score-good' : 'score-poor'}">
                    ${validationResult.overallScore.toFixed(1)}%
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Production Status</div>
                <div class="metric-value ${validationResult.readyForProduction ? 'status-ready' : 'status-not-ready'}" style="color: white;">
                    ${validationResult.readyForProduction ? 'READY' : 'NOT READY'}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Tests Passed</div>
                <div class="metric-value">
                    ${jsonResults.summary.passedTests}/${jsonResults.summary.totalTests}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Critical Issues</div>
                <div class="metric-value ${validationResult.criticalIssues.length === 0 ? 'score-excellent' : 'score-poor'}">
                    ${validationResult.criticalIssues.length}
                </div>
            </div>
        </div>

        <div class="content">
            <div class="section">
                <h2>Overall Progress</h2>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${validationResult.overallScore}%"></div>
                </div>
                <p style="text-align: center; margin: 10px 0;">
                    ${validationResult.overallScore.toFixed(1)}% Complete - ${validationResult.readyForProduction ? 'Ready for Production' : 'Additional Work Required'}
                </p>
            </div>

            <div class="section">
                <h2>Validation Results</h2>
                <div class="test-results">
                    ${validationResult.results.map((result: any) => `
                        <div class="test-item ${result.status.toLowerCase()}">
                            <div class="test-name">${result.component}</div>
                            <div class="test-score">${result.score.toFixed(1)}/100</div>
                        </div>
                    `).join('')}
                </div>
            </div>

            ${validationResult.criticalIssues.length > 0 ? `
                <div class="section">
                    <div class="critical-issues">
                        <h3>🚨 Critical Issues (Must Be Resolved)</h3>
                        <ul class="issue-list">
                            ${validationResult.criticalIssues.map((issue: string) => `
                                <li>❌ ${issue}</li>
                            `).join('')}
                        </ul>
                    </div>
                </div>
            ` : ''}

            <div class="section">
                <h2>Performance Targets</h2>
                <div class="test-results">
                    ${validationResult.results.filter((r: any) =>
                        r.component.includes('Performance') ||
                        r.component.includes('SWE-Bench') ||
                        r.component.includes('Speed') ||
                        r.component.includes('Vector Search') ||
                        r.component.includes('QUIC')
                    ).map((result: any) => `
                        <div class="test-item ${result.status.toLowerCase()}">
                            <div class="test-name">${result.component}</div>
                            <div class="test-score">${result.score.toFixed(1)}/100</div>
                        </div>
                    `).join('')}
                </div>
            </div>

            <div class="section">
                <h2>Cognitive Features</h2>
                <div class="test-results">
                    ${validationResult.results.filter((r: any) =>
                        r.component.includes('Cognitive') ||
                        r.component.includes('Temporal') ||
                        r.component.includes('Self') ||
                        r.component.includes('Adaptive')
                    ).map((result: any) => `
                        <div class="test-item ${result.status.toLowerCase()}">
                            <div class="test-name">${result.component}</div>
                            <div class="test-score">${result.score.toFixed(1)}/100</div>
                        </div>
                    `).join('')}
                </div>
            </div>
        </div>

        <div class="timestamp">
            Validation completed: ${new Date().toLocaleString()}<br>
            Environment: ${jsonResults.configuration.environment}<br>
            Report generated by RAN Intelligent Multi-Agent System
        </div>
    </div>
</body>
</html>
  `;
}

// Execute the validation
if (require.main === module) {
  main().catch(error => {
    console.error('Fatal error:', error);
    process.exit(3);
  });
}