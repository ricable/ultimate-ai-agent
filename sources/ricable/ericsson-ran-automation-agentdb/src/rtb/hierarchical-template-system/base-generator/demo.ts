/**
 * Base Template Auto-Generator Demo
 *
 * This demo shows how to use the base template generator system
 * to automatically generate Priority 9 base templates from XML constraints
 * and CSV parameter specifications.
 */

import { BaseTemplateOrchestrator, generateBaseTemplatesQuick } from './index';
import { performanceMonitor } from './performance-monitor';

export async function runBaseTemplateDemo(): Promise<void> {
  console.log('🚀 Starting Base Template Auto-Generator Demo');
  console.log('=' .repeat(60));

  // Start performance monitoring
  performanceMonitor.startMonitoring();

  try {
    // Example 1: Quick generation with default settings
    console.log('\n📖 Example 1: Quick Template Generation');
    console.log('-' .repeat(40));

    performanceMonitor.startStage('quick_generation');

    try {
      const result1 = await generateBaseTemplatesQuick(
        './data/MPnh.xml',           // XML schema file
        './output/templates',         // Output directory
        {
          templateGeneration: {
            optimizationLevel: 'cognitive',
            generateCustomFunctions: true
          },
          validation: {
            enabled: true,
            level: 'standard',
            generateReport: true
          }
        }
      );

      performanceMonitor.endStage('quick_generation');

      console.log(`✅ Generated ${result1.templates.length} templates`);
      console.log(`📊 Processing stats: ${result1.stats.totalParameters} parameters, ${result1.stats.totalMOClasses} MO classes`);
      console.log(`⏱️  Total time: ${(result1.stats.xmlProcessingTime + result1.stats.hierarchyProcessingTime + result1.stats.validationTime).toFixed(2)}s`);

      if (result1.errors.length > 0) {
        console.log(`❌ Errors: ${result1.errors.length}`);
        result1.errors.slice(0, 3).forEach(error => console.log(`   - ${error}`));
      }

      if (result1.warnings.length > 0) {
        console.log(`⚠️  Warnings: ${result1.warnings.length}`);
        result1.warnings.slice(0, 3).forEach(warning => console.log(`   - ${warning}`));
      }

    } catch (error) {
      performanceMonitor.recordError('quick_generation', 'generation_failed', String(error), 'high');
      console.error(`❌ Quick generation failed: ${error}`);
    }

    // Example 2: Advanced configuration with multiple sources
    console.log('\n📖 Example 2: Advanced Template Generation');
    console.log('-' .repeat(40));

    performanceMonitor.startStage('advanced_generation');

    try {
      const orchestrator = new BaseTemplateOrchestrator({
        xmlParsing: {
          streaming: true,
          batchSize: 500,
          memoryLimit: 1024 // 1GB
        },
        csvProcessing: {
          strictMode: false,
          validateConstraints: true,
          mergeDuplicates: true
        },
        templateGeneration: {
          priority: 9,
          includeDefaults: true,
          includeConstraints: true,
          generateCustomFunctions: true,
          optimizationLevel: 'cognitive'
        },
        validation: {
          enabled: true,
          level: 'strict',
          generateReport: true
        }
      });

      const result2 = await orchestrator.generateBaseTemplates({
        xmlPath: './data/MPnh.xml',
        structCsvPath: './data/StructParameters.csv',
        spreadsheetCsvPath: './data/Spreadsheets_Parameters.csv',
        outputPath: './output/advanced-templates'
      });

      performanceMonitor.endStage('advanced_generation');

      console.log(`✅ Generated ${result2.templates.length} advanced templates`);
      console.log(`📊 Processing stats: ${result2.stats.totalParameters} parameters, ${result2.stats.totalMOClasses} MO classes`);
      console.log(`💾 Memory usage: ${result2.stats.memoryUsage.toFixed(2)}MB`);

      // Show template details
      console.log('\n📋 Generated Templates:');
      result2.templates.slice(0, 5).forEach(template => {
        console.log(`   - ${template.templateId}: ${template.parameters.length} parameters (Score: ${template.validationResults.score.toFixed(2)})`);
      });

    } catch (error) {
      performanceMonitor.recordError('advanced_generation', 'generation_failed', String(error), 'high');
      console.error(`❌ Advanced generation failed: ${error}`);
    }

    // Example 3: Performance analysis
    console.log('\n📖 Example 3: Performance Analysis');
    console.log('-' .repeat(40));

    performanceMonitor.startStage('performance_analysis');

    try {
      // Simulate processing large dataset
      const mockResult = await simulateLargeDatasetProcessing();

      performanceMonitor.endStage('performance_analysis');

      console.log(`✅ Processed ${mockResult.processedItems} items`);
      console.log(`📊 Throughput: ${mockResult.throughput.toFixed(2)} items/second`);

    } catch (error) {
      performanceMonitor.recordError('performance_analysis', 'analysis_failed', String(error), 'medium');
      console.error(`❌ Performance analysis failed: ${error}`);
    }

  } catch (error) {
    console.error(`❌ Demo failed: ${error}`);
  } finally {
    // Stop performance monitoring and generate report
    const metrics = performanceMonitor.stopMonitoring();

    console.log('\n📊 Performance Report');
    console.log('=' .repeat(60));
    console.log(performanceMonitor.generateReport());

    // Save detailed report
    const fs = require('fs').promises;
    try {
      await fs.writeFile('./output/performance-report.md', performanceMonitor.generateReport());
      await fs.writeFile('./output/performance-metrics.json', performanceMonitor.generateJSON());
      console.log('\n💾 Performance reports saved to ./output/');
    } catch (error) {
      console.warn(`Failed to save performance reports: ${error}`);
    }
  }
}

/**
 * Simulate processing of large dataset for performance analysis
 */
async function simulateLargeDatasetProcessing(): Promise<{
  processedItems: number;
  throughput: number;
}> {
  const startTime = Date.now();
  let processedItems = 0;

  // Simulate processing 10,000 items
  const totalItems = 10000;
  const batchSize = 100;

  for (let i = 0; i < totalItems; i += batchSize) {
    // Simulate batch processing
    await new Promise(resolve => setTimeout(resolve, 1));

    processedItems += Math.min(batchSize, totalItems - i);

    // Simulate occasional errors
    if (Math.random() < 0.001) {
      performanceMonitor.recordError('batch_processing', 'simulation_error', `Simulated error at item ${i}`, 'low');
    }

    // Simulate occasional warnings
    if (Math.random() < 0.005) {
      performanceMonitor.recordWarning('batch_processing', `Processing slow at item ${i}`, 'Consider optimizing batch size');
    }
  }

  const duration = Date.now() - startTime;
  const throughput = processedItems / (duration / 1000);

  // Record custom metrics
  performanceMonitor.recordCustomMetric('total_simulation_items', totalItems);
  performanceMonitor.recordCustomMetric('batch_size', batchSize);
  performanceMonitor.recordCustomMetric('simulation_duration_ms', duration);

  return { processedItems, throughput };
}

/**
 * Generate sample data for testing
 */
export async function generateSampleData(): Promise<void> {
  console.log('📝 Generating sample data for testing...');

  const fs = require('fs').promises;
  const path = require('path');

  // Create output directory
  await fs.mkdir('./data', { recursive: true });

  // Generate sample MPnh.xml content (simplified)
  const sampleMPnhXML = `<?xml version="1.0" encoding="UTF-8"?>
<schema>
  <moc name="ManagedElement">
    <parameter name="userLabel" type="String" description="User defined label"/>
    <parameter name="vendorName" type="String" description="Vendor name"/>
  </moc>

  <moc name="ENodeBFunction">
    <parameter name="eNodeBPlmnId" type="Integer32" minValue="1" maxValue="999" description="PLMN ID"/>
    <parameter name="eNodeBId" type="Integer32" minValue="1" maxValue="268435455" description="eNodeB identifier"/>
  </moc>

  <moc name="EUtranCellFDD">
    <parameter name="qRxLevMin" type="Integer32" minValue="-140" maxValue="-44" defaultValue="-64" description="Minimum required RX level"/>
    <parameter name="qQualMin" type="Integer32" minValue="-34" maxValue="-3" defaultValue="-8" description="Minimum required quality"/>
    <parameter name="cellIndividualOffset" type="Integer32" minValue="-30" maxValue="30" defaultValue="0" description="Cell individual offset"/>
  </moc>
</schema>`;

  await fs.writeFile('./data/MPnh.xml', sampleMPnhXML);
  console.log('✅ Generated sample MPnh.xml');

  // Generate sample StructParameters.csv
  const sampleStructCSV = `Parameter Name,Parameter Type,MO Class,Parent Structure,Structure Level,Description
userLabel,String,ManagedElement,attributes,1,User defined label
vendorName,String,ManagedElement,attributes,1,Vendor name
eNodeBPlmnId,Integer32,ENodeBFunction,attributes,1,PLMN ID
eNodeBId,Integer32,ENodeBFunction,attributes,1,eNodeB identifier
qRxLevMin,Integer32,EUtranCellFDD,attributes,1,Minimum required RX level
qQualMin,Integer32,EUtranCellFDD,attributes,1,Minimum required quality
cellIndividualOffset,Integer32,EUtranCellFDD,attributes,1,Cell individual offset`;

  await fs.writeFile('./data/StructParameters.csv', sampleStructCSV);
  console.log('✅ Generated sample StructParameters.csv');

  // Generate sample Spreadsheets_Parameters.csv
  const sampleSpreadsheetCSV = `Name,MO Class,Parameter Type,VS Data Type,Description,Default Value,Min Value,Max Value,Category,Feature
userLabel,ManagedElement,string,String,User defined label,,UserDefined,basic
vendorName,ManagedElement,string,String,Vendor name,,UserDefined,basic
eNodeBPlmnId,ENodeBFunction,int,Integer32,PLMN ID,1,1,999,identification,4G
eNodeBId,ENodeBFunction,int,Integer32,eNodeB identifier,1,1,268435455,identification,4G
qRxLevMin,EUtranCellFDD,int,Integer32,Minimum required RX level,-64,-140,-44,coverage,4G
qQualMin,EUtranCellFDD,int,Integer32,Minimum required quality,-8,-34,-3,coverage,4G
cellIndividualOffset,EUtranCellFDD,int,Integer32,Cell individual offset,0,-30,30,optimization,4G`;

  await fs.writeFile('./data/Spreadsheets_Parameters.csv', sampleSpreadsheetCSV);
  console.log('✅ Generated sample Spreadsheets_Parameters.csv');

  // Generate sample momt_tree.txt
  const sampleMomtTree = `ManagedElement
  ENodeBFunction
    EUtranCellFDD
    EUtranCellTDD
  EPCFunction
    MobilityManagementEntity
  PolicyManagementFunction`;

  await fs.writeFile('./data/momt_tree.txt', sampleMomtTree);
  console.log('✅ Generated sample momt_tree.txt');

  // Generate sample momtl_LDN.txt
  const sampleLDN = `ManagedElement
ManagedElement.ENodeBFunction
ManagedElement.ENodeBFunction.EUtranCellFDD
ManagedElement.ENodeBFunction.EUtranCellTDD
ManagedElement.EPCFunction
ManagedElement.EPCFunction.MobilityManagementEntity
ManagedElement.PolicyManagementFunction`;

  await fs.writeFile('./data/momtl_LDN.txt', sampleLDN);
  console.log('✅ Generated sample momtl_LDN.txt');

  // Generate sample reservedby.txt
  const sampleReservedBy = `ENodeBFunction -> ManagedElement [contains] [1-1] {eNodeB requires managed element}
EUtranCellFDD -> ENodeBFunction [contains] [1-*] {cell requires eNodeB function}
MobilityManagementEntity -> EPCFunction [contains] [1-1] {MME requires EPC function}
EUtranCellTDD -> ENodeBFunction [contains] [1-*] {TDD cell requires eNodeB function}`;

  await fs.writeFile('./data/reservedby.txt', sampleReservedBy);
  console.log('✅ Generated sample reservedby.txt');

  // Create output directory
  await fs.mkdir('./output', { recursive: true });
  await fs.mkdir('./output/templates', { recursive: true });
  await fs.mkdir('./output/advanced-templates', { recursive: true });

  console.log('✅ Sample data generation complete!');
}

/**
 * Run the demo if this file is executed directly
 */
if (require.main === module) {
  generateSampleData()
    .then(() => runBaseTemplateDemo())
    .catch(error => {
      console.error('Demo failed:', error);
      process.exit(1);
    });
}