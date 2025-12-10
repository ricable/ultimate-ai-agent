/**
 * Base Template Auto-Generator
 *
 * This module provides automatic generation of Priority 9 base templates
 * from XML constraints and CSV parameter specifications.
 *
 * Main components:
 * - StreamingXMLParser: Processes 100MB+ MPnh.xml files efficiently
 * - CSVParameterProcessor: Handles StructParameters.csv and Spreadsheets_Parameters.csv
 * - BaseTemplateGenerator: Creates RTB templates with priority-based inheritance
 * - TemplateConstraintValidator: Validates templates against XML constraints
 *
 * Usage:
 * ```typescript
 * import { BaseTemplateOrchestrator } from './index';
 *
 * const orchestrator = new BaseTemplateOrchestrator();
 * const templates = await orchestrator.generateBaseTemplates({
 *   xmlPath: './MPnh.xml',
 *   structCsvPath: './StructParameters.csv',
 *   spreadsheetCsvPath: './Spreadsheets_Parameters.csv',
 *   optimizationLevel: 'cognitive'
 * });
 * ```
 */

export { StreamingXMLParser } from './xml-parser';
export { CSVParameterProcessor } from './csv-processor';
export { BaseTemplateGenerator } from './template-generator';
export { TemplateConstraintValidator } from './constraint-validator';

export type {
  XMLParseResult,
  XMLSchemaElement,
  XMLParsingContext
} from './xml-parser';

export type {
  CSVProcessingResult,
  StructParameter,
  SpreadsheetParameter,
  CSVProcessingConfig
} from './csv-processor';

export type {
  BaseTemplateConfig,
  GeneratedTemplate,
  TemplateGenerationMetadata,
  TemplateValidationResult,
  TemplateInheritanceChain,
  ConflictResolution
} from './template-generator';

export type {
  ValidationResult,
  ValidationError,
  ValidationWarning,
  ConstraintValidationRule,
  TemplateConstraintProfile,
  XMLConstraintSchema,
  ParameterTypeDefinition,
  GlobalConstraint,
  DependencyRule,
  ValidationProfile
} from './constraint-validator';

import { StreamingXMLParser } from './xml-parser';
import { CSVParameterProcessor } from './csv-processor';
import { BaseTemplateGenerator, BaseTemplateConfig } from './template-generator';
import { TemplateConstraintValidator } from './constraint-validator';
import { GeneratedTemplate } from './template-generator';
import { ProcessingStats } from '../../types/rtb-types';

export interface BaseTemplateOrchestratorConfig {
  xmlParsing?: {
    streaming: boolean;
    batchSize?: number;
    memoryLimit?: number;
  };
  csvProcessing?: {
    strictMode: boolean;
    skipInvalidRows: boolean;
    validateConstraints: boolean;
    mergeDuplicates: boolean;
  };
  templateGeneration?: {
    priority: number;
    includeDefaults: boolean;
    includeConstraints: boolean;
    generateCustomFunctions: boolean;
    optimizationLevel: 'basic' | 'enhanced' | 'cognitive';
  };
  validation?: {
    enabled: boolean;
    level: 'strict' | 'standard' | 'lenient';
    generateReport: boolean;
  };
}

export interface GenerationRequest {
  xmlPath: string;
  structCsvPath?: string;
  spreadsheetCsvPath?: string;
  outputPath?: string;
  config?: BaseTemplateOrchestratorConfig;
}

export interface GenerationResult {
  templates: GeneratedTemplate[];
  stats: ProcessingStats;
  validationResult?: Map<string, any>;
  report?: string;
  errors: string[];
  warnings: string[];
}

export class BaseTemplateOrchestrator {
  private xmlParser: StreamingXMLParser;
  private csvProcessor: CSVParameterProcessor;
  private templateGenerator: BaseTemplateGenerator;
  private constraintValidator: TemplateConstraintValidator;
  private config: BaseTemplateOrchestratorConfig;

  constructor(config: BaseTemplateOrchestratorConfig = {}) {
    this.config = {
      xmlParsing: {
        streaming: true,
        batchSize: 1000,
        memoryLimit: 2048,
        ...config.xmlParsing
      },
      csvProcessing: {
        strictMode: false,
        skipInvalidRows: true,
        validateConstraints: true,
        mergeDuplicates: true,
        ...config.csvProcessing
      },
      templateGeneration: {
        priority: 9,
        includeDefaults: true,
        includeConstraints: true,
        generateCustomFunctions: true,
        optimizationLevel: 'cognitive',
        ...config.templateGeneration
      },
      validation: {
        enabled: true,
        level: 'standard',
        generateReport: true,
        ...config.validation
      }
    };

    this.xmlParser = new StreamingXMLParser(this.config.xmlParsing);
    this.csvProcessor = new CSVParameterProcessor(this.config.csvProcessing);
    this.templateGenerator = new BaseTemplateGenerator();
    this.constraintValidator = new TemplateConstraintValidator();
  }

  /**
   * Generate base templates from XML and CSV sources
   */
  async generateBaseTemplates(request: GenerationRequest): Promise<GenerationResult> {
    console.log('🚀 Starting Base Template Generation Process');
    console.log(`📄 XML Source: ${request.xmlPath}`);
    console.log(`📊 CSV Sources: ${request.structCsvPath || 'N/A'}, ${request.spreadsheetCsvPath || 'N/A'}`);
    console.log(`⚙️  Configuration: ${JSON.stringify(this.config, null, 2)}`);

    const startTime = Date.now();
    const errors: string[] = [];
    const warnings: string[] = [];

    try {
      // Step 1: Parse XML schema
      console.log('\n📖 Step 1: Parsing XML Schema...');
      const xmlResult = await this.parseXMLSchema(request.xmlPath, errors, warnings);

      // Step 2: Process CSV data
      console.log('\n📊 Step 2: Processing CSV Data...');
      const csvResult = await this.processCSVData(request, errors, warnings);

      // Step 3: Generate templates
      console.log('\n🏗️  Step 3: Generating Templates...');
      const templates = await this.generateTemplates(xmlResult, csvResult, errors, warnings);

      // Step 4: Validate templates
      let validationResult;
      if (this.config.validation?.enabled) {
        console.log('\n✅ Step 4: Validating Templates...');
        validationResult = await this.validateTemplates(templates, errors, warnings);
      }

      // Step 5: Generate report
      let report;
      if (this.config.validation?.generateReport && validationResult) {
        console.log('\n📋 Step 5: Generating Report...');
        report = this.constraintValidator.generateValidationReport(validationResult);
      }

      // Step 6: Save templates if output path provided
      if (request.outputPath) {
        console.log('\n💾 Step 6: Saving Templates...');
        await this.saveTemplates(templates, request.outputPath, report);
      }

      const processingTime = (Date.now() - startTime) / 1000;
      const totalParameters = templates.reduce((sum, t) => sum + t.parameters.length, 0);
      const totalMOClasses = new Set(templates.flatMap(t => t.parameters.map(p => p.hierarchy[0]))).size;

      const stats: ProcessingStats = {
        xmlProcessingTime: xmlResult.stats.xmlProcessingTime,
        hierarchyProcessingTime: csvResult.stats.hierarchyProcessingTime,
        validationTime: validationResult ?
          Array.from(validationResult.values()).reduce((sum, r) => sum + r.processingTime, 0) : 0,
        totalParameters,
        totalMOClasses,
        totalRelationships: 0, // TODO: Calculate from MO relationships
        memoryUsage: process.memoryUsage().heapUsed / 1024 / 1024,
        errorCount: errors.length,
        warningCount: warnings.length
      };

      console.log('\n🎉 Base Template Generation Complete!');
      console.log(`⏱️  Total Time: ${processingTime.toFixed(2)}s`);
      console.log(`📈 Stats: ${totalParameters} parameters, ${totalMOClasses} MO classes`);
      console.log(`📝 Templates Generated: ${templates.length}`);
      console.log(`❌ Errors: ${errors.length}, ⚠️  Warnings: ${warnings.length}`);

      return {
        templates,
        stats,
        validationResult,
        report,
        errors,
        warnings
      };

    } catch (error) {
      const errorMsg = `Base template generation failed: ${error}`;
      errors.push(errorMsg);
      console.error(`❌ ${errorMsg}`);
      throw new Error(errorMsg);
    }
  }

  /**
   * Parse XML schema from file
   */
  private async parseXMLSchema(
    xmlPath: string,
    errors: string[],
    warnings: string[]
  ): Promise<any> {
    try {
      const fs = require('fs').promises;

      // Check if file exists
      try {
        await fs.access(xmlPath);
      } catch (error) {
        throw new Error(`XML file not found: ${xmlPath}`);
      }

      const xmlResult = await this.xmlParser.parseMPnhXML(xmlPath);

      // Add any errors from XML parsing
      errors.push(...xmlResult.errors);
      warnings.push(...xmlResult.warnings);

      console.log(`✅ XML parsing complete: ${xmlResult.parameters.length} parameters, ${xmlResult.moClasses.size} MO classes`);

      return xmlResult;

    } catch (error) {
      const errorMsg = `Failed to parse XML schema: ${error}`;
      errors.push(errorMsg);
      throw new Error(errorMsg);
    }
  }

  /**
   * Process CSV data from files
   */
  private async processCSVData(
    request: GenerationRequest,
    errors: string[],
    warnings: string[]
  ): Promise<any> {
    try {
      const fs = require('fs').promises;
      let structResult, spreadsheetResult;

      // Process StructParameters.csv if provided
      if (request.structCsvPath) {
        try {
          await fs.access(request.structCsvPath);
          structResult = await this.csvProcessor.processStructParameters(request.structCsvPath);
          console.log(`✅ StructParameters.csv processed: ${structResult.parameters.length} parameters`);
        } catch (error) {
          const warning = `StructParameters.csv not found or invalid: ${error}`;
          warnings.push(warning);
          console.warn(`⚠️  ${warning}`);
          structResult = { parameters: [], parameterHierarchy: new Map(), parameterGroups: new Map(), stats: { totalParameters: 0, totalMOClasses: 0, hierarchyProcessingTime: 0, validationTime: 0, xmlProcessingTime: 0, totalRelationships: 0, memoryUsage: 0, errorCount: 0, warningCount: 0 }, errors: [], warnings: [] };
        }
      } else {
        console.log('ℹ️  StructParameters.csv not provided, skipping...');
        structResult = { parameters: [], parameterHierarchy: new Map(), parameterGroups: new Map(), stats: { totalParameters: 0, totalMOClasses: 0, hierarchyProcessingTime: 0, validationTime: 0, xmlProcessingTime: 0, totalRelationships: 0, memoryUsage: 0, errorCount: 0, warningCount: 0 }, errors: [], warnings: [] };
      }

      // Process Spreadsheets_Parameters.csv if provided
      if (request.spreadsheetCsvPath) {
        try {
          await fs.access(request.spreadsheetCsvPath);
          spreadsheetResult = await this.csvProcessor.processSpreadsheetParameters(request.spreadsheetCsvPath);
          console.log(`✅ Spreadsheets_Parameters.csv processed: ${spreadsheetResult.parameters.length} parameters`);
        } catch (error) {
          const warning = `Spreadsheets_Parameters.csv not found or invalid: ${error}`;
          warnings.push(warning);
          console.warn(`⚠️  ${warning}`);
          spreadsheetResult = { parameters: [], parameterHierarchy: new Map(), parameterGroups: new Map(), stats: { totalParameters: 0, totalMOClasses: 0, hierarchyProcessingTime: 0, validationTime: 0, xmlProcessingTime: 0, totalRelationships: 0, memoryUsage: 0, errorCount: 0, warningCount: 0 }, errors: [], warnings: [] };
        }
      } else {
        console.log('ℹ️  Spreadsheets_Parameters.csv not provided, skipping...');
        spreadsheetResult = { parameters: [], parameterHierarchy: new Map(), parameterGroups: new Map(), stats: { totalParameters: 0, totalMOClasses: 0, hierarchyProcessingTime: 0, validationTime: 0, xmlProcessingTime: 0, totalRelationships: 0, memoryUsage: 0, errorCount: 0, warningCount: 0 }, errors: [], warnings: [] };
      }

      // Merge CSV results
      const mergedResult = await this.csvProcessor.mergeParameters(structResult, spreadsheetResult);

      // Add any errors from CSV processing
      errors.push(...structResult.errors, ...spreadsheetResult.errors);
      warnings.push(...structResult.warnings, ...spreadsheetResult.warnings);

      console.log(`✅ CSV processing complete: ${mergedResult.parameters.length} merged parameters`);

      return mergedResult;

    } catch (error) {
      const errorMsg = `Failed to process CSV data: ${error}`;
      errors.push(errorMsg);
      throw new Error(errorMsg);
    }
  }

  /**
   * Generate templates from parsed data
   */
  private async generateTemplates(
    xmlResult: any,
    csvResult: any,
    errors: string[],
    warnings: string[]
  ): Promise<GeneratedTemplate[]> {
    try {
      const config: BaseTemplateConfig = {
        priority: this.config.templateGeneration!.priority!,
        templateType: 'base',
        includeDefaults: this.config.templateGeneration!.includeDefaults!,
        includeConstraints: this.config.templateGeneration!.includeConstraints!,
        generateCustomFunctions: this.config.templateGeneration!.generateCustomFunctions!,
        optimizationLevel: this.config.templateGeneration!.optimizationLevel!
      };

      const templates = await this.templateGenerator.generateBaseTemplates(xmlResult, csvResult, config);

      console.log(`✅ Template generation complete: ${templates.length} templates generated`);

      // Log template summary
      for (const template of templates) {
        const isValid = template.validationResults.isValid;
        const status = isValid ? '✅' : '❌';
        console.log(`   ${status} ${template.templateId} (${template.parameters.length} parameters, score: ${template.validationResults.score.toFixed(2)})`);

        // Add validation errors if any
        if (!isValid) {
          for (const error of template.validationResults.errors) {
            errors.push(`Template ${template.templateId}: ${error.message}`);
          }
        }

        // Add validation warnings if any
        for (const warning of template.validationResults.warnings) {
          warnings.push(`Template ${template.templateId}: ${warning.message}`);
        }
      }

      return templates;

    } catch (error) {
      const errorMsg = `Failed to generate templates: ${error}`;
      errors.push(errorMsg);
      throw new Error(errorMsg);
    }
  }

  /**
   * Validate generated templates
   */
  private async validateTemplates(
    templates: GeneratedTemplate[],
    errors: string[],
    warnings: string[]
  ): Promise<Map<string, any>> {
    try {
      const validationResult = await this.constraintValidator.validateTemplates(templates);

      let validCount = 0;
      for (const [templateId, result] of validationResult) {
        if (result.isValid) {
          validCount++;
        } else {
          for (const error of result.errors) {
            errors.push(`Validation [${templateId}]: ${error.message}`);
          }
        }

        for (const warning of result.warnings) {
          warnings.push(`Validation [${templateId}]: ${warning.message}`);
        }
      }

      console.log(`✅ Template validation complete: ${validCount}/${templates.length} templates valid`);

      return validationResult;

    } catch (error) {
      const errorMsg = `Failed to validate templates: ${error}`;
      errors.push(errorMsg);
      throw new Error(errorMsg);
    }
  }

  /**
   * Save templates to output directory
   */
  private async saveTemplates(
    templates: GeneratedTemplate[],
    outputPath: string,
    report?: string
  ): Promise<void> {
    try {
      const fs = require('fs').promises;
      const path = require('path');

      // Ensure output directory exists
      await fs.mkdir(outputPath, { recursive: true });

      // Save each template as individual JSON file
      for (const template of templates) {
        const templateFileName = `${template.templateId}.json`;
        const templateFilePath = path.join(outputPath, templateFileName);

        const templateData = {
          template: template.template,
          metadata: template.metadata,
          validation: template.validationResults
        };

        await fs.writeFile(templateFilePath, JSON.stringify(templateData, null, 2));
        console.log(`💾 Saved: ${templateFileName}`);
      }

      // Save combined template file
      const combinedData = {
        generatedAt: new Date().toISOString(),
        templates: templates.map(t => ({
          templateId: t.templateId,
          template: t.template,
          metadata: t.metadata,
          validation: t.validationResults
        }))
      };

      const combinedFilePath = path.join(outputPath, 'base-templates-combined.json');
      await fs.writeFile(combinedFilePath, JSON.stringify(combinedData, null, 2));
      console.log(`💾 Saved: base-templates-combined.json`);

      // Save validation report if available
      if (report) {
        const reportFilePath = path.join(outputPath, 'validation-report.md');
        await fs.writeFile(reportFilePath, report);
        console.log(`💾 Saved: validation-report.md`);
      }

      console.log(`✅ All templates saved to: ${outputPath}`);

    } catch (error) {
      throw new Error(`Failed to save templates: ${error}`);
    }
  }

  /**
   * Get processing statistics
   */
  getProcessingStats(): {
    memoryUsage: number;
    uptime: number;
    config: BaseTemplateOrchestratorConfig;
  } {
    return {
      memoryUsage: process.memoryUsage().heapUsed / 1024 / 1024,
      uptime: process.uptime(),
      config: this.config
    };
  }

  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<BaseTemplateOrchestratorConfig>): void {
    this.config = {
      ...this.config,
      ...newConfig,
      xmlParsing: { ...this.config.xmlParsing, ...newConfig.xmlParsing },
      csvProcessing: { ...this.config.csvProcessing, ...newConfig.csvProcessing },
      templateGeneration: { ...this.config.templateGeneration, ...newConfig.templateGeneration },
      validation: { ...this.config.validation, ...newConfig.validation }
    };

    // Re-initialize components with new config
    this.xmlParser = new StreamingXMLParser(this.config.xmlParsing);
    this.csvProcessor = new CSVParameterProcessor(this.config.csvProcessing);
  }
}

/**
 * Quick start function for basic template generation
 */
export async function generateBaseTemplatesQuick(
  xmlPath: string,
  outputPath?: string,
  options?: Partial<BaseTemplateOrchestratorConfig>
): Promise<GenerationResult> {
  const orchestrator = new BaseTemplateOrchestrator(options);

  return await orchestrator.generateBaseTemplates({
    xmlPath,
    outputPath,
    config: options
  });
}

/**
 * Export default orchestrator for easy usage
 */
export default BaseTemplateOrchestrator;