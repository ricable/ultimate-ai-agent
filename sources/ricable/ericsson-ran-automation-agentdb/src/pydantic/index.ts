/**
 * Phase 5 Implementation - Pydantic Schema Generation Module
 *
 * Main entry point for XML to Pydantic model generation with comprehensive
 * type mapping, validation, and cognitive learning integration
 */

export { TypeMapper } from './type-mapper';
export { SchemaEngine } from './schema-engine';
export { ValidationFramework } from './validation-framework';
export { XmlToPydanticGenerator } from './xml-to-pydantic-generator';

export type {
  TypeMappingConfig,
  TypeMapping,
  MappingResult,
  MappingStatistics
} from './type-mapper';

export type {
  SchemaGenerationConfig,
  ModelGenerationOptions,
  GeneratedClass,
  GeneratedField,
  GeneratedMethod,
  GeneratedValidator,
  SchemaGenerationMetrics
} from './schema-engine';

export type {
  ValidationConfig,
  ValidationResult,
  ValidationError,
  ValidationWarning,
  ValidationDetail,
  ValidationPerformanceMetrics,
  ValidationRule,
  CrossParameterRule,
  ConditionalValidation,
  ModelValidationSchema,
  CustomValidator
} from './validation-framework';

export type {
  PydanticGeneratorConfig,
  GenerationProgress,
  GeneratedModel,
  ValidationRule as GeneratorValidationRule,
  GenerationResult,
  GenerationStatistics,
  ValidationStatistics,
  CognitiveInsight,
  GenerationError,
  GenerationWarning
} from './xml-to-pydantic-generator';

/**
 * Convenience function to create a complete XML to Pydantic generator
 */
export function createXmlToPydanticGenerator(config: PydanticGeneratorConfig) {
  return new XmlToPydanticGenerator(config);
}

/**
 * Convenience function to create a type mapper
 */
export function createTypeMapper(config?: TypeMappingConfig) {
  return new TypeMapper(config);
}

/**
 * Convenience function to create a schema engine
 */
export function createSchemaEngine(config?: SchemaGenerationConfig) {
  return new SchemaEngine(config);
}

/**
 * Convenience function to create a validation framework
 */
export function createValidationFramework(config?: ValidationConfig) {
  return new ValidationFramework(config);
}

// Re-export common utilities
export * from '../types/rtb-types';