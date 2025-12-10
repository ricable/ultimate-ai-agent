/**
 * Interfaces for Template Variant Generation
 */

export interface TemplateVariantConfig {
  enableOptimization: boolean;
  validationMode: 'strict' | 'lenient';
  performanceMode: 'speed' | 'quality';
}

export interface VariantGeneratorConfig {
  maxTemplates: number;
  cacheEnabled: boolean;
  logLevel: 'debug' | 'info' | 'warn' | 'error';
}

export interface GeneratedTemplate {
  id: string;
  variantType: 'urban' | 'mobility' | 'sleep';
  template: any;
  metadata: {
    generatedAt: string;
    config: any;
    optimizations: string[];
  };
}

export interface VariantMetrics {
  templateId: string;
  variantType: string;
  generationTime: number;
  parameterCount: number;
  customFunctionCount: number;
  validationResults: {
    valid: boolean;
    errors: string[];
    warnings: string[];
  };
}