export interface RTBParameter {
  id: string;
  name: string;
  vsDataType: string;
  type: string;
  constraints?: ConstraintSpec[] | Record<string, any>;
  description?: string;
  defaultValue?: any;
  hierarchy: string[];
  source: string;
  extractedAt: Date;
  structureGroups?: string[];
  navigationPaths?: string[];
}

export interface MOClass {
  id: string;
  name: string;
  parentClass: string;
  cardinality: MOCardinality;
  flags: Record<string, any>;
  children: string[];
  attributes: string[];
  derivedClasses: string[];
}

export interface MOHierarchy {
  rootClass: string;
  classes: Map<string, MOClass>;
  relationships: Map<string, MORelationship>;
  cardinality: Map<string, MOCardinality>;
  inheritanceChain: Map<string, string[]>;
}

export interface MOCardinality {
  minimum: number;
  maximum: number;
  type: 'single' | 'bounded' | 'unbounded';
}

export interface MORelationship {
  parentId: string;
  relationType: string;
  count?: number;
  description: string;
}

export interface LDNPattern {
  path: string[];
  cardinality: Array<{ mo: string; cardinality: string }>;
  hierarchyLevel: number;
  rootMO: string;
  leafMO: string;
  isValidLDN: boolean;
  description: string;
}

export interface LDNHierarchy {
  rootPatterns: LDNPattern[];
  patternsByLevel: Map<number, LDNPattern[]>;
  lldnStructure: Map<string, LDNPattern>;
  navigationPaths: Map<string, LDNPattern[]>;
}

export interface ReservedByRelationship {
  sourceClass: string;
  targetClass: string;
  relationshipType: 'reserves' | 'depends_on' | 'requires' | 'modifies';
  cardinality: MOCardinality;
  constraints?: Record<string, any>;
  description: string;
}

export interface ReservedByHierarchy {
  totalRelationships: number;
  relationships: Map<string, ReservedByRelationship>;
  classDependencies: Map<string, string[]>;
  constraintValidation: Map<string, ConstraintValidator>;
  circularDependencies: string[];
}

export interface ConstraintValidator {
  validatorType: 'range' | 'enum' | 'pattern' | 'custom';
  rules: any[];
  errorMessage: string;
}

export interface ParameterSpec {
  name: string;
  vsDataType: string;
  type: string;
  constraints?: ConstraintSpec[];
  description?: string;
  defaultValue?: any;
  deprecated?: boolean;
  introduced?: string;
  deprecatedSince?: string;
}

export interface ConstraintSpec {
  type: 'range' | 'enum' | 'pattern' | 'length' | 'required' | 'custom';
  value: any;
  errorMessage?: string;
  severity: 'error' | 'warning' | 'info';
}

export interface RTBTemplate {
  meta?: TemplateMeta;
  custom?: CustomFunction[];
  configuration: Record<string, any>;
  conditions?: Record<string, ConditionOperator>;
  evaluations?: Record<string, EvaluationOperator>;
}

export interface TemplateMeta {
  version: string;
  author: string[];
  description: string;
  tags?: string[];
  environment?: string;
  priority?: number;
  inherits_from?: string | string[];
  source?: string;
}

export interface CustomFunction {
  name: string;
  args: string[];
  body: string[];
}

export interface ConditionOperator {
  if: string;
  then: Record<string, any>;
  else: string | Record<string, any>;
}

export interface EvaluationOperator {
  eval: string;
  args?: any[];
}

export interface RTBProcessorConfig {
  xmlParserOptions?: {
    streaming: boolean;
    batchSize?: number;
    memoryLimit?: number;
  };
  hierarchyOptions?: {
    validateLDN: boolean;
    strictCardinality: boolean;
    enableReservations: boolean;
  };
  validatorOptions?: {
    enableConstraints: boolean;
    customValidators?: Record<string, ConstraintValidator>;
    strictMode: boolean;
  };
}

export interface ProcessingStats {
  xmlProcessingTime: number;
  hierarchyProcessingTime: number;
  validationTime: number;
  totalParameters: number;
  totalMOClasses: number;
  totalRelationships: number;
  memoryUsage: number;
  errorCount: number;
  warningCount: number;
}