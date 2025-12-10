import { SectionType, SectionDetectorResult } from '../types/prp.types';

/**
 * Detects the type of a section based on its key and content structure
 */
export function detectSectionType(key: string, value: any): SectionDetectorResult {
  const normalizedKey = key.toLowerCase().replace(/_/g, '').replace(/\s+/g, '');
  
  // Check metadata fields
  if (['title', 'version', 'author', 'date', 'status', 'documenttype'].includes(normalizedKey)) {
    return { type: 'metadata', confidence: 1.0 };
  }
  
  // Check context sections (including common markdown headers)
  if (normalizedKey === 'context' || normalizedKey === 'overview' || 
      normalizedKey === 'executivesummary' || normalizedKey === 'problemstatement' ||
      normalizedKey === 'visionstatement' || normalizedKey === 'proposedsolution' ||
      normalizedKey === 'goal' || normalizedKey === 'objective' || normalizedKey === 'purpose' ||
      normalizedKey === 'why' || normalizedKey === 'rationale' || normalizedKey === 'what' ||
      normalizedKey === 'description' || normalizedKey === 'background') {
    return { type: 'context', confidence: 1.0 };
  }
  
  // Check personas
  if (normalizedKey.includes('persona') || normalizedKey.includes('user') || 
      normalizedKey === 'stakeholders' || normalizedKey === 'targetaudience') {
    // Always treat these as personas, even if structure doesn't match perfectly
    return { type: 'personas', confidence: 0.9 };
  }
  
  // Check flows/journeys
  if (normalizedKey.includes('flow') || normalizedKey.includes('journey') ||
      normalizedKey.includes('workflow') || normalizedKey === 'userexperience') {
    return { type: 'flows', confidence: 0.9 };
  }
  
  // Check metrics (including common markdown headers)
  if (normalizedKey.includes('metric') || normalizedKey.includes('success') || 
      normalizedKey.includes('kpi') || normalizedKey === 'estimatedimpact' ||
      normalizedKey === 'successmetrics' || normalizedKey === 'successcriteria') {
    return { type: 'metrics', confidence: 0.9 };
  }
  
  // Check implementation plans (including common markdown headers)
  if (normalizedKey.includes('plan') || normalizedKey.includes('phase') || 
      normalizedKey.includes('implementation') || normalizedKey.includes('roadmap') ||
      normalizedKey === 'timeline' || normalizedKey === 'rolloutplan' ||
      normalizedKey === 'migrationstrategy' || normalizedKey === 'implementationplan') {
    return { type: 'plan', confidence: 0.9 };
  }
  
  // Check validation/testing (including common markdown headers)
  if (normalizedKey.includes('validation') || normalizedKey.includes('test') || 
      normalizedKey.includes('gate') || normalizedKey === 'compliance' ||
      normalizedKey.includes('quality') || normalizedKey === 'accessibilitystandards' ||
      normalizedKey === 'acceptancecriteria' || normalizedKey === 'qualitygates') {
    return { type: 'list', confidence: 0.8 };
  }
  
  // Check risk assessment
  if (normalizedKey.includes('risk') || normalizedKey === 'riskassessment') {
    return { type: 'list', confidence: 0.9 };
  }
  
  // Check design/architecture sections
  if (normalizedKey.includes('design') || normalizedKey.includes('architecture') ||
      normalizedKey.includes('component') || normalizedKey === 'tokensystem' ||
      normalizedKey === 'designprinciples' || normalizedKey === 'designguidelines') {
    return { type: 'object', confidence: 0.8 };
  }
  
  // Check budget/resources
  if (normalizedKey.includes('budget') || normalizedKey.includes('resource') ||
      normalizedKey.includes('cost') || normalizedKey === 'team' || 
      normalizedKey === 'budgetestimate' || normalizedKey === 'budgetandresources') {
    return { type: 'keyvalue', confidence: 0.9 };
  }
  
  // Check feature requirements specifically
  if (normalizedKey === 'featurerequirements' || normalizedKey === 'features' ||
      normalizedKey === 'capabilities') {
    return { type: 'features', confidence: 0.9 };
  }
  
  // Check requirements
  if (normalizedKey.includes('requirement') || 
      normalizedKey === 'technicalrequirements') {
    return { type: 'object', confidence: 0.8 };
  }
  
  // Check data/information sections
  if (normalizedKey.includes('data') || normalizedKey.includes('information') ||
      normalizedKey === 'currentstateanalysis' || normalizedKey === 'informationarchitecture') {
    return { type: 'object', confidence: 0.8 };
  }
  
  // Check governance/process sections
  if (normalizedKey.includes('governance') || normalizedKey.includes('process') ||
      normalizedKey === 'governancemodel' || normalizedKey === 'testingstrategy') {
    return { type: 'object', confidence: 0.8 };
  }
  
  // Check technical sections
  if (normalizedKey.includes('technical') || normalizedKey.includes('tech') ||
      normalizedKey === 'aimodelspecifications' || normalizedKey === 'performancerequirements' ||
      normalizedKey === 'toolingandinfrastructure' || normalizedKey === 'monitoringandanalytics') {
    return { type: 'object', confidence: 0.8 };
  }
  
  // Analyze value structure
  if (Array.isArray(value)) {
    return { type: 'list', confidence: 0.7 };
  }
  
  if (typeof value === 'object' && value !== null) {
    // Check if it's a simple key-value object
    if (isSimpleKeyValue(value)) {
      return { type: 'keyvalue', confidence: 0.7 };
    }
    
    // Check if it's a complex nested object
    if (hasNestedObjects(value)) {
      return { type: 'object', confidence: 0.7 };
    }
  }
  
  // Default fallback
  return { type: 'generic', confidence: 0.5 };
}

/**
 * Checks if the value structure matches a persona pattern
 */
function isPersonaStructure(value: any): boolean {
  if (typeof value !== 'object' || value === null) return false;
  
  // Check if it's a collection of personas
  const values = Object.values(value);
  if (values.length === 0) return false;
  
  // Check if first value has persona-like properties
  const firstValue = values[0];
  if (typeof firstValue !== 'object') return false;
  
  const personaKeys = ['name', 'role', 'goals', 'pain_points', 'journey', 'workflow'];
  return personaKeys.some(key => key in firstValue);
}

/**
 * Checks if an object is a simple key-value structure
 */
function isSimpleKeyValue(obj: any): boolean {
  if (typeof obj !== 'object' || obj === null) return false;
  
  const values = Object.values(obj);
  return values.every(val => 
    typeof val === 'string' || 
    typeof val === 'number' || 
    typeof val === 'boolean'
  );
}

/**
 * Checks if an object has nested objects
 */
function hasNestedObjects(obj: any): boolean {
  if (typeof obj !== 'object' || obj === null) return false;
  
  const values = Object.values(obj);
  return values.some(val => 
    typeof val === 'object' && 
    val !== null && 
    !Array.isArray(val)
  );
}

/**
 * Gets a suggested icon based on section key
 */
export function getSectionIcon(key: string): string {
  const normalizedKey = key.toLowerCase();
  
  if (normalizedKey.includes('persona') || normalizedKey.includes('user')) return 'Users';
  if (normalizedKey.includes('flow') || normalizedKey.includes('journey')) return 'Workflow';
  if (normalizedKey.includes('metric') || normalizedKey.includes('success')) return 'BarChart3';
  if (normalizedKey.includes('plan') || normalizedKey.includes('implementation')) return 'Clock';
  if (normalizedKey.includes('context') || normalizedKey.includes('overview')) return 'Brain';
  if (normalizedKey.includes('technical') || normalizedKey.includes('tech')) return 'Code';
  if (normalizedKey.includes('validation') || normalizedKey.includes('test')) return 'Shield';
  if (normalizedKey.includes('component') || normalizedKey.includes('architecture')) return 'Layers';
  
  return 'FileText';
}

/**
 * Formats a section key into a human-readable title
 */
export function formatSectionTitle(key: string): string {
  return key
    .replace(/_/g, ' ')
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}