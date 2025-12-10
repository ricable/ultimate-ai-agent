// Main component exports
export { PRPViewer } from './PRPViewer';

// Section component exports
export { MetadataSection } from './sections/MetadataSection';
export { ContextSection } from './sections/ContextSection';
export { PersonaSection } from './sections/PersonaSection';
export { FlowSection } from './sections/FlowSection';
export { MetricsSection } from './sections/MetricsSection';
export { PlanSection } from './sections/PlanSection';
export { ListSection } from './sections/ListSection';
export { ObjectSection } from './sections/ObjectSection';
export { KeyValueSection } from './sections/KeyValueSection';
export { FeatureSection } from './sections/FeatureSection';
export { GenericSection } from './sections/GenericSection';

// Renderer exports
export { SectionRenderer } from './renderers/SectionRenderer';

// Type exports
export * from './types/prp.types';

// Utility exports
export { detectSectionType, formatSectionTitle, getSectionIcon } from './utils/sectionDetector';
export { formatKey, formatValue, truncateText, getAccentColor } from './utils/formatters';