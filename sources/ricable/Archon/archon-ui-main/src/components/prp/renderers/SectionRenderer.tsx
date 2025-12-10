import React from 'react';
import { 
  Brain, Users, Workflow, BarChart3, Clock, Shield, 
  Code, Layers, FileText, List, Hash, Box 
} from 'lucide-react';
import { detectSectionType, formatSectionTitle } from '../utils/sectionDetector';
import { getAccentColor } from '../utils/formatters';

// Import all section components
import { ContextSection } from '../sections/ContextSection';
import { PersonaSection } from '../sections/PersonaSection';
import { FlowSection } from '../sections/FlowSection';
import { MetricsSection } from '../sections/MetricsSection';
import { PlanSection } from '../sections/PlanSection';
import { ListSection } from '../sections/ListSection';
import { ObjectSection } from '../sections/ObjectSection';
import { KeyValueSection } from '../sections/KeyValueSection';
import { FeatureSection } from '../sections/FeatureSection';
import { GenericSection } from '../sections/GenericSection';
import { RolloutPlanSection } from '../sections/RolloutPlanSection';
import { TokenSystemSection } from '../sections/TokenSystemSection';

interface SectionRendererProps {
  sectionKey: string;
  data: any;
  index: number;
  isDarkMode?: boolean;
  sectionOverrides?: Record<string, React.ComponentType<any>>;
}

/**
 * Dynamically renders sections based on their type
 */
export const SectionRenderer: React.FC<SectionRendererProps> = ({
  sectionKey,
  data,
  index,
  isDarkMode = false,
  sectionOverrides = {},
}) => {
  // Skip metadata fields (handled by MetadataSection)
  const metadataFields = ['title', 'version', 'author', 'date', 'status', 'document_type'];
  if (metadataFields.includes(sectionKey)) {
    return null;
  }
  
  // Check for custom override first
  if (sectionOverrides[sectionKey]) {
    const CustomComponent = sectionOverrides[sectionKey];
    return <CustomComponent data={data} title={formatSectionTitle(sectionKey)} />;
  }
  
  // Detect section type
  const { type } = detectSectionType(sectionKey, data);
  
  // Get appropriate icon based on section key
  const getIcon = () => {
    const normalizedKey = sectionKey.toLowerCase();
    if (normalizedKey.includes('context') || normalizedKey.includes('overview')) return <Brain className="w-5 h-5" />;
    if (normalizedKey.includes('persona') || normalizedKey.includes('user')) return <Users className="w-5 h-5" />;
    if (normalizedKey.includes('flow') || normalizedKey.includes('journey')) return <Workflow className="w-5 h-5" />;
    if (normalizedKey.includes('metric') || normalizedKey.includes('success')) return <BarChart3 className="w-5 h-5" />;
    if (normalizedKey.includes('plan') || normalizedKey.includes('implementation')) return <Clock className="w-5 h-5" />;
    if (normalizedKey.includes('validation') || normalizedKey.includes('gate')) return <Shield className="w-5 h-5" />;
    if (normalizedKey.includes('technical') || normalizedKey.includes('tech')) return <Code className="w-5 h-5" />;
    if (normalizedKey.includes('architecture')) return <Layers className="w-5 h-5" />;
    if (Array.isArray(data)) return <List className="w-5 h-5" />;
    if (typeof data === 'object') return <Box className="w-5 h-5" />;
    return <FileText className="w-5 h-5" />;
  };
  
  // Get accent color based on section or index
  const getColor = () => {
    const normalizedKey = sectionKey.toLowerCase();
    if (normalizedKey.includes('context')) return 'blue';
    if (normalizedKey.includes('persona')) return 'purple';
    if (normalizedKey.includes('flow') || normalizedKey.includes('journey')) return 'orange';
    if (normalizedKey.includes('metric')) return 'green';
    if (normalizedKey.includes('plan')) return 'cyan';
    if (normalizedKey.includes('validation')) return 'emerald';
    return getAccentColor(index);
  };
  
  const commonProps = {
    title: formatSectionTitle(sectionKey),
    data,
    icon: getIcon(),
    accentColor: getColor(),
    isDarkMode,
    defaultOpen: index < 5, // Open first 5 sections by default
    isCollapsible: true, // Make all sections collapsible by default
  };
  
  // Check for specific section types by key name first
  const normalizedKey = sectionKey.toLowerCase();
  
  // Special handling for rollout plans
  if (normalizedKey.includes('rollout') || normalizedKey === 'rollout_plan') {
    return <RolloutPlanSection {...commonProps} />;
  }
  
  // Special handling for token systems
  if (normalizedKey.includes('token') || normalizedKey === 'token_system' || 
      normalizedKey === 'design_tokens' || normalizedKey === 'design_system') {
    return <TokenSystemSection {...commonProps} />;
  }
  
  // Render based on detected type
  switch (type) {
    case 'context':
      return <ContextSection {...commonProps} />;
      
    case 'personas':
      return <PersonaSection {...commonProps} />;
      
    case 'flows':
      return <FlowSection {...commonProps} />;
      
    case 'metrics':
      return <MetricsSection {...commonProps} />;
      
    case 'plan':
      return <PlanSection {...commonProps} />;
      
    case 'list':
      return <ListSection {...commonProps} />;
      
    case 'keyvalue':
      return <KeyValueSection {...commonProps} />;
      
    case 'object':
      return <ObjectSection {...commonProps} />;
      
    case 'features':
      return <FeatureSection {...commonProps} />;
      
    case 'generic':
    default:
      return <GenericSection {...commonProps} />;
  }
};