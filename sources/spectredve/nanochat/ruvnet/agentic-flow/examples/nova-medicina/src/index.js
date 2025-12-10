/**
 * Nova Medicina - Main Entry Point
 * AI-powered medical analysis system with anti-hallucination safeguards
 *
 * @author ruv (github.com/ruvnet, ruv.io)
 */

export { default as Analyzer } from './analyzer.js';
export { default as Verifier } from './verifier.js';
export { default as ProviderSearch } from './provider-search.js';
export { default as ConfigManager } from './config-manager.js';

// Version info
export const VERSION = '1.0.0';
export const MEDICAL_DISCLAIMER = `
⚠️  MEDICAL DISCLAIMER:
This tool is designed to SUPPLEMENT, not replace, professional medical advice.
Always consult qualified healthcare providers for medical decisions.
`;

console.log('Nova Medicina module loaded');
