// File: frontend/src/components/documents/index.ts
/**
 * Document Processing Components
 * Comprehensive document upload, processing, and management components
 */

export { default as DocumentUpload } from './DocumentUpload';
export type { DocumentUploadProps, DocumentUploadResult, UploadState, UploadedFile } from './DocumentUpload';

export { default as DocumentList } from './DocumentList';
export type { Document, DocumentListProps, DocumentListState } from './DocumentList';

export { default as DocumentDashboard } from './DocumentDashboard';
export type { DocumentDashboardProps, DashboardView, DashboardState } from './DocumentDashboard';

// Re-export for convenience
export {
  DocumentUpload,
  DocumentList,
  DocumentDashboard
};