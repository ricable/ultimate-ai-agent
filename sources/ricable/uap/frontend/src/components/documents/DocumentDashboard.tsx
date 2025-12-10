import React, { useState, useCallback } from 'react';
import { Upload, List, Settings, BarChart3 } from 'lucide-react';
import DocumentUpload, { DocumentUploadResult } from './DocumentUpload';
import DocumentList, { Document } from './DocumentList';

export interface DocumentDashboardProps {
  className?: string;
}

export type DashboardView = 'upload' | 'list' | 'analytics' | 'settings';

export interface DashboardState {
  activeView: DashboardView;
  recentUploads: DocumentUploadResult[];
  selectedDocument: Document | null;
  showUploadSuccess: boolean;
}

const DocumentDashboard: React.FC<DocumentDashboardProps> = ({ className = '' }) => {
  const [state, setState] = useState<DashboardState>({
    activeView: 'upload',
    recentUploads: [],
    selectedDocument: null,
    showUploadSuccess: false
  });

  const handleUploadSuccess = useCallback((result: DocumentUploadResult) => {
    setState(prev => ({
      ...prev,
      recentUploads: [result, ...prev.recentUploads.slice(0, 9)], // Keep last 10
      showUploadSuccess: true,
      activeView: 'list' // Switch to list view after successful upload
    }));

    // Hide success message after 3 seconds
    setTimeout(() => {
      setState(prev => ({ ...prev, showUploadSuccess: false }));
    }, 3000);
  }, []);

  const handleUploadError = useCallback((error: string) => {
    console.error('Upload error:', error);
    // Error handling is managed by the DocumentUpload component
  }, []);

  const handleDocumentSelect = useCallback((document: Document) => {
    setState(prev => ({ ...prev, selectedDocument: document }));
    // You could open a modal or navigate to a detail view here
    console.log('Selected document:', document);
  }, []);

  const handleDocumentDelete = useCallback((documentId: string) => {
    setState(prev => ({
      ...prev,
      recentUploads: prev.recentUploads.filter(upload => upload.document_id !== documentId),
      selectedDocument: prev.selectedDocument?.document_id === documentId ? null : prev.selectedDocument
    }));
  }, []);

  const setActiveView = useCallback((view: DashboardView) => {
    setState(prev => ({ ...prev, activeView: view }));
  }, []);

  const navigationItems = [
    { id: 'upload', label: 'Upload', icon: Upload },
    { id: 'list', label: 'Documents', icon: List },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
    { id: 'settings', label: 'Settings', icon: Settings },
  ];

  const renderContent = () => {
    switch (state.activeView) {
      case 'upload':
        return (
          <div className="space-y-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 mb-2">
                Upload Documents
              </h1>
              <p className="text-gray-600">
                Upload and process documents using AI-powered analysis
              </p>
            </div>
            
            <DocumentUpload
              onUploadSuccess={handleUploadSuccess}
              onUploadError={handleUploadError}
              processImmediately={true}
            />

            {/* Recent Uploads */}
            {state.recentUploads.length > 0 && (
              <div className="mt-8">
                <h2 className="text-xl font-semibold text-gray-900 mb-4">
                  Recent Uploads
                </h2>
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                  {state.recentUploads.slice(0, 6).map((upload) => (
                    <div
                      key={upload.document_id || upload.filename}
                      className="p-4 bg-white border border-gray-200 rounded-lg shadow-sm"
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <h3 className="text-sm font-medium text-gray-900 truncate">
                            {upload.filename}
                          </h3>
                          <p className="text-xs text-gray-500 mt-1">
                            {new Date(upload.upload_timestamp).toLocaleString()}
                          </p>
                          <span className={`inline-block mt-2 px-2 py-1 text-xs rounded-full ${
                            upload.processing_status === 'completed' 
                              ? 'bg-green-100 text-green-800'
                              : upload.processing_status === 'processing'
                              ? 'bg-yellow-100 text-yellow-800'
                              : 'bg-gray-100 text-gray-800'
                          }`}>
                            {upload.processing_status}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        );

      case 'list':
        return (
          <div className="space-y-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 mb-2">
                Document Library
              </h1>
              <p className="text-gray-600">
                Manage and analyze your uploaded documents
              </p>
            </div>
            
            <DocumentList
              onDocumentSelect={handleDocumentSelect}
              onDocumentDelete={handleDocumentDelete}
              refreshInterval={30000}
            />
          </div>
        );

      case 'analytics':
        return (
          <div className="space-y-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 mb-2">
                Document Analytics
              </h1>
              <p className="text-gray-600">
                Insights and statistics about your document processing
              </p>
            </div>
            
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
              {/* Placeholder analytics cards */}
              <div className="p-6 bg-white border border-gray-200 rounded-lg shadow-sm">
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  Total Documents
                </h3>
                <p className="text-3xl font-bold text-blue-600">
                  {state.recentUploads.length}
                </p>
                <p className="text-sm text-gray-500 mt-2">
                  Documents processed
                </p>
              </div>
              
              <div className="p-6 bg-white border border-gray-200 rounded-lg shadow-sm">
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  Processing Rate
                </h3>
                <p className="text-3xl font-bold text-green-600">98%</p>
                <p className="text-sm text-gray-500 mt-2">
                  Success rate
                </p>
              </div>
              
              <div className="p-6 bg-white border border-gray-200 rounded-lg shadow-sm">
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  Storage Used
                </h3>
                <p className="text-3xl font-bold text-purple-600">2.4 GB</p>
                <p className="text-sm text-gray-500 mt-2">
                  Of available storage
                </p>
              </div>
            </div>

            <div className="p-8 text-center bg-gray-50 rounded-lg">
              <BarChart3 className="w-16 h-16 mx-auto mb-4 text-gray-400" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                Analytics Coming Soon
              </h3>
              <p className="text-gray-600">
                Detailed analytics and insights will be available in a future release.
              </p>
            </div>
          </div>
        );

      case 'settings':
        return (
          <div className="space-y-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 mb-2">
                Document Settings
              </h1>
              <p className="text-gray-600">
                Configure document processing preferences
              </p>
            </div>
            
            <div className="bg-white border border-gray-200 rounded-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                Processing Options
              </h2>
              
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-medium text-gray-900">
                      Auto-process uploads
                    </h3>
                    <p className="text-sm text-gray-500">
                      Automatically start processing when documents are uploaded
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    defaultChecked
                    className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-medium text-gray-900">
                      OCR for scanned documents
                    </h3>
                    <p className="text-sm text-gray-500">
                      Extract text from scanned PDFs and images
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    defaultChecked
                    className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-medium text-gray-900">
                      Table extraction
                    </h3>
                    <p className="text-sm text-gray-500">
                      Extract and structure tables from documents
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    defaultChecked
                    className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                  />
                </div>
              </div>
            </div>

            <div className="p-8 text-center bg-gray-50 rounded-lg">
              <Settings className="w-16 h-16 mx-auto mb-4 text-gray-400" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                More Settings Coming Soon
              </h3>
              <p className="text-gray-600">
                Additional configuration options will be available in a future release.
              </p>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className={`min-h-screen bg-gray-50 ${className}`}>
      {/* Success Notification */}
      {state.showUploadSuccess && (
        <div className="fixed top-4 right-4 z-50 p-4 bg-green-100 border border-green-200 text-green-800 rounded-lg shadow-lg">
          <div className="flex items-center space-x-2">
            <Upload className="w-5 h-5" />
            <span>Document uploaded successfully!</span>
          </div>
        </div>
      )}

      {/* Navigation */}
      <nav className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex">
              <div className="flex-shrink-0 flex items-center">
                <h1 className="text-xl font-bold text-gray-900">
                  Document Processing
                </h1>
              </div>
              <div className="hidden sm:ml-8 sm:flex sm:space-x-8">
                {navigationItems.map((item) => {
                  const Icon = item.icon;
                  return (
                    <button
                      key={item.id}
                      onClick={() => setActiveView(item.id as DashboardView)}
                      className={`inline-flex items-center px-1 pt-1 text-sm font-medium border-b-2 ${
                        state.activeView === item.id
                          ? 'border-blue-500 text-blue-600'
                          : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                      }`}
                    >
                      <Icon className="w-4 h-4 mr-2" />
                      {item.label}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Mobile Navigation */}
      <div className="sm:hidden bg-white border-b border-gray-200">
        <div className="px-4 py-2">
          <select
            value={state.activeView}
            onChange={(e) => setActiveView(e.target.value as DashboardView)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            {navigationItems.map((item) => (
              <option key={item.id} value={item.id}>
                {item.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          {renderContent()}
        </div>
      </main>
    </div>
  );
};

export default DocumentDashboard;