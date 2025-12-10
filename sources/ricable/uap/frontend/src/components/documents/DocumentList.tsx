import React, { useState, useEffect, useCallback } from 'react';
import { 
  File, 
  Download, 
  Trash2, 
  Eye, 
  Search, 
  Filter, 
  RefreshCw,
  Loader2,
  AlertCircle,
  CheckCircle,
  Clock
} from 'lucide-react';

export interface Document {
  document_id: string;
  filename: string;
  file_size: number;
  processed_size?: number;
  content_type: string;
  upload_timestamp: string;
  processing_timestamp?: string;
  status: 'completed' | 'processing' | 'error';
  error_message?: string;
  page_count?: number;
  word_count?: number;
  language?: string;
}

export interface DocumentListProps {
  onDocumentSelect?: (document: Document) => void;
  onDocumentDelete?: (documentId: string) => void;
  refreshInterval?: number; // milliseconds
}

export interface DocumentListState {
  documents: Document[];
  loading: boolean;
  error: string | null;
  searchQuery: string;
  statusFilter: string;
  currentPage: number;
  hasMore: boolean;
  totalDocuments: number;
}

const DocumentList: React.FC<DocumentListProps> = ({
  onDocumentSelect,
  onDocumentDelete,
  refreshInterval = 30000 // 30 seconds
}) => {
  const [state, setState] = useState<DocumentListState>({
    documents: [],
    loading: true,
    error: null,
    searchQuery: '',
    statusFilter: 'all',
    currentPage: 0,
    hasMore: true,
    totalDocuments: 0
  });

  const [selectedDocuments, setSelectedDocuments] = useState<Set<string>>(new Set());

  const fetchDocuments = useCallback(async (page: number = 0, limit: number = 20, reset: boolean = false) => {
    try {
      setState(prev => ({ ...prev, loading: true, error: null }));

      const params = new URLSearchParams({
        limit: limit.toString(),
        offset: (page * limit).toString(),
        ...(state.statusFilter !== 'all' && { status_filter: state.statusFilter })
      });

      const response = await fetch(`/api/documents?${params}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        }
      });

      if (!response.ok) {
        throw new Error('Failed to fetch documents');
      }

      const data = await response.json();

      setState(prev => ({
        ...prev,
        documents: reset ? data.documents : [...prev.documents, ...data.documents],
        loading: false,
        hasMore: data.has_more,
        totalDocuments: data.total,
        currentPage: page
      }));

    } catch (error) {
      setState(prev => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to load documents'
      }));
    }
  }, [state.statusFilter]);

  const refreshDocuments = useCallback(() => {
    fetchDocuments(0, 20, true);
  }, [fetchDocuments]);

  const loadMoreDocuments = useCallback(() => {
    if (!state.loading && state.hasMore) {
      fetchDocuments(state.currentPage + 1, 20, false);
    }
  }, [fetchDocuments, state.loading, state.hasMore, state.currentPage]);

  const deleteDocument = useCallback(async (documentId: string) => {
    try {
      const response = await fetch(`/api/documents/${documentId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        }
      });

      if (!response.ok) {
        throw new Error('Failed to delete document');
      }

      setState(prev => ({
        ...prev,
        documents: prev.documents.filter(doc => doc.document_id !== documentId),
        totalDocuments: prev.totalDocuments - 1
      }));

      setSelectedDocuments(prev => {
        const newSet = new Set(prev);
        newSet.delete(documentId);
        return newSet;
      });

      onDocumentDelete?.(documentId);

    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to delete document'
      }));
    }
  }, [onDocumentDelete]);

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (isoString: string): string => {
    return new Date(isoString).toLocaleString();
  };

  const getStatusIcon = (status: Document['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'processing':
        return <Clock className="w-4 h-4 text-yellow-500" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      default:
        return null;
    }
  };

  const getStatusColor = (status: Document['status']) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'processing':
        return 'bg-yellow-100 text-yellow-800';
      case 'error':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const filteredDocuments = state.documents.filter(doc => {
    const matchesSearch = doc.filename.toLowerCase().includes(state.searchQuery.toLowerCase());
    const matchesStatus = state.statusFilter === 'all' || doc.status === state.statusFilter;
    return matchesSearch && matchesStatus;
  });

  // Auto-refresh effect
  useEffect(() => {
    if (refreshInterval > 0) {
      const interval = setInterval(refreshDocuments, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [refreshDocuments, refreshInterval]);

  // Initial load
  useEffect(() => {
    fetchDocuments(0, 20, true);
  }, [state.statusFilter]);

  const handleDocumentSelect = (document: Document) => {
    onDocumentSelect?.(document);
  };

  const toggleDocumentSelection = (documentId: string) => {
    setSelectedDocuments(prev => {
      const newSet = new Set(prev);
      if (newSet.has(documentId)) {
        newSet.delete(documentId);
      } else {
        newSet.add(documentId);
      }
      return newSet;
    });
  };

  const deleteSelectedDocuments = async () => {
    const documentIds = Array.from(selectedDocuments);
    for (const id of documentIds) {
      await deleteDocument(id);
    }
    setSelectedDocuments(new Set());
  };

  return (
    <div className="w-full max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Documents</h2>
          <p className="text-gray-600">
            {state.totalDocuments} document{state.totalDocuments !== 1 ? 's' : ''} total
          </p>
        </div>
        <button
          onClick={refreshDocuments}
          disabled={state.loading}
          className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${state.loading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-4 mb-6">
        <div className="flex-1">
          <div className="relative">
            <Search className="absolute left-3 top-3 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search documents..."
              value={state.searchQuery}
              onChange={(e) => setState(prev => ({ ...prev, searchQuery: e.target.value }))}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <Filter className="w-4 h-4 text-gray-400" />
          <select
            value={state.statusFilter}
            onChange={(e) => setState(prev => ({ ...prev, statusFilter: e.target.value }))}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="all">All Status</option>
            <option value="completed">Completed</option>
            <option value="processing">Processing</option>
            <option value="error">Error</option>
          </select>
        </div>
      </div>

      {/* Bulk Actions */}
      {selectedDocuments.size > 0 && (
        <div className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex justify-between items-center">
            <span className="text-sm text-blue-700">
              {selectedDocuments.size} document{selectedDocuments.size !== 1 ? 's' : ''} selected
            </span>
            <button
              onClick={deleteSelectedDocuments}
              className="flex items-center space-x-2 px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700"
            >
              <Trash2 className="w-4 h-4" />
              <span>Delete Selected</span>
            </button>
          </div>
        </div>
      )}

      {/* Error Message */}
      {state.error && (
        <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-5 h-5 text-red-500" />
            <span className="text-red-700">{state.error}</span>
          </div>
        </div>
      )}

      {/* Document List */}
      <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
        {filteredDocuments.length === 0 ? (
          <div className="p-8 text-center">
            {state.loading ? (
              <div className="flex items-center justify-center space-x-2">
                <Loader2 className="w-6 h-6 animate-spin text-blue-500" />
                <span className="text-gray-600">Loading documents...</span>
              </div>
            ) : (
              <div>
                <File className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No documents found</h3>
                <p className="text-gray-600">
                  {state.searchQuery || state.statusFilter !== 'all' 
                    ? 'Try adjusting your search or filters'
                    : 'Upload your first document to get started'
                  }
                </p>
              </div>
            )}
          </div>
        ) : (
          <div className="divide-y divide-gray-200">
            {filteredDocuments.map((document) => (
              <div
                key={document.document_id}
                className={`p-4 hover:bg-gray-50 ${selectedDocuments.has(document.document_id) ? 'bg-blue-50' : ''}`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <input
                      type="checkbox"
                      checked={selectedDocuments.has(document.document_id)}
                      onChange={() => toggleDocumentSelection(document.document_id)}
                      className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                    />
                    <File className="w-8 h-8 text-gray-400" />
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <h3 className="text-sm font-medium text-gray-900">
                          {document.filename}
                        </h3>
                        <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(document.status)}`}>
                          {getStatusIcon(document.status)}
                          <span className="ml-1">{document.status}</span>
                        </span>
                      </div>
                      <div className="flex items-center space-x-4 mt-1 text-sm text-gray-500">
                        <span>{formatFileSize(document.file_size)}</span>
                        <span>{formatDate(document.upload_timestamp)}</span>
                        {document.word_count && (
                          <span>{document.word_count.toLocaleString()} words</span>
                        )}
                        {document.page_count && (
                          <span>{document.page_count} pages</span>
                        )}
                      </div>
                      {document.error_message && (
                        <p className="text-sm text-red-600 mt-1">{document.error_message}</p>
                      )}
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => handleDocumentSelect(document)}
                      className="p-2 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded"
                      title="View document"
                    >
                      <Eye className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => deleteDocument(document.document_id)}
                      className="p-2 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded"
                      title="Delete document"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Load More */}
        {state.hasMore && filteredDocuments.length > 0 && (
          <div className="p-4 border-t border-gray-200">
            <button
              onClick={loadMoreDocuments}
              disabled={state.loading}
              className="w-full py-2 text-blue-600 hover:bg-blue-50 rounded disabled:opacity-50"
            >
              {state.loading ? (
                <div className="flex items-center justify-center space-x-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Loading...</span>
                </div>
              ) : (
                'Load More'
              )}
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default DocumentList;