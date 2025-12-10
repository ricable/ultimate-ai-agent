import React, { useState, useCallback, useRef } from 'react';
import { Upload, File, X, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';

export interface DocumentUploadProps {
  onUploadSuccess?: (result: DocumentUploadResult) => void;
  onUploadError?: (error: string) => void;
  maxFileSize?: number; // in bytes
  allowedTypes?: string[];
  processImmediately?: boolean;
}

export interface DocumentUploadResult {
  success: boolean;
  document_id?: string;
  filename: string;
  file_size: number;
  content_type: string;
  upload_timestamp: string;
  processing_status: string;
  error_message?: string;
}

export interface UploadState {
  isDragging: boolean;
  isUploading: boolean;
  uploadProgress?: number;
  uploadedFiles: UploadedFile[];
  errors: string[];
}

export interface UploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
  status: 'uploading' | 'completed' | 'error';
  result?: DocumentUploadResult;
  error?: string;
}

const DocumentUpload: React.FC<DocumentUploadProps> = ({
  onUploadSuccess,
  onUploadError,
  maxFileSize = 50 * 1024 * 1024, // 50MB default
  allowedTypes = ['.pdf', '.docx', '.doc', '.txt', '.md', '.html', '.rtf', '.odt'],
  processImmediately = true
}) => {
  const [uploadState, setUploadState] = useState<UploadState>({
    isDragging: false,
    isUploading: false,
    uploadedFiles: [],
    errors: []
  });

  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateFile = useCallback((file: File): string | null => {
    // Check file size
    if (file.size > maxFileSize) {
      return `File size (${formatFileSize(file.size)}) exceeds maximum allowed size (${formatFileSize(maxFileSize)})`;
    }

    // Check file type
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!allowedTypes.includes(fileExtension)) {
      return `File type ${fileExtension} is not supported. Allowed types: ${allowedTypes.join(', ')}`;
    }

    return null;
  }, [maxFileSize, allowedTypes]);

  const uploadFile = useCallback(async (file: File): Promise<DocumentUploadResult> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('process_immediately', processImmediately.toString());

    const response = await fetch('/api/documents/upload', {
      method: 'POST',
      body: formData,
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('access_token')}` // Adjust based on your auth implementation
      }
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Upload failed');
    }

    return await response.json();
  }, [processImmediately]);

  const handleFiles = useCallback(async (files: FileList | File[]) => {
    const fileArray = Array.from(files);
    const newErrors: string[] = [];
    const validFiles: File[] = [];

    // Validate files
    fileArray.forEach(file => {
      const error = validateFile(file);
      if (error) {
        newErrors.push(`${file.name}: ${error}`);
      } else {
        validFiles.push(file);
      }
    });

    if (newErrors.length > 0) {
      setUploadState(prev => ({
        ...prev,
        errors: [...prev.errors, ...newErrors]
      }));
      newErrors.forEach(error => onUploadError?.(error));
    }

    if (validFiles.length === 0) return;

    // Create file entries
    const fileEntries: UploadedFile[] = validFiles.map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      name: file.name,
      size: file.size,
      type: file.type,
      status: 'uploading'
    }));

    setUploadState(prev => ({
      ...prev,
      isUploading: true,
      uploadedFiles: [...prev.uploadedFiles, ...fileEntries]
    }));

    // Upload files
    for (let i = 0; i < validFiles.length; i++) {
      const file = validFiles[i];
      const fileEntry = fileEntries[i];

      try {
        const result = await uploadFile(file);
        
        setUploadState(prev => ({
          ...prev,
          uploadedFiles: prev.uploadedFiles.map(f => 
            f.id === fileEntry.id 
              ? { ...f, status: 'completed', result }
              : f
          )
        }));

        onUploadSuccess?.(result);
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Upload failed';
        
        setUploadState(prev => ({
          ...prev,
          uploadedFiles: prev.uploadedFiles.map(f => 
            f.id === fileEntry.id 
              ? { ...f, status: 'error', error: errorMessage }
              : f
          )
        }));

        onUploadError?.(errorMessage);
      }
    }

    setUploadState(prev => ({ ...prev, isUploading: false }));
  }, [validateFile, uploadFile, onUploadSuccess, onUploadError]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setUploadState(prev => ({ ...prev, isDragging: true }));
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setUploadState(prev => ({ ...prev, isDragging: false }));
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setUploadState(prev => ({ ...prev, isDragging: false }));
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFiles(files);
    }
  }, [handleFiles]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFiles(files);
    }
    // Reset input value to allow selecting the same file again
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, [handleFiles]);

  const handleClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const removeFile = useCallback((fileId: string) => {
    setUploadState(prev => ({
      ...prev,
      uploadedFiles: prev.uploadedFiles.filter(f => f.id !== fileId)
    }));
  }, []);

  const clearErrors = useCallback(() => {
    setUploadState(prev => ({ ...prev, errors: [] }));
  }, []);

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getStatusIcon = (status: UploadedFile['status']) => {
    switch (status) {
      case 'uploading':
        return <Loader2 className="w-4 h-4 animate-spin text-blue-500" />;
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      default:
        return null;
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      {/* Upload Area */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
        className={`
          relative border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
          ${uploadState.isDragging 
            ? 'border-blue-400 bg-blue-50' 
            : 'border-gray-300 hover:border-gray-400'
          }
          ${uploadState.isUploading ? 'opacity-50 pointer-events-none' : ''}
        `}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept={allowedTypes.join(',')}
          onChange={handleFileSelect}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        
        <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">
          Upload Documents
        </h3>
        <p className="text-gray-600 mb-2">
          Drag and drop files here, or click to select files
        </p>
        <p className="text-sm text-gray-500">
          Supported formats: {allowedTypes.join(', ')}
        </p>
        <p className="text-sm text-gray-500">
          Maximum file size: {formatFileSize(maxFileSize)}
        </p>
      </div>

      {/* Error Messages */}
      {uploadState.errors.length > 0 && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex justify-between items-start">
            <div>
              <h4 className="text-sm font-medium text-red-800 mb-2">Upload Errors:</h4>
              <ul className="text-sm text-red-700 space-y-1">
                {uploadState.errors.map((error, index) => (
                  <li key={index}>• {error}</li>
                ))}
              </ul>
            </div>
            <button
              onClick={clearErrors}
              className="text-red-400 hover:text-red-600"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* File List */}
      {uploadState.uploadedFiles.length > 0 && (
        <div className="mt-6 space-y-3">
          <h4 className="text-lg font-medium text-gray-900">
            Uploaded Files ({uploadState.uploadedFiles.length})
          </h4>
          {uploadState.uploadedFiles.map((file) => (
            <div
              key={file.id}
              className="flex items-center justify-between p-4 bg-gray-50 rounded-lg border"
            >
              <div className="flex items-center space-x-3">
                <File className="w-8 h-8 text-gray-400" />
                <div>
                  <p className="text-sm font-medium text-gray-900">
                    {file.name}
                  </p>
                  <p className="text-sm text-gray-500">
                    {formatFileSize(file.size)}
                    {file.result?.processing_status && (
                      <span className="ml-2">• {file.result.processing_status}</span>
                    )}
                  </p>
                  {file.error && (
                    <p className="text-sm text-red-600 mt-1">{file.error}</p>
                  )}
                </div>
              </div>
              
              <div className="flex items-center space-x-3">
                {getStatusIcon(file.status)}
                <button
                  onClick={() => removeFile(file.id)}
                  className="text-gray-400 hover:text-red-600"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Upload Progress */}
      {uploadState.isUploading && (
        <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-center space-x-3">
            <Loader2 className="w-5 h-5 animate-spin text-blue-500" />
            <span className="text-sm text-blue-700">Uploading files...</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default DocumentUpload;