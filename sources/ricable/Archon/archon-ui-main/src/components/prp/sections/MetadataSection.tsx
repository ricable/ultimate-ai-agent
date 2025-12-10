import React from 'react';
import { Award, Users, Clock, Tag, FileText } from 'lucide-react';
import { PRPContent } from '../types/prp.types';

interface MetadataSectionProps {
  content: PRPContent;
  isDarkMode?: boolean;
}

/**
 * Renders the metadata header section of a PRP document
 */
export const MetadataSection: React.FC<MetadataSectionProps> = ({ content, isDarkMode = false }) => {
  const getIcon = (field: string) => {
    switch (field) {
      case 'version': return <Award className="w-4 h-4 text-blue-500" />;
      case 'author': return <Users className="w-4 h-4 text-purple-500" />;
      case 'date': return <Clock className="w-4 h-4 text-green-500" />;
      case 'status': return <Tag className="w-4 h-4 text-orange-500" />;
      default: return <FileText className="w-4 h-4 text-gray-500" />;
    }
  };
  
  const formatStatus = (status: string) => {
    const statusColors = {
      draft: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400',
      review: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400',
      approved: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
      published: 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400',
    };
    
    const colorClass = statusColors[status.toLowerCase() as keyof typeof statusColors] || 
                      'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400';
    
    return (
      <span className={`px-2 py-1 rounded-full text-xs font-medium ${colorClass}`}>
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </span>
    );
  };
  
  const metadataFields = ['version', 'author', 'date', 'status'];
  const hasMetadata = metadataFields.some(field => content[field]);
  
  if (!hasMetadata && !content.title) {
    return null;
  }
  
  return (
    <div className="mb-8 p-6 rounded-xl bg-gradient-to-r from-blue-500/10 to-purple-500/10 border border-blue-200 dark:border-blue-800">
      <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
        {content.title || 'Product Requirements Prompt'}
      </h1>
      
      <div className="flex flex-wrap gap-4 text-sm">
        {metadataFields.map(field => {
          const value = content[field];
          if (!value) return null;
          
          return (
            <div key={field} className="flex items-center gap-2">
              {getIcon(field)}
              {field === 'status' ? (
                formatStatus(value)
              ) : (
                <span className="text-gray-600 dark:text-gray-400">
                  {field === 'version' && 'Version'} {value}
                </span>
              )}
            </div>
          );
        })}
        
        {content.document_type && (
          <div className="flex items-center gap-2">
            <FileText className="w-4 h-4 text-indigo-500" />
            <span className="text-gray-600 dark:text-gray-400 capitalize">
              {content.document_type}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};