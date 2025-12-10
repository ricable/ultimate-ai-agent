import React from 'react';
import { Card } from '../ui/Card';

export const KnowledgeItemSkeleton: React.FC = () => {
  return (
    <Card className="relative overflow-hidden">
      {/* Shimmer effect overlay */}
      <div className="absolute inset-0 -translate-x-full animate-[shimmer_2s_infinite] bg-gradient-to-r from-transparent via-white/10 to-transparent" />
      
      {/* Icon skeleton */}
      <div className="flex items-start gap-4 mb-4">
        <div className="w-10 h-10 bg-gray-200 dark:bg-zinc-800 rounded-lg animate-pulse" />
        
        {/* Title and metadata skeleton */}
        <div className="flex-1">
          <div className="h-6 bg-gray-200 dark:bg-zinc-800 rounded w-3/4 mb-2 animate-pulse" />
          <div className="h-4 bg-gray-200 dark:bg-zinc-800 rounded w-1/2 animate-pulse" />
        </div>
      </div>
      
      {/* Description skeleton */}
      <div className="space-y-2 mb-4">
        <div className="h-4 bg-gray-200 dark:bg-zinc-800 rounded animate-pulse" />
        <div className="h-4 bg-gray-200 dark:bg-zinc-800 rounded w-5/6 animate-pulse" />
      </div>
      
      {/* Tags skeleton */}
      <div className="flex gap-2 mb-4">
        <div className="h-6 w-16 bg-gray-200 dark:bg-zinc-800 rounded-full animate-pulse" />
        <div className="h-6 w-20 bg-gray-200 dark:bg-zinc-800 rounded-full animate-pulse" />
      </div>
      
      {/* Footer skeleton */}
      <div className="flex justify-between items-center">
        <div className="h-4 bg-gray-200 dark:bg-zinc-800 rounded w-32 animate-pulse" />
        <div className="flex gap-2">
          <div className="w-8 h-8 bg-gray-200 dark:bg-zinc-800 rounded animate-pulse" />
          <div className="w-8 h-8 bg-gray-200 dark:bg-zinc-800 rounded animate-pulse" />
        </div>
      </div>
    </Card>
  );
};

export const KnowledgeGridSkeleton: React.FC = () => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {[...Array(6)].map((_, index) => (
        <KnowledgeItemSkeleton key={index} />
      ))}
    </div>
  );
};

export const KnowledgeTableSkeleton: React.FC = () => {
  return (
    <Card>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-200 dark:border-zinc-800">
              {[...Array(5)].map((_, index) => (
                <th key={index} className="text-left p-4">
                  <div className="h-4 bg-gray-200 dark:bg-zinc-800 rounded w-20 animate-pulse" />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[...Array(5)].map((_, rowIndex) => (
              <tr key={rowIndex} className="border-b border-gray-100 dark:border-zinc-900">
                {[...Array(5)].map((_, colIndex) => (
                  <td key={colIndex} className="p-4">
                    <div className="h-4 bg-gray-200 dark:bg-zinc-800 rounded animate-pulse" />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
};