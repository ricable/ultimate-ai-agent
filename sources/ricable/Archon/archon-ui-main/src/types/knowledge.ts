export interface KnowledgeItem {
  id: string;
  title: string;
  description: string;
  source: string;
  sourceType: 'url' | 'file';
  sourceUrl?: string;
  fileName?: string;
  fileType?: string;
  knowledgeType: 'technical' | 'business';
  tags: string[];
  lastUpdated: string;
  nextUpdate?: string;
  status: 'active' | 'processing' | 'error';
  metadata: {
    size: string;
    pageCount?: number;
    wordCount?: number;
    lastScraped?: string;
  };
}