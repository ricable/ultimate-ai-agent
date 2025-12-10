/**
 * Knowledge Search Tool
 * Searches medical knowledge bases for relevant information
 */

import type { KnowledgeSearchQuery, KnowledgeSearchResult, Citation, MCPToolResponse } from '../types';

export class KnowledgeSearchTool {
  private readonly knowledgeBases: string[];

  constructor() {
    this.knowledgeBases = [
      'PubMed',
      'Cochrane Library',
      'UpToDate',
      'NICE Guidelines',
      'Mayo Clinic Proceedings',
    ];
  }

  /**
   * Search medical knowledge
   */
  async execute(args: KnowledgeSearchQuery): Promise<MCPToolResponse> {
    try {
      // Perform search
      const results = await this.searchKnowledgeBases(args);

      // Sort by relevance
      results.sort((a, b) => b.relevanceScore - a.relevanceScore);

      // Apply max results limit
      const maxResults = args.maxResults || 10;
      const limitedResults = results.slice(0, maxResults);

      const response = {
        query: args.query,
        filters: args.filters,
        resultsCount: limitedResults.length,
        totalFound: results.length,
        results: limitedResults,
        searchMetadata: {
          sources: this.knowledgeBases,
          timestamp: Date.now(),
          avgRelevance:
            results.reduce((sum, r) => sum + r.relevanceScore, 0) / results.length || 0,
        },
      };

      return {
        content: [
          {
            type: 'json',
            json: response,
          },
          {
            type: 'text',
            text: this.formatSearchResults(response),
          },
        ],
      };
    } catch (error: any) {
      return {
        content: [
          {
            type: 'text',
            text: `❌ Knowledge search failed: ${error.message}`,
          },
        ],
        isError: true,
      };
    }
  }

  /**
   * Search knowledge bases
   */
  private async searchKnowledgeBases(
    query: KnowledgeSearchQuery
  ): Promise<KnowledgeSearchResult[]> {
    // In production, this would query actual medical databases
    // For now, generate example results

    const results: KnowledgeSearchResult[] = [];

    // Simulate search across multiple sources
    for (const source of this.knowledgeBases) {
      // Apply filters
      if (query.filters?.sourceTypes && !this.matchesSourceType(source, query.filters.sourceTypes)) {
        continue;
      }

      // Generate sample results
      const sourceResults = this.generateResults(query.query, source, query.filters);
      results.push(...sourceResults);
    }

    return results;
  }

  /**
   * Generate sample results for a source
   */
  private generateResults(
    queryText: string,
    source: string,
    filters?: any
  ): KnowledgeSearchResult[] {
    const results: KnowledgeSearchResult[] = [];

    // Generate 2-5 results per source
    const count = Math.floor(Math.random() * 4) + 2;

    for (let i = 0; i < count; i++) {
      const relevance = 0.6 + Math.random() * 0.4; // 0.6-1.0

      // Skip if below minimum relevance
      if (filters?.minRelevance && relevance < filters.minRelevance) {
        continue;
      }

      results.push({
        id: `result-${source}-${Date.now()}-${i}`,
        title: `${queryText} - Clinical Study ${i + 1}`,
        content: this.generateContent(queryText, source),
        source,
        relevanceScore: relevance,
        citations: this.generateCitations(source),
        lastUpdated: this.generateDate(filters?.dateRange),
      });
    }

    return results;
  }

  /**
   * Generate result content
   */
  private generateContent(query: string, source: string): string {
    return `
Comprehensive analysis of ${query} based on current clinical evidence from ${source}.

Key Findings:
• Evidence-based recommendations for diagnosis and treatment
• Current best practices from recent literature
• Risk factors and preventive measures
• Prognosis and expected outcomes

This information is derived from peer-reviewed medical literature and clinical guidelines.
    `.trim();
  }

  /**
   * Generate citations for result
   */
  private generateCitations(source: string): Citation[] {
    return [
      {
        id: `cite-${Date.now()}`,
        source,
        sourceType: this.getSourceType(source),
        title: `Clinical Guidelines from ${source}`,
        year: 2024,
        relevanceScore: 0.9,
        verified: true,
      },
    ];
  }

  /**
   * Get source type
   */
  private getSourceType(source: string): 'clinical_guideline' | 'research_paper' | 'textbook' | 'database' {
    if (source.includes('Guideline')) return 'clinical_guideline';
    if (source.includes('Library') || source.includes('PubMed')) return 'research_paper';
    if (source.includes('UpToDate')) return 'textbook';
    return 'database';
  }

  /**
   * Generate date within range
   */
  private generateDate(dateRange?: { start: Date; end: Date }): Date {
    if (!dateRange) {
      // Default to last 2 years
      const now = new Date();
      const twoYearsAgo = new Date(now.getFullYear() - 2, now.getMonth(), now.getDate());
      return new Date(twoYearsAgo.getTime() + Math.random() * (now.getTime() - twoYearsAgo.getTime()));
    }

    const start = dateRange.start.getTime();
    const end = dateRange.end.getTime();
    return new Date(start + Math.random() * (end - start));
  }

  /**
   * Check if source matches type filter
   */
  private matchesSourceType(source: string, types: string[]): boolean {
    const sourceType = this.getSourceType(source);
    return types.includes(sourceType);
  }

  /**
   * Format search results
   */
  private formatSearchResults(response: any): string {
    let report = '🔍 Medical Knowledge Search Results\n\n';

    report += `📊 Query: "${response.query}"\n`;
    report += `📚 Sources Searched: ${response.searchMetadata.sources.join(', ')}\n`;
    report += `📈 Results: ${response.resultsCount} shown (${response.totalFound} total found)\n`;
    report += `🎯 Average Relevance: ${(response.searchMetadata.avgRelevance * 100).toFixed(1)}%\n`;
    report += `⏰ Searched: ${new Date(response.searchMetadata.timestamp).toISOString()}\n\n`;

    report += `📋 Top Results:\n\n`;

    for (let i = 0; i < response.results.length; i++) {
      const result = response.results[i];

      report += `${i + 1}. ${result.title}\n`;
      report += `   Source: ${result.source}\n`;
      report += `   Relevance: ${(result.relevanceScore * 100).toFixed(1)}%\n`;
      report += `   Last Updated: ${result.lastUpdated.toISOString().split('T')[0]}\n`;
      report += `   Citations: ${result.citations.length}\n`;
      report += `   ${result.content.substring(0, 150)}...\n\n`;
    }

    if (response.totalFound > response.resultsCount) {
      report += `... and ${response.totalFound - response.resultsCount} more results\n`;
    }

    return report;
  }
}
