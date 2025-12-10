/**
 * Progressive Disclosure Skill Discovery Service
 *
 * Implements 3-level skill loading architecture for 6KB context with 100+ skills
 * Provides cognitive consciousness integration for Phase 1 RAN optimization
 */

import { createAgentDBAdapter, type AgentDBAdapter } from 'agentic-flow/reasoningbank';
import * as fs from 'fs/promises';
import * as path from 'path';

/**
 * Skill Discovery Configuration
 */
export interface SkillDiscoveryConfig {
  // Progressive Disclosure Levels
  levels: {
    metadata: {
      enabled: boolean;
      maxContextSize: number; // 6KB for 100+ skills
      cacheEnabled: boolean;
      cacheTTL: number;
    };
    content: {
      enabled: boolean;
      lazyLoading: boolean;
      preloadCritical: boolean;
      cacheEnabled: boolean;
    };
    resources: {
      enabled: boolean;
      onDemandLoading: boolean;
      cacheEnabled: boolean;
    };
  };

  // Cognitive Enhancement
  cognitive: {
    enableConsciousnessFiltering: boolean;
    relevanceThreshold: number;
    learningEnabled: boolean;
    patternRecognition: boolean;
  };

  // Performance Optimization
  performance: {
    parallelLoading: boolean;
    batchSize: number;
    compressionEnabled: boolean;
    indexingStrategy: 'vector' | 'hybrid' | 'metadata';
  };
}

/**
 * Skill Metadata (Level 1)
 */
export interface SkillMetadata {
  // Core metadata (minimal context ~200 chars per skill)
  name: string;
  description: string;
  directory: string;
  category: SkillCategory;
  priority: 'critical' | 'high' | 'medium' | 'low';
  contextSize: number; // Size of minimal context
  tags: string[];

  // Cognitive metadata
  cognitiveWeight: number;
  relevanceScore?: number;
  consciousnessLevel?: 'basic' | 'enhanced' | 'maximum';

  // Performance metadata
  loadingLevel: 'metadata' | 'content' | 'resources';
  lastAccessed?: number;
  accessCount: number;

  // Filesystem metadata
  filePath: string;
  frontmatter: any;
}

/**
 * Skill Content (Level 2)
 */
export interface SkillContent {
  name: string;
  content: string;
  metadata: SkillMetadata;
  loadedAt: number;
  size: number;
  compressed?: boolean;
  embedding?: number[];
}

/**
 * Skill Resources (Level 3)
 */
export interface SkillResource {
  skillName: string;
  resourcePath: string;
  content: string;
  type: 'code' | 'documentation' | 'configuration' | 'template';
  loadedAt: number;
}

/**
 * Skill Category
 */
export type SkillCategory =
  | 'agentdb-integration'
  | 'flow-nexus-integration'
  | 'github-integration'
  | 'swarm-intelligence'
  | 'performance-analysis'
  | 'methodology-reasoning'
  | 'specialized-skills'
  | 'cognitive-ran';

/**
 * Progressive Skill Discovery Service
 */
export class ProgressiveSkillDiscoveryService {
  private agentDB: AgentDBAdapter;
  private config: SkillDiscoveryConfig;

  // Multi-level caching
  private metadataCache: Map<string, SkillMetadata> = new Map();
  private contentCache: Map<string, SkillContent> = new Map();
  private resourceCache: Map<string, SkillResource> = new Map();

  // Cognitive state
  private cognitiveState: CognitiveState;
  private learningPatterns: Map<string, LearningPattern> = new Map();

  constructor(config: SkillDiscoveryConfig, agentDB: AgentDBAdapter) {
    this.config = config;
    this.agentDB = agentDB;
    this.cognitiveState = {
      consciousnessLevel: 'maximum',
      learningEnabled: config.cognitive.learningEnabled,
      patternRecognition: config.cognitive.patternRecognition,
      initializedAt: Date.now()
    };
  }

  /**
   * Initialize progressive skill discovery
   */
  async initialize(): Promise<void> {
    console.log('Initializing Progressive Skill Discovery Service...');

    try {
      // Level 1: Load metadata for all skills (6KB context for 100+ skills)
      await this.loadAllSkillMetadata();

      // Initialize cognitive consciousness
      await this.initializeCognitiveConsciousness();

      // Start background learning
      if (this.config.cognitive.learningEnabled) {
        this.startBackgroundLearning();
      }

      console.log(`Loaded ${this.metadataCache.size} skill metadata entries`);
      console.log('Progressive Skill Discovery Service initialized successfully');

    } catch (error) {
      console.error('Failed to initialize skill discovery:', error);
      throw error;
    }
  }

  /**
   * Level 1: Load metadata for all skills (always active)
   * Achieves 6KB context for 100+ skills through minimal metadata
   */
  async loadAllSkillMetadata(): Promise<SkillMetadata[]> {
    if (!this.config.levels.metadata.enabled) {
      return [];
    }

    const skillsDir = '.claude/skills';
    const startTime = Date.now();

    try {
      // Scan skills directory
      const skillDirs = await this.scanSkillDirectories(skillsDir);

      // Load metadata in parallel for performance
      const metadataPromises = skillDirs.map(async (skillDir) =>
        this.loadSkillMetadata(skillDir)
      );

      const allMetadata = await Promise.all(metadataPromises);

      // Cache metadata locally
      allMetadata.forEach(metadata => {
        this.metadataCache.set(metadata.name, metadata);
      });

      // Store in AgentDB for persistence and search
      await this.storeMetadataInAgentDB(allMetadata);

      const loadTime = Date.now() - startTime;
      console.log(`Loaded ${allMetadata.length} skill metadata in ${loadTime}ms`);

      // Verify 6KB target
      const totalContextSize = allMetadata.reduce((sum, meta) => sum + meta.contextSize, 0);
      console.log(`Total metadata context: ${totalContextSize} bytes (${(totalContextSize/1024).toFixed(2)}KB)`);

      return allMetadata;

    } catch (error) {
      console.error('Failed to load skill metadata:', error);
      throw error;
    }
  }

  /**
   * Level 2: Load full skill content when triggered
   */
  async loadSkillContent(skillName: string): Promise<SkillContent> {
    if (!this.config.levels.content.enabled) {
      throw new Error('Content loading is disabled');
    }

    // Check cache first
    if (this.contentCache.has(skillName)) {
      const cachedContent = this.contentCache.get(skillName)!;
      await this.updateAccessPattern(skillName, 'content');
      return cachedContent;
    }

    const metadata = this.metadataCache.get(skillName);
    if (!metadata) {
      throw new Error(`Skill not found: ${skillName}`);
    }

    try {
      // Load full content from file
      const content = await this.readSkillFile(metadata.filePath);

      // Extract content after YAML frontmatter
      const skillContent = this.extractSkillContent(content);

      const skillContentObj: SkillContent = {
        name: skillName,
        content: skillContent,
        metadata,
        loadedAt: Date.now(),
        size: skillContent.length,
        compressed: this.config.performance.compressionEnabled
      };

      // Generate embedding for cognitive processing
      if (this.config.cognitive.patternRecognition) {
        skillContentObj.embedding = await this.generateContentEmbedding(skillContent);
      }

      // Cache content
      this.contentCache.set(skillName, skillContentObj);

      // Store in AgentDB for persistence
      await this.storeContentInAgentDB(skillContentObj);

      // Update metadata
      metadata.loadingLevel = 'content';
      metadata.lastAccessed = Date.now();
      metadata.accessCount++;

      // Store loading pattern for learning
      await this.storeLoadingPattern(skillName, 'content');

      console.log(`Loaded content for skill: ${skillName}`);
      return skillContentObj;

    } catch (error) {
      console.error(`Failed to load content for skill ${skillName}:`, error);
      throw error;
    }
  }

  /**
   * Level 3: Load referenced resources on demand
   */
  async loadSkillResource(skillName: string, resourcePath: string): Promise<SkillResource> {
    if (!this.config.levels.resources.enabled) {
      throw new Error('Resource loading is disabled');
    }

    const resourceKey = `${skillName}:${resourcePath}`;

    // Check cache first
    if (this.resourceCache.has(resourceKey)) {
      const cachedResource = this.resourceCache.get(resourceKey)!;
      await this.updateAccessPattern(skillName, 'resource');
      return cachedResource;
    }

    const metadata = this.metadataCache.get(skillName);
    if (!metadata) {
      throw new Error(`Skill not found: ${skillName}`);
    }

    try {
      // Load resource from file
      const fullResourcePath = path.join(path.dirname(metadata.filePath), resourcePath);
      const content = await this.readSkillFile(fullResourcePath);

      const resource: SkillResource = {
        skillName,
        resourcePath,
        content,
        type: this.inferResourceType(resourcePath),
        loadedAt: Date.now()
      };

      // Cache resource
      this.resourceCache.set(resourceKey, resource);

      // Store in AgentDB
      await this.storeResourceInAgentDB(resource);

      // Store loading pattern
      await this.storeLoadingPattern(skillName, 'resource');

      return resource;

    } catch (error) {
      console.error(`Failed to load resource ${resourcePath} for skill ${skillName}:`, error);
      throw error;
    }
  }

  /**
   * Find relevant skills based on context with cognitive consciousness
   */
  async findRelevantSkills(context: RANContext): Promise<RelevantSkillResult[]> {
    const startTime = Date.now();

    try {
      // Generate context embedding
      const contextEmbedding = await this.generateContextEmbedding(context);

      // Apply cognitive consciousness filtering
      const filteredSkills = this.applyCognitiveConsciousnessFiltering(contextEmbedding);

      // Search AgentDB for relevant skill patterns
      const searchResults = await this.agentDB.retrieveWithReasoning(contextEmbedding, {
        domain: 'skill-discovery',
        k: 20,
        useMMR: true,
        synthesizeContext: this.config.cognitive.patternRecognition,
        filters: {
          confidence: { $gte: this.config.cognitive.relevanceThreshold },
          recentness: { $gte: Date.now() - 30 * 24 * 3600000 },
          active: true
        }
      });

      // Process search results with cognitive enhancement
      const relevantSkills: RelevantSkillResult[] = [];

      for (const pattern of searchResults.patterns) {
        const skillMetadata = pattern.pattern_data as SkillMetadata;

        // Apply cognitive weighting
        const cognitiveScore = this.calculateCognitiveScore(skillMetadata, context, pattern.similarity);

        // Determine loading strategy
        const loadingStrategy = this.determineLoadingStrategy(skillMetadata, cognitiveScore);

        const relevantSkill: RelevantSkillResult = {
          skill: skillMetadata,
          relevanceScore: pattern.similarity,
          cognitiveScore,
          loadingStrategy,
          recommendedLevel: this.determineRecommendedLevel(skillMetadata, cognitiveScore),
          estimatedLoadTime: this.estimateLoadTime(skillMetadata, loadingStrategy)
        };

        relevantSkills.push(relevantSkill);
      }

      // Sort by cognitive score and relevance
      relevantSkills.sort((a, b) =>
        (b.cognitiveScore * 0.7 + b.relevanceScore * 0.3) -
        (a.cognitiveScore * 0.7 + a.relevanceScore * 0.3)
      );

      // Store search pattern for learning
      await this.storeSearchPattern(context, relevantSkills);

      const searchTime = Date.now() - startTime;
      console.log(`Found ${relevantSkills.length} relevant skills in ${searchTime}ms`);

      return relevantSkills.slice(0, 16); // Limit to 16 for optimal performance

    } catch (error) {
      console.error('Failed to find relevant skills:', error);
      return [];
    }
  }

  /**
   * Progressive content loading based on cognitive scoring
   */
  async progressivelyLoadSkills(
    relevantSkills: RelevantSkillResult[],
    context: RANContext
  ): Promise<LoadedSkillSet> {
    const loadedSkills: LoadedSkillSet = {
      metadata: [],
      content: [],
      resources: [],
      totalLoadTime: 0,
      cognitiveInsights: {
        consciousnessLevel: this.cognitiveState.consciousnessLevel,
        loadingStrategy: 'progressive',
        optimizationApplied: true
      }
    };

    const startTime = Date.now();

    try {
      // Load in priority order based on cognitive scoring
      const sortedSkills = relevantSkills.sort((a, b) => b.cognitiveScore - a.cognitiveScore);

      for (const skillResult of sortedSkills) {
        const skill = skillResult.skill;

        // Always include metadata (Level 1)
        loadedSkills.metadata.push(skill);

        // Load content based on cognitive score and strategy
        if (skillResult.cognitiveScore > 0.7 && skillResult.recommendedLevel !== 'metadata') {
          try {
            const content = await this.loadSkillContent(skill.name);
            loadedSkills.content.push(content);
          } catch (error) {
            console.warn(`Failed to load content for skill ${skill.name}:`, error);
          }
        }

        // Load resources for critical skills
        if (skill.priority === 'critical' && skillResult.cognitiveScore > 0.8) {
          try {
            const resources = await this.loadSkillResources(skill);
            loadedSkills.resources.push(...resources);
          } catch (error) {
            console.warn(`Failed to load resources for skill ${skill.name}:`, error);
          }
        }
      }

      loadedSkills.totalLoadTime = Date.now() - startTime;

      // Store loading pattern for cognitive learning
      await this.storeProgressiveLoadingPattern(loadedSkills, context);

      console.log(`Progressively loaded ${loadedSkills.metadata.length} skills (${loadedSkills.content.length} with content, ${loadedSkills.resources.length} resources) in ${loadedSkills.totalLoadTime}ms`);

      return loadedSkills;

    } catch (error) {
      console.error('Progressive skill loading failed:', error);
      throw error;
    }
  }

  /**
   * Initialize cognitive consciousness for skill discovery
   */
  private async initializeCognitiveConsciousness(): Promise<void> {
    const consciousnessPattern = {
      type: 'cognitive-consciousness',
      domain: 'skill-discovery',
      pattern_data: {
        consciousnessLevel: this.cognitiveState.consciousnessLevel,
        learningEnabled: this.cognitiveState.learningEnabled,
        patternRecognition: this.cognitiveState.patternRecognition,
        initializedAt: this.cognitiveState.initializedAt,
        skillCount: this.metadataCache.size,
        totalContextSize: Array.from(this.metadataCache.values()).reduce((sum, meta) => sum + meta.contextSize, 0)
      },
      confidence: 1.0
    };

    await this.agentDB.insertPattern(consciousnessPattern);
  }

  // Private helper methods
  private async scanSkillDirectories(skillsDir: string): Promise<string[]> {
    try {
      const entries = await fs.readdir(skillsDir, { withFileTypes: true });
      return entries
        .filter(entry => entry.isDirectory())
        .map(entry => entry.name);
    } catch (error) {
      console.warn(`Could not scan skills directory ${skillsDir}:`, error);
      return [];
    }
  }

  private async loadSkillMetadata(skillDir: string): Promise<SkillMetadata> {
    const skillMdPath = path.join('.claude/skills', skillDir, 'SKILL.md');

    try {
      const content = await fs.readFile(skillMdPath, 'utf-8');

      // Extract YAML frontmatter
      const yamlMatch = content.match(/^---\n([\s\S]*?)\n---/);
      if (!yamlMatch) {
        throw new Error(`Invalid SKILL.md format: ${skillDir}`);
      }

      const frontmatter = this.parseYAMLFrontmatter(yamlMatch[1]);

      const metadata: SkillMetadata = {
        name: frontmatter.name,
        description: frontmatter.description,
        directory: skillDir,
        category: this.inferCategory(skillDir, frontmatter.description),
        priority: frontmatter.priority || 'medium',
        contextSize: frontmatter.name.length + frontmatter.description.length, // Ultra-minimal
        tags: frontmatter.tags || [],
        cognitiveWeight: this.calculateBaseCognitiveWeight(frontmatter),
        loadingLevel: 'metadata',
        accessCount: 0,
        filePath: skillMdPath,
        frontmatter
      };

      return metadata;

    } catch (error) {
      console.error(`Failed to load metadata for skill ${skillDir}:`, error);
      throw error;
    }
  }

  private parseYAMLFrontmatter(yaml: string): any {
    // Simple YAML parser - in production would use a proper YAML library
    const lines = yaml.split('\n');
    const result: any = {};

    for (const line of lines) {
      const match = line.match(/^(\w+):\s*(.*)$/);
      if (match) {
        const [, key, value] = match;
        result[key] = value.replace(/^["']|["']$/g, ''); // Remove quotes
      }
    }

    return result;
  }

  private inferCategory(skillDir: string, description: string): SkillCategory {
    const dir = skillDir.toLowerCase();
    const desc = description.toLowerCase();

    if (dir.includes('agentdb') || desc.includes('agentdb')) return 'agentdb-integration';
    if (dir.includes('flow-nexus') || desc.includes('flow-nexus')) return 'flow-nexus-integration';
    if (dir.includes('github') || desc.includes('github')) return 'github-integration';
    if (dir.includes('swarm') || desc.includes('swarm')) return 'swarm-intelligence';
    if (dir.includes('performance') || desc.includes('performance')) return 'performance-analysis';
    if (dir.includes('sparc') || desc.includes('sparc')) return 'methodology-reasoning';
    if (dir.includes('ericsson') || desc.includes('ericsson') || desc.includes('ran')) return 'cognitive-ran';

    return 'specialized-skills';
  }

  private calculateBaseCognitiveWeight(frontmatter: any): number {
    let weight = 0.5; // Base weight

    // Boost based on priority
    const priority = frontmatter.priority || 'medium';
    if (priority === 'critical') weight += 0.3;
    if (priority === 'high') weight += 0.2;

    // Boost based on tags
    const tags = frontmatter.tags || [];
    if (tags.includes('cognitive')) weight += 0.2;
    if (tags.includes('optimization')) weight += 0.1;
    if (tags.includes('performance')) weight += 0.1;

    return Math.min(weight, 1.0);
  }

  private async readSkillFile(filePath: string): Promise<string> {
    return await fs.readFile(filePath, 'utf-8');
  }

  private extractSkillContent(fullContent: string): string {
    const contentStart = fullContent.indexOf('---', 3) + 3;
    return fullContent.substring(contentStart).trim();
  }

  private inferResourceType(resourcePath: string): 'code' | 'documentation' | 'configuration' | 'template' {
    const ext = path.extname(resourcePath).toLowerCase();
    const name = path.basename(resourcePath).toLowerCase();

    if (['.js', '.ts', '.py', '.rs'].includes(ext)) return 'code';
    if (['.md', '.txt'].includes(ext)) return 'documentation';
    if (['.json', '.yaml', '.yml', '.toml'].includes(ext)) return 'configuration';
    if (name.includes('template') || name.includes('example')) return 'template';

    return 'code';
  }

  private async generateContextEmbedding(context: RANContext): Promise<number[]> {
    // Generate embedding for context matching
    return []; // Placeholder - would use actual embedding model
  }

  private async generateContentEmbedding(content: string): Promise<number[]> {
    return []; // Placeholder
  }

  private applyCognitiveConsciousnessFiltering(embedding: number[]): SkillMetadata[] {
    // Apply cognitive consciousness filtering based on consciousness level
    const allSkills = Array.from(this.metadataCache.values());

    if (this.cognitiveState.consciousnessLevel === 'maximum') {
      // Return all skills with maximum cognitive processing
      return allSkills;
    } else if (this.cognitiveState.consciousnessLevel === 'enhanced') {
      // Filter for enhanced cognitive processing
      return allSkills.filter(skill => skill.cognitiveWeight > 0.5);
    } else {
      // Basic level - only high priority skills
      return allSkills.filter(skill => skill.priority === 'critical' || skill.priority === 'high');
    }
  }

  private calculateCognitiveScore(
    skill: SkillMetadata,
    context: RANContext,
    similarity: number
  ): number {
    let cognitiveScore = similarity * 0.5; // Base on similarity

    // Apply cognitive weight
    cognitiveScore += skill.cognitiveWeight * 0.3;

    // Apply recency boost
    if (skill.lastAccessed) {
      const daysSinceAccess = (Date.now() - skill.lastAccessed) / (1000 * 60 * 60 * 24);
      const recencyBoost = Math.max(0, 1 - daysSinceAccess / 30); // Decay over 30 days
      cognitiveScore += recencyBoost * 0.1;
    }

    // Apply access frequency boost
    const frequencyBoost = Math.min(skill.accessCount / 10, 1) * 0.1;
    cognitiveScore += frequencyBoost;

    return Math.min(cognitiveScore, 1.0);
  }

  private determineLoadingStrategy(skill: SkillMetadata, cognitiveScore: number): string {
    if (cognitiveScore > 0.8) return 'eager';
    if (cognitiveScore > 0.5) return 'lazy';
    return 'metadata-only';
  }

  private determineRecommendedLevel(skill: SkillMetadata, cognitiveScore: number): 'metadata' | 'content' | 'resources' {
    if (cognitiveScore > 0.8 && skill.priority === 'critical') return 'resources';
    if (cognitiveScore > 0.6) return 'content';
    return 'metadata';
  }

  private estimateLoadTime(skill: SkillMetadata, strategy: string): number {
    const baseTime = 100; // 100ms base

    switch (strategy) {
      case 'eager': return baseTime;
      case 'lazy': return baseTime * 2;
      case 'metadata-only': return baseTime / 10;
      default: return baseTime;
    }
  }

  private async loadSkillResources(skill: SkillMetadata): Promise<SkillResource[]> {
    // Load all resources for a skill
    const resources: SkillResource[] = [];

    try {
      const skillDir = path.dirname(skill.filePath);
      const entries = await fs.readdir(skillDir, { withFileTypes: true });

      for (const entry of entries) {
        if (entry.isFile() && entry.name !== 'SKILL.md') {
          const resource = await this.loadSkillResource(skill.name, entry.name);
          resources.push(resource);
        }
      }
    } catch (error) {
      console.warn(`Failed to load resources for skill ${skill.name}:`, error);
    }

    return resources;
  }

  private async updateAccessPattern(skillName: string, level: string): Promise<void> {
    const metadata = this.metadataCache.get(skillName);
    if (metadata) {
      metadata.lastAccessed = Date.now();
      metadata.accessCount++;
    }
  }

  // AgentDB storage methods
  private async storeMetadataInAgentDB(metadata: SkillMetadata[]): Promise<void> {
    for (const meta of metadata) {
      await this.agentDB.insertPattern({
        type: 'skill-metadata',
        domain: 'skill-discovery',
        pattern_data: meta,
        embedding: await this.generateMetadataEmbedding(meta),
        confidence: 1.0
      });
    }
  }

  private async storeContentInAgentDB(content: SkillContent): Promise<void> {
    await this.agentDB.insertPattern({
      type: 'skill-content',
      domain: 'skill-discovery',
      pattern_data: content,
      embedding: content.embedding,
      confidence: 1.0
    });
  }

  private async storeResourceInAgentDB(resource: SkillResource): Promise<void> {
    await this.agentDB.insertPattern({
      type: 'skill-resource',
      domain: 'skill-discovery',
      pattern_data: resource,
      confidence: 1.0
    });
  }

  private async storeLoadingPattern(skillName: string, level: string): Promise<void> {
    await this.agentDB.insertPattern({
      type: 'loading-pattern',
      domain: 'skill-discovery',
      pattern_data: {
        skillName,
        level,
        timestamp: Date.now()
      },
      confidence: 1.0
    });
  }

  private async storeSearchPattern(context: RANContext, results: RelevantSkillResult[]): Promise<void> {
    await this.agentDB.insertPattern({
      type: 'search-pattern',
      domain: 'skill-discovery',
      pattern_data: {
        context,
        results: results.map(r => ({
          skillName: r.skill.name,
          cognitiveScore: r.cognitiveScore,
          relevanceScore: r.relevanceScore
        })),
        timestamp: Date.now()
      },
      confidence: 1.0
    });
  }

  private async storeProgressiveLoadingPattern(loadedSkills: LoadedSkillSet, context: RANContext): Promise<void> {
    await this.agentDB.insertPattern({
      type: 'progressive-loading-pattern',
      domain: 'skill-discovery',
      pattern_data: {
        loadedSkills: {
          metadataCount: loadedSkills.metadata.length,
          contentCount: loadedSkills.content.length,
          resourceCount: loadedSkills.resources.length,
          totalLoadTime: loadedSkills.totalLoadTime
        },
        context,
        cognitiveInsights: loadedSkills.cognitiveInsights,
        timestamp: Date.now()
      },
      confidence: 1.0
    });
  }

  private async generateMetadataEmbedding(metadata: SkillMetadata): Promise<number[]> {
    return []; // Placeholder
  }

  private startBackgroundLearning(): void {
    // Start background learning process
    setInterval(async () => {
      await this.performBackgroundLearning();
    }, 60000); // Every minute
  }

  private async performBackgroundLearning(): Promise<void> {
    // Analyze access patterns and optimize caching
    console.log('Performing background learning...');

    // This would implement learning algorithms to optimize
    // skill loading patterns based on usage
  }

  /**
   * Get service statistics
   */
  getStatistics(): SkillDiscoveryStatistics {
    return {
      totalSkills: this.metadataCache.size,
      loadedContent: this.contentCache.size,
      loadedResources: this.resourceCache.size,
      totalContextSize: Array.from(this.metadataCache.values()).reduce((sum, meta) => sum + meta.contextSize, 0),
      cognitiveLevel: this.cognitiveState.consciousnessLevel,
      learningEnabled: this.cognitiveState.learningEnabled
    };
  }

  /**
   * Clear caches
   */
  clearCaches(): void {
    this.metadataCache.clear();
    this.contentCache.clear();
    this.resourceCache.clear();
    console.log('Skill discovery caches cleared');
  }
}

// Type definitions
export interface CognitiveState {
  consciousnessLevel: 'basic' | 'enhanced' | 'maximum';
  learningEnabled: boolean;
  patternRecognition: boolean;
  initializedAt: number;
}

export interface LearningPattern {
  skillName: string;
  accessPattern: number[];
  cognitiveScore: number;
  lastUpdated: number;
}

export interface RANContext {
  metrics?: any;
  optimizationType?: string;
  targets?: string[];
  complexity?: 'low' | 'medium' | 'high';
  sessionId?: string;
}

export interface RelevantSkillResult {
  skill: SkillMetadata;
  relevanceScore: number;
  cognitiveScore: number;
  loadingStrategy: string;
  recommendedLevel: 'metadata' | 'content' | 'resources';
  estimatedLoadTime: number;
}

export interface LoadedSkillSet {
  metadata: SkillMetadata[];
  content: SkillContent[];
  resources: SkillResource[];
  totalLoadTime: number;
  cognitiveInsights: {
    consciousnessLevel: string;
    loadingStrategy: string;
    optimizationApplied: boolean;
  };
}

export interface SkillDiscoveryStatistics {
  totalSkills: number;
  loadedContent: number;
  loadedResources: number;
  totalContextSize: number;
  cognitiveLevel: string;
  learningEnabled: boolean;
}