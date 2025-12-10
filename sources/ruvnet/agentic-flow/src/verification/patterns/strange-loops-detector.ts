/**
 * Strange Loops Pattern Detection
 * Identify circular reasoning, detect logical contradictions, validate causal chains
 */

export interface LogicalPattern {
  type: 'circular' | 'contradiction' | 'invalid-causal' | 'self-reference';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  location: PatternLocation[];
  chain: string[];
  suggestion: string;
}

export interface PatternLocation {
  start: number;
  end: number;
  context: string;
}

export interface CausalChain {
  nodes: CausalNode[];
  edges: CausalEdge[];
  valid: boolean;
  cycles: CausalNode[][];
  contradictions: ChainContradiction[];
}

export interface CausalNode {
  id: string;
  claim: string;
  evidence: string[];
  confidence: number;
}

export interface CausalEdge {
  from: string;
  to: string;
  type: 'causes' | 'enables' | 'prevents' | 'correlates';
  strength: number;
}

export interface ChainContradiction {
  node1: string;
  node2: string;
  reason: string;
  severity: 'low' | 'medium' | 'high';
}

export interface RecursivePattern {
  depth: number;
  pattern: string;
  instances: string[];
  problematic: boolean;
  reason?: string;
}

export class StrangeLoopsDetector {
  private readonly MAX_RECURSION_DEPTH = 5;
  private readonly CONTRADICTION_KEYWORDS = [
    ['always', 'never'],
    ['all', 'none'],
    ['increase', 'decrease'],
    ['cause', 'prevent'],
    ['effective', 'ineffective'],
  ];

  /**
   * Detect circular reasoning patterns
   */
  async detectCircularReasoning(
    text: string,
    context?: Record<string, any>
  ): Promise<LogicalPattern[]> {
    const patterns: LogicalPattern[] = [];

    // Extract claims and their justifications
    const claims = this.extractClaims(text);

    // Build reasoning graph
    const reasoningGraph = this.buildReasoningGraph(claims);

    // Detect cycles in reasoning
    const cycles = this.findCycles(reasoningGraph);

    for (const cycle of cycles) {
      patterns.push({
        type: 'circular',
        severity: this.assessCycleSeverity(cycle),
        description: 'Circular reasoning detected: conclusion depends on itself',
        location: this.locateCycle(text, cycle),
        chain: cycle.map(node => node.claim),
        suggestion: 'Break circular logic by introducing independent evidence',
      });
    }

    return patterns;
  }

  /**
   * Extract claims from text
   */
  private extractClaims(text: string): CausalNode[] {
    const claims: CausalNode[] = [];

    // Split into sentences
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);

    sentences.forEach((sentence, index) => {
      // Identify causal language
      const hasCausalLanguage = /\b(because|therefore|thus|hence|consequently|as a result)\b/i.test(sentence);

      if (hasCausalLanguage || sentence.length > 20) {
        claims.push({
          id: `claim_${index}`,
          claim: sentence.trim(),
          evidence: [],
          confidence: 0.5,
        });
      }
    });

    return claims;
  }

  /**
   * Build reasoning graph from claims
   */
  private buildReasoningGraph(claims: CausalNode[]): {
    nodes: CausalNode[];
    edges: CausalEdge[];
  } {
    const edges: CausalEdge[] = [];

    // Identify relationships between claims
    for (let i = 0; i < claims.length; i++) {
      for (let j = i + 1; j < claims.length; j++) {
        const relationship = this.detectRelationship(claims[i], claims[j]);

        if (relationship) {
          edges.push({
            from: claims[i].id,
            to: claims[j].id,
            type: relationship.type,
            strength: relationship.strength,
          });
        }
      }
    }

    return { nodes: claims, edges };
  }

  /**
   * Detect relationship between two claims
   */
  private detectRelationship(
    claim1: CausalNode,
    claim2: CausalNode
  ): { type: CausalEdge['type']; strength: number } | null {
    const text1 = claim1.claim.toLowerCase();
    const text2 = claim2.claim.toLowerCase();

    // Check for shared keywords (potential relationship)
    const words1 = new Set(text1.split(/\s+/).filter(w => w.length > 4));
    const words2 = new Set(text2.split(/\s+/).filter(w => w.length > 4));
    const overlap = [...words1].filter(w => words2.has(w));

    if (overlap.length < 2) return null;

    // Determine relationship type
    let type: CausalEdge['type'] = 'correlates';
    if (/\b(cause|lead to|result in)\b/i.test(text1) || /\b(cause|lead to|result in)\b/i.test(text2)) {
      type = 'causes';
    } else if (/\b(enable|allow|permit)\b/i.test(text1) || /\b(enable|allow|permit)\b/i.test(text2)) {
      type = 'enables';
    } else if (/\b(prevent|block|inhibit)\b/i.test(text1) || /\b(prevent|block|inhibit)\b/i.test(text2)) {
      type = 'prevents';
    }

    const strength = overlap.length / Math.max(words1.size, words2.size);

    return { type, strength };
  }

  /**
   * Find cycles in reasoning graph using DFS
   */
  private findCycles(graph: {
    nodes: CausalNode[];
    edges: CausalEdge[];
  }): CausalNode[][] {
    const cycles: CausalNode[][] = [];
    const visited = new Set<string>();
    const recursionStack = new Set<string>();
    const currentPath: CausalNode[] = [];

    const dfs = (nodeId: string): void => {
      visited.add(nodeId);
      recursionStack.add(nodeId);

      const node = graph.nodes.find(n => n.id === nodeId);
      if (node) currentPath.push(node);

      // Find outgoing edges
      const outgoing = graph.edges.filter(e => e.from === nodeId);

      for (const edge of outgoing) {
        if (!visited.has(edge.to)) {
          dfs(edge.to);
        } else if (recursionStack.has(edge.to)) {
          // Cycle detected - extract cycle from current path
          const cycleStart = currentPath.findIndex(n => n.id === edge.to);
          if (cycleStart >= 0) {
            cycles.push([...currentPath.slice(cycleStart)]);
          }
        }
      }

      recursionStack.delete(nodeId);
      currentPath.pop();
    };

    for (const node of graph.nodes) {
      if (!visited.has(node.id)) {
        dfs(node.id);
      }
    }

    return cycles;
  }

  /**
   * Assess cycle severity
   */
  private assessCycleSeverity(cycle: CausalNode[]): 'low' | 'medium' | 'high' | 'critical' {
    // Severity based on cycle length and claim confidence
    if (cycle.length <= 2) return 'critical'; // Direct self-reference
    if (cycle.length === 3) return 'high';

    const avgConfidence = cycle.reduce((sum, node) => sum + node.confidence, 0) / cycle.length;
    if (avgConfidence > 0.7) return 'medium';

    return 'low';
  }

  /**
   * Locate cycle in text
   */
  private locateCycle(text: string, cycle: CausalNode[]): PatternLocation[] {
    const locations: PatternLocation[] = [];

    for (const node of cycle) {
      const index = text.indexOf(node.claim);
      if (index >= 0) {
        locations.push({
          start: index,
          end: index + node.claim.length,
          context: node.claim,
        });
      }
    }

    return locations;
  }

  /**
   * Detect logical contradictions
   */
  async detectContradictions(
    text: string,
    context?: Record<string, any>
  ): Promise<LogicalPattern[]> {
    const patterns: LogicalPattern[] = [];

    // Extract claims
    const claims = this.extractClaims(text);

    // Check for keyword contradictions
    for (let i = 0; i < claims.length; i++) {
      for (let j = i + 1; j < claims.length; j++) {
        const contradiction = this.checkContradiction(claims[i], claims[j]);

        if (contradiction) {
          patterns.push({
            type: 'contradiction',
            severity: contradiction.severity,
            description: contradiction.reason,
            location: [
              ...this.locateCycle(text, [claims[i]]),
              ...this.locateCycle(text, [claims[j]]),
            ],
            chain: [claims[i].claim, claims[j].claim],
            suggestion: 'Resolve contradictory statements with additional evidence',
          });
        }
      }
    }

    return patterns;
  }

  /**
   * Check for contradiction between two claims
   */
  private checkContradiction(
    claim1: CausalNode,
    claim2: CausalNode
  ): ChainContradiction | null {
    const text1 = claim1.claim.toLowerCase();
    const text2 = claim2.claim.toLowerCase();

    // Check for keyword-based contradictions
    for (const [word1, word2] of this.CONTRADICTION_KEYWORDS) {
      if (text1.includes(word1) && text2.includes(word2)) {
        return {
          node1: claim1.id,
          node2: claim2.id,
          reason: `Contradictory terms detected: "${word1}" vs "${word2}"`,
          severity: 'high',
        };
      }
      if (text1.includes(word2) && text2.includes(word1)) {
        return {
          node1: claim1.id,
          node2: claim2.id,
          reason: `Contradictory terms detected: "${word2}" vs "${word1}"`,
          severity: 'high',
        };
      }
    }

    // Check for negation contradictions
    const hasNegation1 = /\b(not|no|never|none)\b/i.test(text1);
    const hasNegation2 = /\b(not|no|never|none)\b/i.test(text2);

    if (hasNegation1 !== hasNegation2) {
      // Check if they're talking about the same thing
      const words1 = new Set(text1.split(/\s+/).filter(w => w.length > 4));
      const words2 = new Set(text2.split(/\s+/).filter(w => w.length > 4));
      const overlap = [...words1].filter(w => words2.has(w));

      if (overlap.length >= 2) {
        return {
          node1: claim1.id,
          node2: claim2.id,
          reason: 'Affirmation vs negation of similar claim',
          severity: 'medium',
        };
      }
    }

    return null;
  }

  /**
   * Validate causal chains
   */
  async validateCausalChain(
    chain: CausalChain,
    context?: Record<string, any>
  ): Promise<{
    valid: boolean;
    issues: LogicalPattern[];
    strength: number;
  }> {
    const issues: LogicalPattern[] = [];

    // Check for cycles
    if (chain.cycles.length > 0) {
      for (const cycle of chain.cycles) {
        issues.push({
          type: 'circular',
          severity: 'critical',
          description: 'Causal loop detected in chain',
          location: [],
          chain: cycle.map(n => n.claim),
          suggestion: 'Remove circular causation',
        });
      }
    }

    // Check for contradictions
    for (const contradiction of chain.contradictions) {
      issues.push({
        type: 'contradiction',
        severity: contradiction.severity,
        description: contradiction.reason,
        location: [],
        chain: [
          chain.nodes.find(n => n.id === contradiction.node1)?.claim || '',
          chain.nodes.find(n => n.id === contradiction.node2)?.claim || '',
        ],
        suggestion: 'Resolve contradictory causal claims',
      });
    }

    // Check for weak links
    const weakEdges = chain.edges.filter(e => e.strength < 0.3);
    if (weakEdges.length > 0) {
      issues.push({
        type: 'invalid-causal',
        severity: 'medium',
        description: `${weakEdges.length} weak causal links detected`,
        location: [],
        chain: weakEdges.map(e => `${e.from} → ${e.to}`),
        suggestion: 'Strengthen causal evidence for weak links',
      });
    }

    // Calculate overall chain strength
    const avgEdgeStrength = chain.edges.length > 0
      ? chain.edges.reduce((sum, e) => sum + e.strength, 0) / chain.edges.length
      : 0;

    const valid = chain.cycles.length === 0 &&
                  chain.contradictions.length === 0 &&
                  avgEdgeStrength >= 0.5;

    return {
      valid,
      issues,
      strength: avgEdgeStrength,
    };
  }

  /**
   * Detect recursive patterns
   */
  async detectRecursivePatterns(
    text: string,
    maxDepth: number = this.MAX_RECURSION_DEPTH
  ): Promise<RecursivePattern[]> {
    const patterns: RecursivePattern[] = [];

    // Look for self-referential statements
    const selfReferences = this.findSelfReferences(text);

    for (const ref of selfReferences) {
      patterns.push({
        depth: ref.depth,
        pattern: 'self-reference',
        instances: [ref.text],
        problematic: ref.depth > maxDepth,
        reason: ref.depth > maxDepth
          ? 'Recursion depth exceeds safe limit'
          : undefined,
      });
    }

    // Look for nested definitions
    const nestedPatterns = this.findNestedDefinitions(text);
    patterns.push(...nestedPatterns);

    return patterns;
  }

  /**
   * Find self-referential statements
   */
  private findSelfReferences(text: string): Array<{
    text: string;
    depth: number;
  }> {
    const references: Array<{ text: string; depth: number }> = [];
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);

    for (const sentence of sentences) {
      // Check for self-reference patterns
      if (/\b(itself|themselves|recursive|circular)\b/i.test(sentence)) {
        references.push({
          text: sentence.trim(),
          depth: 1, // Simple detection - could be enhanced
        });
      }

      // Check for definitions that reference themselves
      const defineMatch = sentence.match(/\b(\w+)\s+(?:is|are|means?)\b.*\b\1\b/i);
      if (defineMatch) {
        references.push({
          text: sentence.trim(),
          depth: 1,
        });
      }
    }

    return references;
  }

  /**
   * Find nested definition patterns
   */
  private findNestedDefinitions(text: string): RecursivePattern[] {
    const patterns: RecursivePattern[] = [];

    // Look for nested parenthetical explanations
    const nestedParens = text.match(/\([^()]*\([^()]*\)[^()]*\)/g);

    if (nestedParens && nestedParens.length > 0) {
      patterns.push({
        depth: 2,
        pattern: 'nested-parenthetical',
        instances: nestedParens,
        problematic: false,
        reason: undefined,
      });
    }

    return patterns;
  }

  /**
   * Get pattern statistics
   */
  getStatistics(patterns: LogicalPattern[]): {
    total: number;
    byType: Record<string, number>;
    bySeverity: Record<string, number>;
    mostCommon: string;
  } {
    const byType: Record<string, number> = {};
    const bySeverity: Record<string, number> = {};

    for (const pattern of patterns) {
      byType[pattern.type] = (byType[pattern.type] || 0) + 1;
      bySeverity[pattern.severity] = (bySeverity[pattern.severity] || 0) + 1;
    }

    const mostCommon = Object.entries(byType)
      .sort((a, b) => b[1] - a[1])[0]?.[0] || 'none';

    return {
      total: patterns.length,
      byType,
      bySeverity,
      mostCommon,
    };
  }
}
