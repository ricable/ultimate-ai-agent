#!/usr/bin/env node
/**
 * Agentic-Flow AST Integration for agentic-jujutsu
 * Enables AI agents to understand and manipulate jj operations
 */

const jj = require('../pkg/node');

/**
 * AST Node Types for Jujutsu Operations
 */
const ASTNodeTypes = {
  OPERATION: 'Operation',
  COMMIT: 'Commit',
  BRANCH: 'Branch',
  CONFLICT: 'Conflict',
  REVISION: 'Revision',
};

/**
 * Parse jj operation into AST
 */
function parseOperationAST(operation) {
  return {
    type: ASTNodeTypes.OPERATION,
    id: operation.id || generateId(),
    command: operation.command,
    timestamp: operation.timestamp || new Date().toISOString(),
    user: operation.user || 'unknown',
    children: [],
    metadata: {
      success: operation.success !== false,
      duration: operation.duration || 0,
    },
  };
}

/**
 * Parse commit into AST
 */
function parseCommitAST(commit) {
  return {
    type: ASTNodeTypes.COMMIT,
    id: commit.change_id || commit.commit_id,
    message: commit.description || '',
    author: commit.author || {},
    timestamp: commit.timestamp || new Date().toISOString(),
    parents: commit.parents || [],
    metadata: {
      is_working_copy: commit.is_working_copy || false,
      branches: commit.branches || [],
    },
  };
}

/**
 * Transform AST for AI agent consumption
 */
function transformASTForAgent(ast) {
  return {
    ...ast,
    __ai_metadata: {
      complexity: calculateComplexity(ast),
      suggestedActions: getSuggestedActions(ast),
      riskLevel: assessRiskLevel(ast),
    },
  };
}

function calculateComplexity(ast) {
  if (ast.type === ASTNodeTypes.CONFLICT) return 'high';
  if (ast.type === ASTNodeTypes.OPERATION && ast.children.length > 5) return 'medium';
  return 'low';
}

function getSuggestedActions(ast) {
  const actions = [];
  
  if (ast.type === ASTNodeTypes.CONFLICT) {
    actions.push('resolve_conflict', 'abandon', 'squash');
  }
  
  if (ast.type === ASTNodeTypes.COMMIT && !ast.metadata.is_working_copy) {
    actions.push('describe', 'edit', 'abandon');
  }
  
  return actions;
}

function assessRiskLevel(ast) {
  if (ast.type === ASTNodeTypes.OPERATION) {
    const destructiveOps = ['abandon', 'squash', 'rebase'];
    if (destructiveOps.some(op => ast.command.includes(op))) {
      return 'high';
    }
  }
  return 'low';
}

/**
 * Generate unique ID
 */
function generateId() {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Agent-Flow Integration API
 */
const AgenticFlowIntegration = {
  /**
   * Convert jj operation to agent-consumable format
   */
  operationToAgent: (operation) => {
    const ast = parseOperationAST(operation);
    return transformASTForAgent(ast);
  },

  /**
   * Convert jj commit to agent-consumable format
   */
  commitToAgent: (commit) => {
    const ast = parseCommitAST(commit);
    return transformASTForAgent(ast);
  },

  /**
   * Batch process operations
   */
  batchProcess: (operations) => {
    return operations.map(op => AgenticFlowIntegration.operationToAgent(op));
  },

  /**
   * Get agent recommendations
   */
  getRecommendations: (ast) => {
    const recommendations = [];
    
    if (ast.__ai_metadata.complexity === 'high') {
      recommendations.push({
        type: 'warning',
        message: 'Complex operation detected - consider breaking into smaller steps',
      });
    }
    
    if (ast.__ai_metadata.riskLevel === 'high') {
      recommendations.push({
        type: 'caution',
        message: 'Potentially destructive operation - ensure backup exists',
      });
    }
    
    return recommendations;
  },
};

module.exports = AgenticFlowIntegration;

// CLI Demo
if (require.main === module) {
  console.log('🤖 Agentic-Flow Integration Demo\n');
  
  const sampleOperation = {
    id: 'op-001',
    command: 'jj new -m "Add feature"',
    user: 'agent-001',
    success: true,
  };
  
  const agentData = AgenticFlowIntegration.operationToAgent(sampleOperation);
  console.log('Agent-consumable data:');
  console.log(JSON.stringify(agentData, null, 2));
  
  console.log('\nRecommendations:');
  const recs = AgenticFlowIntegration.getRecommendations(agentData);
  recs.forEach(rec => console.log(`  [${rec.type}] ${rec.message}`));
}
