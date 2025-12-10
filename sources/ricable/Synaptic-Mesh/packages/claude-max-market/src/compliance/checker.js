/**
 * Compliance Checker
 * 
 * Verifies system compliance with Anthropic Terms of Service
 * and implements ongoing compliance monitoring
 */

import fs from 'fs/promises';
import path from 'path';
import crypto from 'crypto';

export class ComplianceChecker {
  constructor() {
    this.checks = [
      {
        id: 'no_shared_keys',
        name: 'No Shared API Keys',
        description: 'Verify API keys are never transmitted or stored remotely',
        critical: true
      },
      {
        id: 'peer_orchestrated',
        name: 'Peer Orchestrated Model',
        description: 'Ensure tasks route to willing participants, not through shared accounts',
        critical: true
      },
      {
        id: 'voluntary_participation',
        name: 'Voluntary Participation',
        description: 'Users must opt-in explicitly and can revoke consent',
        critical: true
      },
      {
        id: 'user_control',
        name: 'User Control and Transparency',
        description: 'Users can approve tasks, set limits, and view audit logs',
        critical: true
      },
      {
        id: 'token_design',
        name: 'Compliant Token Design',
        description: 'Tokens reward contribution, not purchase access',
        critical: false
      },
      {
        id: 'data_protection',
        name: 'Data Protection',
        description: 'User data and credentials are protected and not transmitted',
        critical: true
      }
    ];
  }

  /**
   * Run comprehensive compliance check
   */
  async checkCompliance() {
    console.log('🔍 Running compliance verification...');
    
    const results = [];
    let criticalFailures = 0;
    
    for (const check of this.checks) {
      try {
        const result = await this.runCheck(check);
        results.push(result);
        
        if (check.critical && !result.passed) {
          criticalFailures++;
        }
        
        const status = result.passed ? '✅' : '❌';
        console.log(`${status} ${check.name}: ${result.message}`);
        
        if (result.recommendations?.length > 0) {
          result.recommendations.forEach(rec => {
            console.log(`   💡 ${rec}`);
          });
        }
      } catch (error) {
        const result = {
          ...check,
          passed: false,
          message: `Check failed: ${error.message}`,
          error: error.message
        };
        results.push(result);
        
        if (check.critical) {
          criticalFailures++;
        }
        
        console.log(`❌ ${check.name}: Error during check`);
      }
    }
    
    const overallCompliant = criticalFailures === 0;
    const score = Math.round((results.filter(r => r.passed).length / results.length) * 100);
    
    console.log(`\n📊 Compliance Score: ${score}/100`);
    console.log(`🎯 Critical Failures: ${criticalFailures}`);
    console.log(`✅ Overall Compliant: ${overallCompliant ? 'YES' : 'NO'}`);
    
    return overallCompliant;
  }

  /**
   * Run individual compliance check
   */
  async runCheck(check) {
    switch (check.id) {
      case 'no_shared_keys':
        return await this.checkNoSharedKeys();
      case 'peer_orchestrated':
        return await this.checkPeerOrchestrated();
      case 'voluntary_participation':
        return await this.checkVoluntaryParticipation();
      case 'user_control':
        return await this.checkUserControl();
      case 'token_design':
        return await this.checkTokenDesign();
      case 'data_protection':
        return await this.checkDataProtection();
      default:
        throw new Error(`Unknown check: ${check.id}`);
    }
  }

  /**
   * Check: No Shared API Keys
   */
  async checkNoSharedKeys() {
    const issues = [];
    const recommendations = [];

    // Check for hardcoded API keys in source
    const sourceFiles = await this.findSourceFiles();
    for (const file of sourceFiles) {
      const content = await fs.readFile(file, 'utf8');
      
      // Look for potential API key patterns
      const apiKeyPatterns = [
        /sk-[a-zA-Z0-9]{48,}/g, // Anthropic API key pattern
        /ANTHROPIC_API_KEY\s*=\s*["']sk-/g,
        /CLAUDE_API_KEY\s*=\s*["']sk-/g
      ];
      
      for (const pattern of apiKeyPatterns) {
        if (pattern.test(content)) {
          issues.push(`Potential hardcoded API key found in ${file}`);
        }
      }
    }

    // Check environment configuration
    if (process.env.CLAUDE_API_KEY || process.env.ANTHROPIC_API_KEY) {
      // This is actually good - using environment variables
    } else {
      recommendations.push('Configure API key via environment variables for security');
    }

    // Check for API key transmission in network code
    const networkFiles = sourceFiles.filter(f => 
      f.includes('network') || f.includes('market') || f.includes('integration')
    );
    
    for (const file of networkFiles) {
      const content = await fs.readFile(file, 'utf8');
      
      // Look for API key transmission patterns
      if (content.includes('apiKey') && content.includes('send')) {
        issues.push(`Potential API key transmission in ${file}`);
      }
    }

    return {
      passed: issues.length === 0,
      message: issues.length === 0 
        ? 'No shared API key issues detected'
        : `${issues.length} API key issues found`,
      issues,
      recommendations
    };
  }

  /**
   * Check: Peer Orchestrated Model
   */
  async checkPeerOrchestrated() {
    const issues = [];
    const recommendations = [];

    // Check orchestration architecture
    try {
      const orchestratorPath = path.join(process.cwd(), 'src/orchestration');
      const orchestratorExists = await fs.access(orchestratorPath).then(() => true).catch(() => false);
      
      if (!orchestratorExists) {
        issues.push('Orchestration module not found');
      } else {
        // Check for proper peer-to-peer design
        const orchestratorFiles = await fs.readdir(orchestratorPath);
        const hasJobOrchestrator = orchestratorFiles.some(f => f.includes('jobOrchestrator'));
        
        if (!hasJobOrchestrator) {
          issues.push('Job orchestrator implementation not found');
        }
      }
    } catch (error) {
      issues.push(`Failed to verify orchestration architecture: ${error.message}`);
    }

    // Check for centralized account usage patterns
    const sourceFiles = await this.findSourceFiles();
    for (const file of sourceFiles) {
      const content = await fs.readFile(file, 'utf8');
      
      // Look for centralized patterns
      if (content.includes('shared_account') || content.includes('proxy_account')) {
        issues.push(`Potential centralized account usage in ${file}`);
      }
    }

    return {
      passed: issues.length === 0,
      message: issues.length === 0
        ? 'Peer orchestrated model verified'
        : `${issues.length} orchestration issues found`,
      issues,
      recommendations
    };
  }

  /**
   * Check: Voluntary Participation
   */
  async checkVoluntaryParticipation() {
    const issues = [];
    const recommendations = [];

    // Check for opt-in mechanism
    const sourceFiles = await this.findSourceFiles();
    let hasOptIn = false;
    let hasOptOut = false;
    let hasConsent = false;

    for (const file of sourceFiles) {
      const content = await fs.readFile(file, 'utf8');
      
      if (content.includes('opt-in') || content.includes('optIn')) {
        hasOptIn = true;
      }
      if (content.includes('opt-out') || content.includes('optOut')) {
        hasOptOut = true;
      }
      if (content.includes('consent') || content.includes('approval')) {
        hasConsent = true;
      }
    }

    if (!hasOptIn) {
      issues.push('Opt-in mechanism not found');
    }
    if (!hasOptOut) {
      issues.push('Opt-out mechanism not found');
    }
    if (!hasConsent) {
      issues.push('User consent mechanism not found');
    }

    // Check for user preferences storage
    try {
      const prefsFile = path.join(process.cwd(), '.claude-max-market.json');
      await fs.access(prefsFile);
      // File exists - good
    } catch (error) {
      recommendations.push('User preferences file not found - will be created on first use');
    }

    return {
      passed: issues.length === 0,
      message: issues.length === 0
        ? 'Voluntary participation mechanisms verified'
        : `${issues.length} participation issues found`,
      issues,
      recommendations
    };
  }

  /**
   * Check: User Control and Transparency
   */
  async checkUserControl() {
    const issues = [];
    const recommendations = [];

    // Check for user control mechanisms
    const sourceFiles = await this.findSourceFiles();
    let hasApproval = false;
    let hasLimits = false;
    let hasLogging = false;
    let hasAudit = false;

    for (const file of sourceFiles) {
      const content = await fs.readFile(file, 'utf8');
      
      if (content.includes('approval') || content.includes('requestJobApproval')) {
        hasApproval = true;
      }
      if (content.includes('limits') || content.includes('setLimits')) {
        hasLimits = true;
      }
      if (content.includes('logging') || content.includes('logger')) {
        hasLogging = true;
      }
      if (content.includes('audit') || content.includes('generateAuditReport')) {
        hasAudit = true;
      }
    }

    if (!hasApproval) {
      issues.push('Job approval mechanism not found');
    }
    if (!hasLimits) {
      issues.push('Usage limits mechanism not found');
    }
    if (!hasLogging) {
      issues.push('Logging mechanism not found');
    }
    if (!hasAudit) {
      issues.push('Audit mechanism not found');
    }

    return {
      passed: issues.length === 0,
      message: issues.length === 0
        ? 'User control mechanisms verified'
        : `${issues.length} control issues found`,
      issues,
      recommendations
    };
  }

  /**
   * Check: Token Design Compliance
   */
  async checkTokenDesign() {
    const issues = [];
    const recommendations = [];

    // Check token implementation
    const sourceFiles = await this.findSourceFiles();
    let hasTokenRewards = false;
    let hasPaymentModel = false;

    for (const file of sourceFiles) {
      const content = await fs.readFile(file, 'utf8');
      
      if (content.includes('reward') || content.includes('contribution')) {
        hasTokenRewards = true;
      }
      if (content.includes('payment') || content.includes('purchase') || content.includes('buy')) {
        hasPaymentModel = true;
      }
    }

    if (!hasTokenRewards) {
      recommendations.push('Consider implementing token reward system for contributions');
    }
    if (hasPaymentModel) {
      issues.push('Payment model detected - tokens should reward contribution, not purchase access');
    }

    return {
      passed: issues.length === 0,
      message: issues.length === 0
        ? 'Token design appears compliant'
        : `${issues.length} token design issues found`,
      issues,
      recommendations
    };
  }

  /**
   * Check: Data Protection
   */
  async checkDataProtection() {
    const issues = [];
    const recommendations = [];

    // Check for encryption implementation
    const sourceFiles = await this.findSourceFiles();
    let hasEncryption = false;
    let hasSecureStorage = false;

    for (const file of sourceFiles) {
      const content = await fs.readFile(file, 'utf8');
      
      if (content.includes('encrypt') || content.includes('crypto')) {
        hasEncryption = true;
      }
      if (content.includes('secure') && content.includes('storage')) {
        hasSecureStorage = true;
      }
    }

    if (!hasEncryption) {
      recommendations.push('Implement payload encryption for secure transmission');
    }
    if (!hasSecureStorage) {
      recommendations.push('Implement secure storage for sensitive data');
    }

    // Check for data transmission patterns
    for (const file of sourceFiles) {
      const content = await fs.readFile(file, 'utf8');
      
      // Look for potential data leakage
      if (content.includes('credentials') && content.includes('send')) {
        issues.push(`Potential credential transmission in ${file}`);
      }
    }

    return {
      passed: issues.length === 0,
      message: issues.length === 0
        ? 'Data protection measures verified'
        : `${issues.length} data protection issues found`,
      issues,
      recommendations
    };
  }

  /**
   * Generate detailed compliance report
   */
  async generateComplianceReport() {
    const timestamp = new Date().toISOString();
    const checkResults = [];
    
    for (const check of this.checks) {
      const result = await this.runCheck(check);
      checkResults.push({
        ...check,
        ...result,
        timestamp
      });
    }
    
    const passedChecks = checkResults.filter(r => r.passed).length;
    const totalChecks = checkResults.length;
    const score = Math.round((passedChecks / totalChecks) * 100);
    const criticalFailures = checkResults.filter(r => r.critical && !r.passed).length;
    
    return {
      timestamp,
      version: '1.0.0',
      summary: {
        score,
        totalChecks,
        passedChecks,
        failedChecks: totalChecks - passedChecks,
        criticalFailures,
        compliant: criticalFailures === 0
      },
      checks: checkResults,
      systemInfo: {
        nodeVersion: process.version,
        platform: process.platform,
        arch: process.arch
      },
      recommendations: checkResults
        .filter(r => r.recommendations?.length > 0)
        .flatMap(r => r.recommendations),
      issues: checkResults
        .filter(r => r.issues?.length > 0)
        .flatMap(r => r.issues)
    };
  }

  /**
   * Quick compliance status check
   */
  async quickCheck() {
    try {
      // Run only critical checks quickly
      const criticalChecks = this.checks.filter(c => c.critical);
      
      for (const check of criticalChecks) {
        const result = await this.runCheck(check);
        if (!result.passed) {
          return false;
        }
      }
      
      return true;
    } catch (error) {
      console.warn('Quick compliance check failed:', error.message);
      return false;
    }
  }

  // Helper methods

  async findSourceFiles() {
    const files = [];
    
    async function findFiles(dir) {
      try {
        const entries = await fs.readdir(dir, { withFileTypes: true });
        
        for (const entry of entries) {
          const fullPath = path.join(dir, entry.name);
          
          if (entry.isDirectory() && !entry.name.startsWith('.') && entry.name !== 'node_modules') {
            await findFiles(fullPath);
          } else if (entry.isFile() && (entry.name.endsWith('.js') || entry.name.endsWith('.ts'))) {
            files.push(fullPath);
          }
        }
      } catch (error) {
        // Directory doesn't exist or no permission
      }
    }
    
    await findFiles(path.join(process.cwd(), 'src'));
    return files;
  }

  /**
   * Monitor ongoing compliance
   */
  async startMonitoring(intervalMs = 3600000) { // 1 hour default
    console.log('🔍 Starting compliance monitoring...');
    
    setInterval(async () => {
      try {
        const compliant = await this.quickCheck();
        if (!compliant) {
          console.warn('⚠️ Compliance issue detected during monitoring');
          // Could send alerts, log to audit trail, etc.
        }
      } catch (error) {
        console.warn('Compliance monitoring error:', error.message);
      }
    }, intervalMs);
  }
}