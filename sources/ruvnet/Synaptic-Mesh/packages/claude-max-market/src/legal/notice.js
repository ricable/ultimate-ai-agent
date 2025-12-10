/**
 * Legal Notice and Terms Display
 * 
 * Implements compliance requirements for Claude Max capacity sharing
 * as per Anthropic Terms of Service
 */

import chalk from 'chalk';

export class LegalNotice {
  constructor() {
    this.version = '1.0.0';
    this.lastUpdated = '2024-01-15';
  }

  /**
   * Display comprehensive legal terms and usage policy
   */
  async displayTerms() {
    console.clear();
    this.printHeader();
    this.printLegalNotice();
    this.printUsagePolicy();
    this.printCompliancePrinciples();
    this.printUserRights();
    this.printContact();
    this.printFooter();
  }

  /**
   * Get terms text for programmatic use
   */
  getTermsText() {
    return {
      legalNotice: this.getLegalNoticeText(),
      usagePolicy: this.getUsagePolicyText(),
      compliancePrinciples: this.getCompliancePrinciplesText(),
      userRights: this.getUserRightsText()
    };
  }

  /**
   * Validate user acceptance of terms
   */
  validateAcceptance(userResponse) {
    const acceptedTerms = [
      'i accept',
      'i agree',
      'yes',
      'accept',
      'agree',
      'understood'
    ];
    
    return acceptedTerms.some(term => 
      userResponse.toLowerCase().trim().includes(term)
    );
  }

  // Private methods

  printHeader() {
    console.log(chalk.blue.bold('═'.repeat(80)));
    console.log(chalk.blue.bold('               SYNAPTIC NEURAL MESH - LEGAL NOTICE'));
    console.log(chalk.blue.bold('                  Claude Max Capacity Sharing'));
    console.log(chalk.blue.bold('═'.repeat(80)));
    console.log(chalk.gray(`Version ${this.version} | Last Updated: ${this.lastUpdated}`));
    console.log();
  }

  printLegalNotice() {
    console.log(chalk.yellow.bold('📋 LEGAL NOTICE'));
    console.log(chalk.yellow('─'.repeat(40)));
    console.log(chalk.white(this.getLegalNoticeText()));
    console.log();
  }

  printUsagePolicy() {
    console.log(chalk.green.bold('📖 USAGE POLICY'));
    console.log(chalk.green('─'.repeat(40)));
    console.log(chalk.white(this.getUsagePolicyText()));
    console.log();
  }

  printCompliancePrinciples() {
    console.log(chalk.cyan.bold('⚖️  COMPLIANCE PRINCIPLES'));
    console.log(chalk.cyan('─'.repeat(40)));
    console.log(chalk.white(this.getCompliancePrinciplesText()));
    console.log();
  }

  printUserRights() {
    console.log(chalk.magenta.bold('🛡️  YOUR RIGHTS AND CONTROLS'));
    console.log(chalk.magenta('─'.repeat(40)));
    console.log(chalk.white(this.getUserRightsText()));
    console.log();
  }

  printContact() {
    console.log(chalk.blue.bold('📞 CONTACT INFORMATION'));
    console.log(chalk.blue('─'.repeat(40)));
    console.log(chalk.white(`
For questions about these terms or compliance issues:
• GitHub Issues: https://github.com/ruvnet/synaptic-neural-mesh/issues
• Email: legal@synaptic-neural-mesh.org
• Documentation: https://synaptic-neural-mesh.org/legal
    `));
  }

  printFooter() {
    console.log(chalk.blue.bold('═'.repeat(80)));
    console.log(chalk.gray('By using this software, you acknowledge that you have read and'));
    console.log(chalk.gray('understood these terms and agree to comply with all requirements.'));
    console.log(chalk.blue.bold('═'.repeat(80)));
    console.log();
  }

  getLegalNoticeText() {
    return `
IMPORTANT: Synaptic Neural Mesh does NOT proxy, resell, or provide 
shared access to Claude Max or any Anthropic services.

This software facilitates VOLUNTARY PEER-TO-PEER coordination between 
users who ALREADY HAVE their own individual Claude Max subscriptions.

KEY LEGAL POINTS:
• Each user maintains their own Claude account and API credentials
• No API keys, tokens, or credentials are ever shared or transmitted
• All Claude requests execute locally on each user's own account
• This is compute coordination, NOT service resale
• Participation is entirely voluntary and user-controlled

ANTHROPIC TOS COMPLIANCE:
This system is designed to be fully compliant with Anthropic's Terms 
of Service by ensuring each user operates within their own licensed 
Claude account boundaries.
`;
  }

  getUsagePolicyText() {
    return `
PERMITTED USES:
✅ Coordinating distributed tasks across willing participants
✅ Sharing computational workload using your own Claude subscription
✅ Contributing spare capacity voluntarily to help others
✅ Earning RUV tokens as recognition for contribution
✅ Setting your own limits, controls, and participation rules

PROHIBITED USES:
❌ Sharing, copying, or transmitting API keys or credentials
❌ Accessing Claude through someone else's account
❌ Commercial resale of Claude access or API capacity
❌ Bypassing Anthropic's usage limits or terms
❌ Running tasks without explicit user approval
❌ Storing or caching Claude API responses long-term

PARTICIPATION MODEL:
This system operates like BOINC or Folding@Home - voluntary compute 
donation with token recognition, NOT commercial service resale.
`;
  }

  getCompliancePrinciplesText() {
    return `
1. NO SHARED API KEYS
   Each participant uses their own Claude Max account exclusively.
   API keys remain local and private to each user.

2. PEER-ORCHESTRATED MODEL
   Tasks are routed to willing participants, not through centralized accounts.
   Each node executes tasks on their own Claude subscription.

3. VOLUNTARY PARTICIPATION
   Users opt-in explicitly and can revoke consent at any time.
   No background processing without user knowledge.

4. TRANSPARENT OPERATION
   All activities are logged and auditable by the user.
   Full visibility into what tasks were executed and when.

5. USER CONTROL
   Users set their own limits, approve individual tasks, and
   maintain complete control over their participation level.

6. TOKEN INCENTIVES
   RUV tokens reward contribution, not access purchase.
   Tokens recognize computational donation, not service payment.
`;
  }

  getUserRightsText() {
    return `
AS A USER, YOU HAVE THE RIGHT TO:

CONSENT MANAGEMENT:
• Approve or deny each individual task before execution
• Set daily, weekly, or monthly usage limits
• Opt out completely at any time with immediate effect
• Revoke consent retroactively for future processing

TRANSPARENCY:
• View detailed logs of all tasks executed using your account
• Access full audit trails of your participation
• Review token earnings and computational contributions
• Monitor resource usage and API consumption

CONTROL:
• Set maximum tokens per task and daily task limits
• Configure timeout values for task execution
• Choose which types of tasks you're willing to process
• Pause participation temporarily without losing settings

PRIVACY:
• Your API credentials never leave your local system
• Task payloads are encrypted end-to-end
• No personal data is stored or transmitted
• Participate pseudonymously if desired

RECOURSE:
• Report compliance violations or concerns
• Request removal of all stored data
• Dispute token calculations or task results
• Access support for technical or legal issues
`;
  }

  /**
   * Generate compliance report for legal review
   */
  generateComplianceReport() {
    return {
      version: this.version,
      lastUpdated: this.lastUpdated,
      complianceChecks: [
        {
          requirement: 'No shared API keys',
          status: 'COMPLIANT',
          description: 'Each user maintains their own Claude credentials locally'
        },
        {
          requirement: 'No service resale',
          status: 'COMPLIANT', 
          description: 'System coordinates peer compute donation, not service access'
        },
        {
          requirement: 'User consent required',
          status: 'COMPLIANT',
          description: 'Explicit opt-in with granular controls and audit trails'
        },
        {
          requirement: 'Voluntary participation',
          status: 'COMPLIANT',
          description: 'Users can opt out at any time with immediate effect'
        },
        {
          requirement: 'Anthropic ToS compliance',
          status: 'COMPLIANT',
          description: 'Design ensures users operate within their licensed boundaries'
        }
      ],
      legalFramework: {
        model: 'Peer compute federation',
        tokenPurpose: 'Contribution recognition, not access payment',
        userControl: 'Complete autonomy over participation and usage',
        dataHandling: 'No API credentials stored or transmitted'
      },
      riskMitigation: [
        'Mandatory user consent with detailed explanations',
        'Granular usage controls and limits',
        'Comprehensive audit logging',
        'Immediate opt-out capabilities',
        'Regular compliance monitoring',
        'Clear separation from service resale'
      ]
    };
  }

  /**
   * Display short compliance summary
   */
  displayQuickTerms() {
    console.log(chalk.blue('📋 Quick Legal Summary:'));
    console.log(chalk.white(`
• You use your own Claude Max account and API key
• Your credentials never leave your computer
• You approve each task individually (or set auto-approval limits)
• You can opt out at any time
• RUV tokens reward contribution, not purchase access
• This is voluntary compute sharing, not service resale
    `));
  }

  /**
   * Check if user needs to review updated terms
   */
  needsTermsReview(lastAcceptedVersion) {
    // Simple version comparison - in practice, use semantic versioning
    return !lastAcceptedVersion || lastAcceptedVersion !== this.version;
  }

  /**
   * Get compliance checklist for implementation verification
   */
  getComplianceChecklist() {
    return [
      {
        item: 'API key isolation',
        description: 'Verify API keys are never transmitted or stored remotely',
        verified: false
      },
      {
        item: 'User consent flow',
        description: 'Ensure explicit opt-in with clear explanations',
        verified: false
      },
      {
        item: 'Usage controls',
        description: 'Implement granular limits and approval mechanisms',
        verified: false
      },
      {
        item: 'Audit logging',
        description: 'Log all activities for user transparency',
        verified: false
      },
      {
        item: 'Opt-out mechanism',
        description: 'Provide immediate and complete opt-out capability',
        verified: false
      },
      {
        item: 'Token design',
        description: 'Ensure tokens reward contribution, not purchase access',
        verified: false
      }
    ];
  }
}