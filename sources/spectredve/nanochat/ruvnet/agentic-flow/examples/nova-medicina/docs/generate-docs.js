#!/usr/bin/env node

/**
 * Documentation Swarm Coordinator
 *
 * Spawns concurrent documentation agents using agentic-flow Task tool
 * to generate comprehensive project documentation with memory coordination.
 *
 * Usage:
 *   node docs/generate-docs.js
 *   npm run generate-docs
 */

import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const CONFIG = {
  sessionId: `docswarm-${Date.now()}`,
  projectRoot: path.resolve(__dirname, '..'),
  docsDir: __dirname,
  memoryNamespace: 'nova-medicina/docs',
  agentTypes: [
    'readme-writer',
    'patient-guide',
    'provider-guide',
    'api-docs',
    'examples-gen',
    'safety-docs'
  ]
};

// Documentation specifications for each agent
const DOC_SPECS = {
  'readme-writer': {
    file: 'README.md',
    description: 'Generate comprehensive README with badges, quick start, and overview',
    requirements: [
      'Project title and description',
      'Status badges (build, coverage, version, license)',
      'Key features list',
      'Quick start guide',
      'Installation instructions',
      'Basic usage examples',
      'Configuration options',
      'Contributing guidelines',
      'License information',
      'Contact and support links'
    ],
    tone: 'professional and welcoming',
    audience: 'developers and technical users'
  },

  'patient-guide': {
    file: 'PATIENT_GUIDE.md',
    description: 'Create patient-friendly documentation in simple language',
    requirements: [
      'What is this project? (plain English)',
      'How does it help patients?',
      'Getting started (step-by-step)',
      'Understanding your results',
      'When to seek medical advice',
      'Privacy and data security',
      'Frequently asked questions',
      'Glossary of terms',
      'Support resources'
    ],
    tone: 'friendly, clear, and reassuring',
    audience: 'patients and caregivers with no technical background'
  },

  'provider-guide': {
    file: 'PROVIDER_GUIDE.md',
    description: 'Generate technical guide for healthcare providers',
    requirements: [
      'Clinical use cases',
      'Integration with EHR systems',
      'Data interpretation guidelines',
      'Clinical validation and accuracy',
      'Regulatory compliance',
      'Best practices for implementation',
      'Troubleshooting technical issues',
      'Advanced configuration',
      'Performance optimization',
      'Security and HIPAA compliance'
    ],
    tone: 'technical and authoritative',
    audience: 'healthcare providers, IT staff, and administrators'
  },

  'api-docs': {
    file: 'API.md',
    description: 'Document all API endpoints with examples',
    requirements: [
      'API overview and architecture',
      'Authentication and authorization',
      'Rate limiting and quotas',
      'Endpoint reference (all routes)',
      'Request/response schemas',
      'Error codes and handling',
      'Code examples (curl, JavaScript, Python)',
      'WebSocket events (if applicable)',
      'Versioning strategy',
      'Migration guides'
    ],
    tone: 'precise and technical',
    audience: 'developers integrating with the API'
  },

  'examples-gen': {
    file: 'EXAMPLES.md',
    description: 'Create practical usage scenarios and code examples',
    requirements: [
      'Basic usage example',
      'Advanced usage scenarios',
      'Integration examples (React, Vue, Node.js)',
      'Common workflows',
      'Error handling patterns',
      'Testing examples',
      'Performance optimization tips',
      'Real-world use cases',
      'Code snippets library',
      'Interactive examples (if applicable)'
    ],
    tone: 'practical and instructive',
    audience: 'developers implementing the project'
  },

  'safety-docs': {
    file: 'SAFETY.md',
    description: 'Generate safety disclaimers and medical warnings',
    requirements: [
      'Medical disclaimer',
      'Limitation of liability',
      'NOT a substitute for professional medical advice',
      'Emergency situations guidance',
      'Data accuracy limitations',
      'Regulatory status and approvals',
      'Risk warnings',
      'Contraindications',
      'Reporting adverse events',
      'Legal compliance notice'
    ],
    tone: 'clear, serious, and legally appropriate',
    audience: 'all users, patients, and healthcare providers'
  }
};

// Utility functions
function executeCommand(command, description) {
  try {
    console.log(`\n🔧 ${description}...`);
    const output = execSync(command, {
      encoding: 'utf-8',
      stdio: 'pipe',
      cwd: CONFIG.projectRoot
    });
    console.log(`✅ ${description} - Complete`);
    return output;
  } catch (error) {
    console.error(`❌ ${description} - Failed:`, error.message);
    return null;
  }
}

function storeInMemory(key, value) {
  const memoryKey = `${CONFIG.memoryNamespace}/${key}`;
  console.log(`💾 Storing in memory: ${memoryKey}`);

  // Store coordination data for agent access
  const memoryData = {
    key: memoryKey,
    value: JSON.stringify(value),
    timestamp: Date.now(),
    sessionId: CONFIG.sessionId
  };

  // Write to local memory file for coordination
  const memoryDir = path.join(CONFIG.projectRoot, '.swarm-memory');
  if (!fs.existsSync(memoryDir)) {
    fs.mkdirSync(memoryDir, { recursive: true });
  }

  const memoryFile = path.join(memoryDir, `${key.replace(/\//g, '_')}.json`);
  fs.writeFileSync(memoryFile, JSON.stringify(memoryData, null, 2));
}

function retrieveFromMemory(key) {
  const memoryFile = path.join(CONFIG.projectRoot, '.swarm-memory', `${key.replace(/\//g, '_')}.json`);

  if (fs.existsSync(memoryFile)) {
    const data = JSON.parse(fs.readFileSync(memoryFile, 'utf-8'));
    return JSON.parse(data.value);
  }
  return null;
}

function initializeSwarmMemory() {
  console.log('\n🧠 Initializing swarm memory...\n');

  // Store project metadata
  storeInMemory('project/metadata', {
    name: 'Nova Medicina',
    description: 'AI-powered healthcare documentation system',
    version: '1.0.0',
    type: 'healthcare-ai',
    framework: 'agentic-flow'
  });

  // Store documentation specifications
  Object.entries(DOC_SPECS).forEach(([agentType, spec]) => {
    storeInMemory(`specs/${agentType}`, spec);
  });

  // Store coordination rules
  storeInMemory('coordination/rules', {
    consistency: 'All documentation must use consistent terminology',
    crossReference: 'Reference other docs when relevant',
    branding: 'Use "Nova Medicina" consistently',
    tone: 'Professional, clear, and user-focused',
    format: 'Markdown with proper headings and structure',
    validation: 'All code examples must be tested'
  });

  console.log('✅ Swarm memory initialized\n');
}

function generateAgentPrompts() {
  const prompts = {};

  Object.entries(DOC_SPECS).forEach(([agentType, spec]) => {
    prompts[agentType] = `
# ${spec.description}

## Your Mission
Generate comprehensive ${spec.file} for the Nova Medicina project.

## Target Audience
${spec.audience}

## Tone
${spec.tone}

## Required Sections
${spec.requirements.map((req, idx) => `${idx + 1}. ${req}`).join('\n')}

## Coordination Instructions

### Before Starting:
\`\`\`bash
# Restore session memory
npx claude-flow@alpha hooks session-restore --session-id "${CONFIG.sessionId}"

# Retrieve project metadata
# Check .swarm-memory/project_metadata.json

# Retrieve documentation specs
# Check .swarm-memory/specs_${agentType}.json

# Retrieve coordination rules
# Check .swarm-memory/coordination_rules.json
\`\`\`

### During Work:
\`\`\`bash
# Store progress updates
npx claude-flow@alpha hooks post-edit --file "${spec.file}" --memory-key "swarm/${agentType}/progress"

# Notify other agents
npx claude-flow@alpha hooks notify --message "${agentType}: Completed section X"
\`\`\`

### After Completion:
\`\`\`bash
# Store final document info
npx claude-flow@alpha hooks post-edit --file "${spec.file}" --memory-key "swarm/${agentType}/complete"

# Mark task complete
npx claude-flow@alpha hooks post-task --task-id "${CONFIG.sessionId}-${agentType}"
\`\`\`

## Consistency Requirements
- Check other agents' progress in .swarm-memory/
- Use consistent terminology (check coordination_rules.json)
- Cross-reference related documentation
- Follow markdown best practices
- Include code examples where relevant
- Add table of contents for long documents

## Output Location
Write to: /home/user/agentic-flow/nova-medicina/docs/${spec.file}

## Quality Checklist
- [ ] All required sections included
- [ ] Appropriate tone for target audience
- [ ] Consistent with project branding
- [ ] Cross-references to other docs
- [ ] Code examples tested (if applicable)
- [ ] Proper markdown formatting
- [ ] Table of contents added
- [ ] Links verified

Remember: You are part of a coordinated swarm. Check memory for consistency!
`;
  });

  return prompts;
}

async function spawnDocumentationAgents() {
  console.log('\n🚀 Spawning Documentation Swarm Agents...\n');
  console.log('📋 Agents to spawn:');

  const agentPrompts = generateAgentPrompts();

  CONFIG.agentTypes.forEach((agentType, idx) => {
    const spec = DOC_SPECS[agentType];
    console.log(`  ${idx + 1}. ${agentType} → ${spec.file}`);
  });

  console.log('\n⚡ CONCURRENT EXECUTION PATTERN:');
  console.log('='.repeat(80));
  console.log('\nIn Claude Code, spawn ALL agents in a SINGLE message using Task tool:\n');

  CONFIG.agentTypes.forEach((agentType) => {
    const spec = DOC_SPECS[agentType];
    console.log(`Task("${spec.description}", ${JSON.stringify(agentPrompts[agentType])}, "coder")`);
    console.log('');
  });

  console.log('='.repeat(80));

  // Store agent prompts for reference
  storeInMemory('agent-prompts/all', agentPrompts);

  console.log('\n📝 Agent prompts stored in memory for coordination');
  console.log(`📂 Memory location: ${CONFIG.projectRoot}/.swarm-memory/`);

  // Create a manifest file for tracking
  const manifest = {
    sessionId: CONFIG.sessionId,
    startTime: new Date().toISOString(),
    agents: CONFIG.agentTypes.map(agentType => ({
      type: agentType,
      file: DOC_SPECS[agentType].file,
      status: 'pending',
      taskId: `${CONFIG.sessionId}-${agentType}`
    })),
    memoryNamespace: CONFIG.memoryNamespace,
    projectRoot: CONFIG.projectRoot
  };

  storeInMemory('manifest', manifest);

  console.log('\n✅ Documentation swarm coordinator ready!');
  console.log(`\n📊 Session ID: ${CONFIG.sessionId}`);
  console.log(`📁 Output directory: ${CONFIG.docsDir}`);

  return manifest;
}

function displayAgentInstructions() {
  console.log('\n' + '='.repeat(80));
  console.log('📖 HOW TO USE THIS COORDINATOR');
  console.log('='.repeat(80));
  console.log(`
This coordinator has prepared everything for concurrent documentation generation.

OPTION 1: Manual Agent Spawning (Recommended for Claude Code)
---------------------------------------------------------------
Copy the Task() calls above and paste them into Claude Code in a SINGLE message.
All agents will run concurrently and coordinate via memory hooks.

OPTION 2: Automated Execution (Requires agentic-flow runtime)
--------------------------------------------------------------
If you have agentic-flow runtime installed:

  npx agentic-flow execute --manifest .swarm-memory/manifest.json

OPTION 3: Individual Agent Execution
-------------------------------------
Run agents one at a time (slower, not recommended):

  node docs/generate-docs.js --agent readme-writer
  node docs/generate-docs.js --agent patient-guide
  # ... etc

MEMORY COORDINATION:
-------------------
All agents coordinate via: ${CONFIG.projectRoot}/.swarm-memory/
- Project metadata: .swarm-memory/project_metadata.json
- Agent specs: .swarm-memory/specs_*.json
- Coordination rules: .swarm-memory/coordination_rules.json
- Agent prompts: .swarm-memory/agent-prompts_all.json

MONITORING:
----------
Watch progress:  tail -f .swarm-memory/*.json
Check manifest:  cat .swarm-memory/manifest.json
View prompts:    cat .swarm-memory/agent-prompts_all.json
`);
  console.log('='.repeat(80) + '\n');
}

// Main execution
async function main() {
  console.log('\n' + '='.repeat(80));
  console.log('🏥 Nova Medicina Documentation Swarm Coordinator');
  console.log('='.repeat(80));

  // Initialize hooks
  console.log('\n🔌 Initializing coordination hooks...');
  executeCommand(
    `npx claude-flow@alpha hooks pre-task --description "Documentation swarm initialization"`,
    'Pre-task hook'
  );

  // Initialize swarm memory
  initializeSwarmMemory();

  // Spawn agents (prepare coordination)
  const manifest = await spawnDocumentationAgents();

  // Display instructions
  displayAgentInstructions();

  // Complete
  console.log('✅ Documentation swarm coordinator setup complete!\n');
  console.log('💡 Next steps: Use Claude Code Task tool to spawn agents concurrently\n');

  return manifest;
}

// Execute if run directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(error => {
    console.error('\n❌ Error:', error.message);
    process.exit(1);
  });
}

export { main, DOC_SPECS, CONFIG };
