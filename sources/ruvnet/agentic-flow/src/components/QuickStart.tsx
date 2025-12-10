import { Terminal, Rocket, Package, GitBranch } from 'lucide-react';

const installSteps = [
  {
    icon: Package,
    title: 'Install Package',
    command: 'npm install -g agentic-flow',
    description: 'Global installation for CLI access',
  },
  {
    icon: Terminal,
    title: 'Set API Key',
    command: 'export ANTHROPIC_API_KEY=sk-ant-...',
    description: 'Configure your Claude API credentials',
  },
  {
    icon: Rocket,
    title: 'Run Your First Agent',
    command: 'npx agentic-flow --agent researcher --task "Analyze trends"',
    description: 'Execute agents with full MCP tool access',
  },
  {
    icon: GitBranch,
    title: 'Or Use npx',
    command: 'npx agentic-flow --help',
    description: 'No installation required, run directly',
  },
];

const examples = [
  {
    title: 'Cost-Optimized Coding',
    code: `npx agentic-flow \\
  --agent coder \\
  --task "Build REST API" \\
  --optimize --priority cost`,
    description: 'Auto-select cheapest model for simple tasks',
  },
  {
    title: 'Quality-First Review',
    code: `npx agentic-flow \\
  --agent reviewer \\
  --task "Security audit" \\
  --optimize --priority quality`,
    description: 'Use flagship models for critical reviews',
  },
  {
    title: 'AgentDB Memory',
    code: `npx agentdb reflexion store \\
  "session-1" "implement_auth" \\
  0.95 true "Success!"`,
    description: 'Store agent learning experiences',
  },
];

const QuickStart = () => {
  return (
    <section className="py-24 px-6 bg-background-light/50">
      <div className="container mx-auto max-w-7xl">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            <span className="text-gradient">Quick Start</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Get up and running in minutes with our simple installation process
          </p>
        </div>

        {/* Installation Steps */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-16">
          {installSteps.map((step, index) => {
            const Icon = step.icon;
            
            return (
              <div
                key={index}
                className="bg-card border border-border rounded-2xl p-6 hover:shadow-glow transition-smooth hover:scale-105 animate-fade-in"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div className="w-12 h-12 bg-gradient-primary rounded-xl flex items-center justify-center mb-4">
                  <Icon className="w-6 h-6 text-foreground" />
                </div>
                
                <div className="text-sm text-primary font-semibold mb-2">
                  Step {index + 1}
                </div>
                
                <h3 className="text-lg font-bold mb-3 text-foreground">
                  {step.title}
                </h3>
                
                <code className="block bg-background rounded-lg p-3 text-xs text-primary font-mono mb-3 break-all">
                  {step.command}
                </code>
                
                <p className="text-sm text-muted-foreground">
                  {step.description}
                </p>
              </div>
            );
          })}
        </div>

        {/* Code Examples */}
        <div className="space-y-8">
          <h3 className="text-3xl font-bold text-center mb-8 text-foreground">
            Common Use Cases
          </h3>
          
          {examples.map((example, index) => (
            <div
              key={index}
              className="bg-card border border-border rounded-2xl p-8 hover:shadow-glow transition-smooth animate-fade-in"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <div className="flex flex-col lg:flex-row gap-6">
                <div className="lg:w-1/3">
                  <h4 className="text-xl font-bold mb-3 text-foreground">
                    {example.title}
                  </h4>
                  <p className="text-muted-foreground">
                    {example.description}
                  </p>
                </div>
                
                <div className="lg:w-2/3">
                  <pre className="bg-background rounded-xl p-6 overflow-x-auto border border-border-light">
                    <code className="text-sm text-primary font-mono">
                      {example.code}
                    </code>
                  </pre>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Programmatic API */}
        <div className="mt-16 bg-gradient-primary rounded-2xl p-8">
          <h3 className="text-2xl font-bold mb-6 text-center text-foreground">
            Programmatic API
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-background/10 rounded-xl p-6 backdrop-blur-sm">
              <div className="text-sm font-semibold mb-3 text-foreground/80">
                Import Components
              </div>
              <pre className="bg-background/20 rounded-lg p-4 overflow-x-auto">
                <code className="text-sm text-foreground font-mono">
{`import { ReflexionMemory } from 'agentic-flow/agentdb';
import { ModelRouter } from 'agentic-flow/router';
import * as reasoningbank from 'agentic-flow/reasoningbank';`}
                </code>
              </pre>
            </div>
            
            <div className="bg-background/10 rounded-xl p-6 backdrop-blur-sm">
              <div className="text-sm font-semibold mb-3 text-foreground/80">
                Use in Code
              </div>
              <pre className="bg-background/20 rounded-lg p-4 overflow-x-auto">
                <code className="text-sm text-foreground font-mono">
{`const router = new ModelRouter();
const response = await router.chat({
  model: 'auto', priority: 'cost'
});`}
                </code>
              </pre>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default QuickStart;
