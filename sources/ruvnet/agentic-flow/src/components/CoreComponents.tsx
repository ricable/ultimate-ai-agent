import { Database, Code, Network, Cpu } from 'lucide-react';

const components = [
  {
    icon: Code,
    name: 'Agent Booster',
    description: 'Ultra-fast local code transformations via Rust/WASM with automatic edit detection',
    features: ['352x faster', '$0 cost', 'Automatic detection', 'Rust/WASM powered'],
    docs: 'https://github.com/ruvnet/agentic-flow/tree/main/agent-booster',
  },
  {
    icon: Database,
    name: 'AgentDB',
    description: 'Advanced memory system with causal reasoning, reflexion, and skill learning',
    features: ['p95 < 50ms', '80% hit rate', '17 CLI commands', 'Causal memory graph'],
    docs: './agentic-flow/src/agentdb/README.md',
  },
  {
    icon: Network,
    name: 'ReasoningBank',
    description: 'Persistent learning memory with semantic search and pattern recognition',
    features: ['46% faster', '100% success', 'Semantic search', 'Pattern learning'],
    docs: 'https://github.com/ruvnet/agentic-flow/tree/main/agentic-flow/src/reasoningbank',
  },
  {
    icon: Cpu,
    name: 'Multi-Model Router',
    description: 'Intelligent cost optimization across 100+ LLM models with auto-selection',
    features: ['85-99% savings', '100+ models', 'Auto-select', 'Quality balanced'],
    docs: 'https://github.com/ruvnet/agentic-flow/tree/main/agentic-flow/src/router',
  },
];

const CoreComponents = () => {
  return (
    <section className="py-24 px-6 bg-background-light/50">
      <div className="container mx-auto max-w-7xl">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            <span className="text-gradient">Core Components</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Modular architecture designed for maximum performance and flexibility
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {components.map((component, index) => {
            const Icon = component.icon;
            
            return (
              <div
                key={index}
                className="bg-card border border-border rounded-2xl p-8 hover:shadow-glow transition-smooth hover:scale-[1.02] animate-fade-in"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div className="flex items-start gap-6">
                  <div className="w-16 h-16 bg-gradient-primary rounded-xl flex items-center justify-center flex-shrink-0">
                    <Icon className="w-8 h-8 text-foreground" />
                  </div>
                  
                  <div className="flex-1">
                    <h3 className="text-2xl font-bold mb-3 text-foreground">
                      {component.name}
                    </h3>
                    
                    <p className="text-muted-foreground mb-6 leading-relaxed">
                      {component.description}
                    </p>
                    
                    <div className="grid grid-cols-2 gap-3 mb-6">
                      {component.features.map((feature, idx) => (
                        <div
                          key={idx}
                          className="flex items-center gap-2 text-sm text-foreground/80"
                        >
                          <div className="w-1.5 h-1.5 bg-primary rounded-full" />
                          {feature}
                        </div>
                      ))}
                    </div>
                    
                    <a
                      href={component.docs}
                      className="inline-flex items-center gap-2 text-primary hover:text-primary-glow transition-smooth font-semibold"
                    >
                      View Documentation →
                    </a>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* CLI Usage */}
        <div className="mt-16 bg-background-light border border-border rounded-2xl p-8">
          <h3 className="text-2xl font-bold mb-6 text-foreground">CLI Usage</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <div className="text-sm text-muted-foreground mb-2">AgentDB - Full CLI</div>
              <code className="block bg-background rounded-lg p-4 text-sm text-primary font-mono">
                npx agentdb reflexion store "session-1" ...
              </code>
            </div>
            <div>
              <div className="text-sm text-muted-foreground mb-2">Multi-Model Router</div>
              <code className="block bg-background rounded-lg p-4 text-sm text-primary font-mono">
                npx agentic-flow --agent coder --optimize
              </code>
            </div>
            <div>
              <div className="text-sm text-muted-foreground mb-2">Agent Booster</div>
              <code className="block bg-background rounded-lg p-4 text-sm text-primary font-mono">
                Automatic on code edits
              </code>
            </div>
            <div>
              <div className="text-sm text-muted-foreground mb-2">ReasoningBank</div>
              <code className="block bg-background rounded-lg p-4 text-sm text-primary font-mono">
                import * as reasoningbank from 'agentic-flow'
              </code>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default CoreComponents;
