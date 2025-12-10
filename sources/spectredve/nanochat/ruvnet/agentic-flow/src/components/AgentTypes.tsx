import { Code, GitPullRequest, TestTube, Layers, Brain, Workflow } from 'lucide-react';

const agentCategories = [
  {
    icon: Code,
    title: 'Core Development',
    agents: [
      { name: 'coder', desc: 'Implementation specialist for clean code' },
      { name: 'reviewer', desc: 'Code review and quality assurance' },
      { name: 'tester', desc: 'Comprehensive testing with 90%+ coverage' },
      { name: 'planner', desc: 'Strategic planning and task decomposition' },
      { name: 'researcher', desc: 'Deep research and information gathering' },
    ],
  },
  {
    icon: Layers,
    title: 'Specialized Agents',
    agents: [
      { name: 'backend-dev', desc: 'REST/GraphQL API development' },
      { name: 'mobile-dev', desc: 'React Native mobile apps' },
      { name: 'ml-developer', desc: 'Machine learning model creation' },
      { name: 'system-architect', desc: 'System design and architecture' },
      { name: 'cicd-engineer', desc: 'CI/CD pipeline creation' },
    ],
  },
  {
    icon: Workflow,
    title: 'Swarm Coordinators',
    agents: [
      { name: 'hierarchical-coordinator', desc: 'Tree-based leadership' },
      { name: 'mesh-coordinator', desc: 'Peer-to-peer coordination' },
      { name: 'adaptive-coordinator', desc: 'Dynamic topology switching' },
      { name: 'swarm-memory-manager', desc: 'Cross-agent memory sync' },
    ],
  },
  {
    icon: GitPullRequest,
    title: 'GitHub Integration',
    agents: [
      { name: 'pr-manager', desc: 'Pull request lifecycle management' },
      { name: 'code-review-swarm', desc: 'Multi-agent code review' },
      { name: 'issue-tracker', desc: 'Intelligent issue management' },
      { name: 'release-manager', desc: 'Automated release coordination' },
      { name: 'workflow-automation', desc: 'GitHub Actions specialist' },
    ],
  },
];

const AgentTypes = () => {
  return (
    <section className="py-24 px-6">
      <div className="container mx-auto max-w-7xl">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            <span className="text-gradient">150+ Specialized Agents</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Purpose-built agents for every development task and workflow
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {agentCategories.map((category, index) => {
            const Icon = category.icon;
            
            return (
              <div
                key={index}
                className="bg-card border border-border rounded-2xl p-8 hover:shadow-glow transition-smooth animate-fade-in"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div className="flex items-center gap-4 mb-6">
                  <div className="w-12 h-12 bg-gradient-primary rounded-xl flex items-center justify-center">
                    <Icon className="w-6 h-6 text-foreground" />
                  </div>
                  <h3 className="text-2xl font-bold text-foreground">
                    {category.title}
                  </h3>
                </div>

                <div className="space-y-3">
                  {category.agents.map((agent, idx) => (
                    <div
                      key={idx}
                      className="bg-background-light rounded-lg p-4 border border-border-light hover:border-primary/50 transition-smooth"
                    >
                      <div className="flex items-start gap-3">
                        <div className="w-2 h-2 bg-primary rounded-full mt-2 flex-shrink-0" />
                        <div className="flex-1">
                          <div className="font-mono text-sm text-primary font-semibold mb-1">
                            {agent.name}
                          </div>
                          <div className="text-sm text-muted-foreground">
                            {agent.desc}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>

        {/* View All */}
        <div className="mt-12 text-center">
          <div className="inline-block bg-card border border-border rounded-xl p-6">
            <p className="text-muted-foreground mb-4">
              Explore all available agents
            </p>
            <code className="block bg-background rounded-lg p-4 text-primary font-mono">
              npx agentic-flow --list
            </code>
          </div>
        </div>
      </div>
    </section>
  );
};

export default AgentTypes;
