import { BarChart3, TrendingDown, Clock, DollarSign } from 'lucide-react';

const metrics = [
  {
    icon: Clock,
    title: 'Code Review (100/day)',
    traditional: { latency: '35s', cost: '$240/mo', accuracy: '70%' },
    agenticFlow: { latency: '0.1s', cost: '$0/mo', accuracy: '90%' },
    improvement: '352x faster, 100% free',
  },
  {
    icon: BarChart3,
    title: 'Migration (1000 files)',
    traditional: { time: '5.87 min', cost: '$10' },
    agenticFlow: { time: '1 sec', cost: '$0' },
    improvement: '350x faster, $10 saved',
  },
  {
    icon: TrendingDown,
    title: 'Refactoring Pipeline',
    traditional: { success: '70%', supervision: 'Required' },
    agenticFlow: { success: '90%', supervision: 'Zero' },
    improvement: '+46% execution speed',
  },
  {
    icon: DollarSign,
    title: 'Cost Optimization',
    traditional: { model: 'Claude Sonnet 4.5', cost: '$240/mo' },
    agenticFlow: { model: 'DeepSeek R1', cost: '$36/mo' },
    improvement: '85% cost reduction',
  },
];

const PerformanceMetrics = () => {
  return (
    <section className="py-24 px-6">
      <div className="container mx-auto max-w-7xl">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            <span className="text-gradient">Real-World Performance</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            See how Agentic Flow transforms traditional workflows with measurable results
          </p>
        </div>

        <div className="space-y-8">
          {metrics.map((metric, index) => {
            const Icon = metric.icon;
            
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
                  <h3 className="text-2xl font-bold text-foreground">{metric.title}</h3>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {/* Traditional */}
                  <div className="bg-background-light/50 rounded-xl p-6 border border-border-light">
                    <div className="text-sm text-muted-foreground mb-4 font-semibold">
                      Traditional Agent
                    </div>
                    <div className="space-y-2">
                      {Object.entries(metric.traditional).map(([key, value]) => (
                        <div key={key} className="flex justify-between text-sm">
                          <span className="text-muted-foreground capitalize">{key}:</span>
                          <span className="text-foreground font-mono">{value}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Agentic Flow */}
                  <div className="bg-primary/10 rounded-xl p-6 border border-primary/30">
                    <div className="text-sm text-primary mb-4 font-semibold">
                      Agentic Flow
                    </div>
                    <div className="space-y-2">
                      {Object.entries(metric.agenticFlow).map(([key, value]) => (
                        <div key={key} className="flex justify-between text-sm">
                          <span className="text-muted-foreground capitalize">{key}:</span>
                          <span className="text-foreground font-mono font-semibold">{value}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Improvement */}
                  <div className="flex items-center justify-center">
                    <div className="text-center">
                      <div className="text-3xl font-bold text-gradient mb-2">
                        {metric.improvement.split(',')[0]}
                      </div>
                      {metric.improvement.split(',')[1] && (
                        <div className="text-lg font-semibold text-secondary">
                          {metric.improvement.split(',')[1]}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Summary Card */}
        <div className="mt-12 bg-gradient-primary rounded-2xl p-8 text-center">
          <p className="text-2xl md:text-3xl font-bold text-foreground">
            The only agent framework that gets faster AND smarter the more you use it
          </p>
        </div>
      </div>
    </section>
  );
};

export default PerformanceMetrics;
