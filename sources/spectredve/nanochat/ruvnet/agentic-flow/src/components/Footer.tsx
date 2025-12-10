import { Github, Book, Package, Twitter, ExternalLink } from 'lucide-react';

const links = {
  documentation: [
    { name: 'Getting Started', url: '#' },
    { name: 'API Reference', url: '#' },
    { name: 'Agent Guide', url: '#' },
    { name: 'Examples', url: '#' },
  ],
  resources: [
    { name: 'GitHub', url: 'https://github.com/ruvnet/agentic-flow', icon: Github },
    { name: 'npm Package', url: 'https://www.npmjs.com/package/agentic-flow', icon: Package },
    { name: 'Documentation', url: 'https://github.com/ruvnet/agentic-flow/tree/main/docs', icon: Book },
  ],
  components: [
    { name: 'Agent Booster', url: '#' },
    { name: 'AgentDB', url: '#' },
    { name: 'ReasoningBank', url: '#' },
    { name: 'Multi-Model Router', url: '#' },
  ],
};

const stats = [
  { label: 'npm Downloads', value: '10K+' },
  { label: 'GitHub Stars', value: '1.2K+' },
  { label: 'Agents Available', value: '150+' },
  { label: 'MCP Tools', value: '213' },
];

const Footer = () => {
  return (
    <footer className="bg-background-light border-t border-border py-16 px-6">
      <div className="container mx-auto max-w-7xl">
        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-16">
          {stats.map((stat, index) => (
            <div
              key={index}
              className="text-center p-6 bg-card rounded-xl border border-border"
            >
              <div className="text-3xl font-bold text-gradient mb-2">
                {stat.value}
              </div>
              <div className="text-sm text-muted-foreground">
                {stat.label}
              </div>
            </div>
          ))}
        </div>

        {/* Links */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-12 mb-16">
          <div>
            <h4 className="text-lg font-bold mb-6 text-foreground">
              Documentation
            </h4>
            <ul className="space-y-3">
              {links.documentation.map((link, index) => (
                <li key={index}>
                  <a
                    href={link.url}
                    className="text-muted-foreground hover:text-primary transition-smooth flex items-center gap-2"
                  >
                    {link.name}
                    <ExternalLink className="w-3 h-3" />
                  </a>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h4 className="text-lg font-bold mb-6 text-foreground">
              Resources
            </h4>
            <ul className="space-y-3">
              {links.resources.map((link, index) => {
                const Icon = link.icon;
                return (
                  <li key={index}>
                    <a
                      href={link.url}
                      className="text-muted-foreground hover:text-primary transition-smooth flex items-center gap-2"
                    >
                      <Icon className="w-4 h-4" />
                      {link.name}
                    </a>
                  </li>
                );
              })}
            </ul>
          </div>

          <div>
            <h4 className="text-lg font-bold mb-6 text-foreground">
              Components
            </h4>
            <ul className="space-y-3">
              {links.components.map((link, index) => (
                <li key={index}>
                  <a
                    href={link.url}
                    className="text-muted-foreground hover:text-primary transition-smooth"
                  >
                    {link.name}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="border-t border-border pt-8">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <div className="text-center md:text-left">
              <div className="text-2xl font-bold text-gradient mb-2">
                Agentic Flow
              </div>
              <p className="text-sm text-muted-foreground">
                The First AI Agent Framework That Gets Smarter AND Faster
              </p>
            </div>

            <div className="flex items-center gap-6">
              <a
                href="https://github.com/ruvnet/agentic-flow"
                className="text-muted-foreground hover:text-primary transition-smooth"
              >
                <Github className="w-6 h-6" />
              </a>
              <a
                href="https://www.npmjs.com/package/agentic-flow"
                className="text-muted-foreground hover:text-primary transition-smooth"
              >
                <Package className="w-6 h-6" />
              </a>
            </div>
          </div>

          <div className="mt-8 text-center text-sm text-muted-foreground">
            <p>
              MIT License © 2025 rUv |{' '}
              <a href="#" className="hover:text-primary transition-smooth">
                Terms
              </a>{' '}
              |{' '}
              <a href="#" className="hover:text-primary transition-smooth">
                Privacy
              </a>
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
