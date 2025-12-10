# Contributing to Sovereign Agentic Stack

Thank you for your interest in contributing! This project integrates multiple open-source components into a cohesive sovereign AI architecture.

## How to Contribute

### Reporting Issues

- Check existing issues first to avoid duplicates
- Provide detailed reproduction steps
- Include system information (macOS version, RAM, chip model)
- Share relevant logs from `gaianet log` or console output

### Suggesting Enhancements

- Describe the enhancement clearly
- Explain the use case and benefits
- Consider backward compatibility

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test thoroughly on Apple Silicon if possible
5. Update documentation as needed
6. Commit with clear messages
7. Push to your fork
8. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Agentic-local.git
cd Agentic-local

# Install dependencies
npm install

# Set up environment
cp .env.example .env

# Run tests
npm run sandbox:test
```

## Code Style

- Follow existing code patterns
- Use meaningful variable and function names
- Comment complex logic
- Keep functions focused and modular

## Testing

Before submitting a PR:

1. Run sandbox security tests: `npm run sandbox:test`
2. Test with actual agent workflows
3. Verify documentation accuracy
4. Check for any breaking changes

## Areas for Contribution

### High Priority

- [ ] Support for additional LLM models (Llama 3, Mistral, etc.)
- [ ] Windows/Linux compatibility improvements
- [ ] Enhanced monitoring and telemetry
- [ ] Additional sandbox language support (Rust, Go)
- [ ] Performance benchmarking suite

### Medium Priority

- [ ] Web UI for agent management
- [ ] Pre-built Docker images
- [ ] Integration tests for full stack
- [ ] Cost calculator tool
- [ ] Model quantization scripts

### Documentation

- [ ] Video tutorials
- [ ] Additional use case examples
- [ ] Troubleshooting guide expansion
- [ ] Performance tuning guide
- [ ] Migration guides from cloud APIs

## Component-Specific Contributions

This project integrates several upstream projects. For issues specific to:

- **WasmEdge**: https://github.com/WasmEdge/WasmEdge
- **LlamaEdge**: https://github.com/LlamaEdge/LlamaEdge
- **GaiaNet**: https://github.com/GaiaNet-AI
- **Ruvnet packages**: https://www.npmjs.com/~ruvnet (contact package maintainer)

Please report issues to the appropriate upstream project.

## Questions?

- Open a GitHub Discussion for questions
- Check the documentation in `docs/`
- Review existing issues

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Celebrate contributions of all sizes

Thank you for helping make sovereign AI accessible to everyone! 🚀
