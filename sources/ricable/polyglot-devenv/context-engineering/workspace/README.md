# Context Engineering Workspace

The **workspace** directory contains all resources for local development and PRP generation.

## Structure

- **features/** - Feature definition files that serve as input for PRP generation
- **templates/** - Environment-specific PRP templates and fragments
- **generators/** - Tools and scripts for generating PRPs
- **docs/** - Documentation specific to workspace usage

## Workflow

1. **Define Features**: Create feature definition files in `features/`
2. **Generate PRPs**: Use templates to generate PRPs from feature definitions
3. **Review PRPs**: Generated PRPs are reviewed before moving to DevPod execution

## Usage

```bash
# Navigate to workspace
cd context-engineering/workspace

# Generate a PRP for Python environment
/generate-prp features/user-api.md --env dev-env/python

# Review generated templates
code templates/python_prp.md
```

## Integration with Personal Workflows

This workspace integrates with personal development aliases:

```bash
# Personal aliases (add to CLAUDE.local.md)
alias prp-gen="cd context-engineering/workspace && /generate-prp"
alias prp-features="code context-engineering/workspace/features"
alias prp-templates="code context-engineering/workspace/templates"
```

## Templates

Templates are organized by environment:
- `python_prp.md` - Python/FastAPI patterns
- `typescript_prp.md` - TypeScript/Node.js patterns  
- `rust_prp.md` - Rust/Tokio patterns
- `go_prp.md` - Go patterns
- `nushell_prp.md` - Nushell automation patterns
- `prp_base.md` - Multi-environment base template

## Next Steps

After PRP generation in workspace, PRPs are executed in the containerized DevPod environments located in `../devpod/`.