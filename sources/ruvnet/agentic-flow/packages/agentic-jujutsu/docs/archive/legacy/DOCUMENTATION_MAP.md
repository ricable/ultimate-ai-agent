# Documentation Organization Map

> Complete documentation structure, organization rationale, and maintenance guidelines

## 📋 Table of Contents

- [Directory Structure](#directory-structure)
- [Organization Rationale](#organization-rationale)
- [File Categories](#file-categories)
- [Maintenance Guidelines](#maintenance-guidelines)
- [Update Procedures](#update-procedures)

## 📁 Directory Structure

```
packages/agentic-jujutsu/
├── README.md                      # Project overview and quick start
├── docs/
│   ├── INDEX.md                   # Master navigation index
│   ├── DOCUMENTATION_MAP.md       # This file
│   │
│   ├── getting-started/           # Quick start guides
│   │   ├── wasm-usage.md          # WASM bindings usage guide
│   │   └── BENCHMARK_QUICK_START.md  # Quick benchmark setup
│   │
│   ├── architecture/              # System design documentation
│   │   └── ARCHITECTURE.md        # Core architecture, ADRs, type system
│   │
│   ├── api/                       # API reference documentation
│   │   ├── HOOKS_INTEGRATION.md   # Hooks API quick reference
│   │   └── hooks-integration.md   # Detailed hooks integration guide
│   │
│   ├── development/               # Developer guides
│   │   └── testing.md             # Testing strategies and guidelines
│   │
│   ├── benchmarks/                # Benchmark system documentation
│   │   ├── README.md              # Benchmark overview
│   │   ├── ARCHITECTURE.md        # Benchmark architecture
│   │   ├── COMPONENT_DESIGN.md    # Component specifications
│   │   ├── DATA_MODELS.md         # Data models and schemas
│   │   ├── INTEGRATION_GUIDE.md   # Integration patterns
│   │   ├── SCALABILITY.md         # Performance analysis
│   │   ├── SUMMARY.md             # Executive summary
│   │   ├── BENCHMARK_DEPENDENCY_GRAPH.md     # Dependency analysis
│   │   ├── BENCHMARK_IMPLEMENTATION_PLAN.md  # Implementation roadmap
│   │   ├── BENCHMARK_VISUAL_SUMMARY.md       # Visual documentation
│   │   └── BENCHMARK_README.md    # Detailed benchmark guide
│   │
│   └── reports/                   # Status and analysis reports
│       ├── BUILD_STATUS.md        # Current build status
│       ├── IMPLEMENTATION_SUMMARY.md          # Implementation details
│       ├── INTEGRATION_SUMMARY.md             # Integration completion
│       ├── RUST_IMPLEMENTATION_SUMMARY.md     # Rust implementation
│       ├── TEST_REPORT.md                     # Test coverage
│       └── WASM_DELIVERABLES_SUMMARY.md       # WASM deliverables
│
├── examples/                      # Code examples
│   ├── javascript/                # JavaScript examples
│   └── integration/               # Integration examples
│
├── src/                           # Source code
├── tests/                         # Test files
├── benches/                       # Benchmark implementations
└── scripts/                       # Build and utility scripts
```

## 🎯 Organization Rationale

### Design Principles

1. **Progressive Disclosure**: Information organized from simple to complex
2. **Role-Based Access**: Documentation grouped by user role and use case
3. **Discoverability**: Clear navigation paths and comprehensive index
4. **Maintainability**: Logical grouping reduces duplication
5. **Scalability**: Structure supports future growth

### Directory Purpose

#### `getting-started/`
**Purpose**: Fast onboarding for new users
**Audience**: First-time users, developers integrating the library
**Content Type**: Quick start guides, tutorials, basic usage examples
**File Naming**: `kebab-case.md` for readability

#### `architecture/`
**Purpose**: System design documentation
**Audience**: System architects, senior developers, contributors
**Content Type**: Architecture Decision Records (ADRs), design patterns, type systems
**File Naming**: `SCREAMING_SNAKE_CASE.md` for official documents

#### `api/`
**Purpose**: API reference and integration guides
**Audience**: Developers using the API, integration engineers
**Content Type**: API specifications, integration guides, usage patterns
**File Naming**: Mixed case based on document formality

#### `development/`
**Purpose**: Contributor and development guides
**Audience**: Contributors, maintainers, QA engineers
**Content Type**: Development workflows, testing strategies, coding standards
**File Naming**: `kebab-case.md` for accessibility

#### `benchmarks/`
**Purpose**: Performance analysis and benchmark documentation
**Audience**: Performance engineers, researchers, optimization specialists
**Content Type**: Benchmark specs, performance analysis, scalability guides
**File Naming**: Mixed case - `README.md` for overview, `SCREAMING_SNAKE_CASE.md` for formal docs

#### `reports/`
**Purpose**: Status updates and analysis reports
**Audience**: Project managers, stakeholders, contributors
**Content Type**: Build status, implementation summaries, progress reports
**File Naming**: `SCREAMING_SNAKE_CASE.md` for official status documents

## 📚 File Categories

### Category 1: Entry Points

**Purpose**: First contact for users

| File | Location | Audience |
|------|----------|----------|
| README.md | Root | Everyone |
| INDEX.md | docs/ | Everyone |
| wasm-usage.md | getting-started/ | New users |
| BENCHMARK_QUICK_START.md | getting-started/ | Benchmark users |

### Category 2: Reference Documentation

**Purpose**: Detailed technical specifications

| File | Location | Topic |
|------|----------|-------|
| ARCHITECTURE.md | architecture/ | System design |
| HOOKS_INTEGRATION.md | api/ | Hooks API |
| hooks-integration.md | api/ | Hooks details |
| testing.md | development/ | Testing |

### Category 3: Benchmark Documentation

**Purpose**: Performance analysis and optimization

| File | Location | Focus |
|------|----------|-------|
| README.md | benchmarks/ | Overview |
| ARCHITECTURE.md | benchmarks/ | Design |
| COMPONENT_DESIGN.md | benchmarks/ | Components |
| DATA_MODELS.md | benchmarks/ | Data structures |
| INTEGRATION_GUIDE.md | benchmarks/ | Integration |
| SCALABILITY.md | benchmarks/ | Performance |
| SUMMARY.md | benchmarks/ | Executive summary |
| Other BENCHMARK_*.md | benchmarks/ | Specialized topics |

### Category 4: Status & Reports

**Purpose**: Current state and progress tracking

| File | Location | Content |
|------|----------|---------|
| BUILD_STATUS.md | reports/ | Build state |
| IMPLEMENTATION_SUMMARY.md | reports/ | Implementation |
| INTEGRATION_SUMMARY.md | reports/ | Integration |
| RUST_IMPLEMENTATION_SUMMARY.md | reports/ | Rust code |
| TEST_REPORT.md | reports/ | Testing |
| WASM_DELIVERABLES_SUMMARY.md | reports/ | WASM |

## 🔧 Maintenance Guidelines

### Adding New Documentation

#### 1. Determine Category

Ask these questions:
- **Is it a quick start guide?** → `getting-started/`
- **Is it architectural?** → `architecture/`
- **Is it API reference?** → `api/`
- **Is it for developers?** → `development/`
- **Is it about benchmarks?** → `benchmarks/`
- **Is it a status report?** → `reports/`

#### 2. Choose File Name

**For guides**: Use `kebab-case.md`
- Examples: `wasm-usage.md`, `testing.md`, `hooks-integration.md`

**For official documents**: Use `SCREAMING_SNAKE_CASE.md`
- Examples: `ARCHITECTURE.md`, `BUILD_STATUS.md`, `TEST_REPORT.md`

**For overviews**: Use `README.md`
- Location: In each major directory

#### 3. Update Index Files

After adding a file, update:
1. `docs/INDEX.md` - Add to appropriate section
2. `docs/DOCUMENTATION_MAP.md` - Add to file categories
3. Parent directory `README.md` (if exists)
4. Root `README.md` - Update links if necessary

#### 4. Link Validation

Ensure all internal links use relative paths:
```markdown
✅ Good: [Architecture](architecture/ARCHITECTURE.md)
❌ Bad:  [Architecture](/docs/architecture/ARCHITECTURE.md)
❌ Bad:  [Architecture](https://github.com/.../ARCHITECTURE.md)
```

### Updating Existing Documentation

#### When to Update

Update documentation when:
- Code changes affect documented behavior
- New features are added
- Bugs are fixed that change documented behavior
- Architecture decisions are made or changed
- Performance characteristics change

#### Update Checklist

- [ ] Update the file content
- [ ] Update "Last Updated" date in file footer
- [ ] Check for broken links
- [ ] Update related files (cross-references)
- [ ] Update INDEX.md if file purpose changed
- [ ] Update DOCUMENTATION_MAP.md if location changed
- [ ] Run link checker (if available)
- [ ] Review by another contributor

### Removing Documentation

#### Before Removing

1. **Check dependencies**: Search for links to the file
2. **Create redirect**: Add note in INDEX.md if content moved
3. **Update references**: Fix all broken links
4. **Archive**: Consider moving to `archive/` instead of deleting

#### Removal Checklist

- [ ] Identify all files linking to removed document
- [ ] Update or remove those links
- [ ] Remove from INDEX.md
- [ ] Remove from DOCUMENTATION_MAP.md
- [ ] Add redirect note if content moved
- [ ] Update related documentation

## 📝 Update Procedures

### Standard Update Workflow

```bash
# 1. Make changes to documentation
vim docs/getting-started/wasm-usage.md

# 2. Update last modified date in file

# 3. Check for broken links
grep -r "wasm-usage.md" docs/

# 4. Update index if needed
vim docs/INDEX.md

# 5. Commit with descriptive message
git add docs/
git commit -m "docs: update WASM usage guide with new examples"
```

### Major Reorganization

When reorganizing documentation structure:

1. **Plan the change**
   - Document new structure
   - Map old paths to new paths
   - Identify affected files

2. **Execute the move**
   - Move files to new locations
   - Update internal links
   - Update INDEX.md
   - Update DOCUMENTATION_MAP.md

3. **Validate the change**
   - Run link checker
   - Build documentation (if applicable)
   - Review all changed files

4. **Document the change**
   - Update DOCUMENTATION_MAP.md history
   - Add note to INDEX.md recent changes
   - Update root README.md if needed

### Quarterly Review

Every 3 months, review:

- [ ] **Accuracy**: Are docs up to date with code?
- [ ] **Completeness**: Are new features documented?
- [ ] **Organization**: Is structure still logical?
- [ ] **Links**: Are all links working?
- [ ] **Examples**: Do code examples still work?
- [ ] **Clarity**: Is documentation clear and helpful?

## 📊 Documentation Metrics

### Coverage Metrics

| Category | Files | Coverage |
|----------|-------|----------|
| Getting Started | 2 | ✅ Complete |
| Architecture | 1 | ✅ Complete |
| API Reference | 2 | ✅ Complete |
| Development | 1 | ⚠️  Needs contribution guide |
| Benchmarks | 11 | ✅ Complete |
| Reports | 6 | ✅ Complete |

### Health Indicators

✅ **Healthy**:
- Clear navigation paths
- Role-based organization
- Comprehensive index
- Regular updates

⚠️ **Needs Attention**:
- Missing contribution guide in development/
- Could add more API examples
- Benchmarks could use more diagrams

## 🔄 Version History

### v1.0 - 2025-11-09
- Initial organized structure
- Created INDEX.md
- Created DOCUMENTATION_MAP.md
- Organized files into categories
- Established maintenance guidelines

### v0.1 - 2025-11-07
- Initial documentation files
- Flat structure in docs/
- Basic README

## 🤝 Contributing to Documentation

### Writing Style Guide

1. **Be concise**: Get to the point quickly
2. **Use examples**: Show, don't just tell
3. **Be consistent**: Follow naming conventions
4. **Use diagrams**: Visual aids help understanding
5. **Test examples**: Ensure code examples work

### Review Checklist

Before submitting documentation changes:

- [ ] Spell check completed
- [ ] Grammar check completed
- [ ] Code examples tested
- [ ] Links validated
- [ ] INDEX.md updated
- [ ] Last updated date current
- [ ] Follows style guide
- [ ] Reviewed by another person

## 📞 Questions?

If you have questions about documentation:

1. Check this DOCUMENTATION_MAP.md
2. Review INDEX.md for navigation
3. Check the specific document's purpose
4. File an issue if still unclear

---

**Last Updated**: 2025-11-09
**Maintained By**: System Architecture Agent
**Version**: 1.0
