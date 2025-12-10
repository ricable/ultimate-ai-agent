# Jujutsu Benchmark Quick Start Guide

**For**: AI Agents & Human Developers
**Created**: 2025-11-09
**Status**: Ready to Use

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites Check

```bash
# Verify tools (run this first)
cd /workspaces/agentic-flow/packages/agentic-jujutsu

# Check Rust
cargo --version  # Should be 1.75+

# Check Git
git --version  # Should be 2.30+

# Install Jujutsu (if not present)
cargo install --git https://github.com/martinvonz/jj.git jj-cli

# Check Docker
docker --version  # Should be 20.10+
docker-compose --version  # Should be 2.0+
```

### Initialize Benchmark Suite

```bash
# Create directory structure
mkdir -p benchmarks/{src,tests,docker,config,templates,scripts,docs,reports}
mkdir -p benchmarks/src/{vcs,worktree,scalability,analysis,security,optimization,learning,reporters}
mkdir -p benchmarks/docs/{algorithms,requirements}

# Initialize Cargo package
cd benchmarks
cat > Cargo.toml << 'EOF'
[package]
name = "agentic-jujutsu-benchmarks"
version = "0.1.0"
edition = "2021"

[dependencies]
criterion = "0.5"
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
chrono = "0.4"

# For Docker integration
bollard = "0.16"

# For reporting
plotters = "0.3"
handlebars = "5.0"
csv = "1.3"

[dev-dependencies]
tempfile = "3.0"
EOF

# Create library root
cat > src/lib.rs << 'EOF'
//! Agentic Jujutsu Benchmark Suite
//!
//! Comprehensive performance benchmarks comparing Jujutsu and Git.

pub mod harness;
pub mod metrics;
pub mod vcs;
pub mod worktree;
pub mod scalability;
pub mod reporters;
EOF
```

---

## 📋 Agent Prompt Library

### Phase 1: Requirements & Architecture

**Prompt 1.1 - Requirements Analysis**:
```
I need you to create a detailed requirements specification for the jujutsu benchmark suite.

Focus on:
1. Define all benchmark scenarios (VCS operations, worktrees, scalability)
2. Specify performance metrics (time, memory, disk I/O, CPU)
3. Set success criteria for each benchmark category
4. Document comparison methodology (jujutsu vs git)

Deliverable: benchmarks/docs/requirements/REQUIREMENTS.md
Use: docs/BENCHMARK_IMPLEMENTATION_PLAN.md as reference
```

**Prompt 1.2 - Architecture Design**:
```
Design the benchmark system architecture for agentic-jujutsu.

Requirements:
1. Component separation (data collection, analysis, reporting)
2. Docker-based reproducible environment
3. Scalable data storage (JSON, AgentDB)
4. Modular benchmark trait system

Deliverable: benchmarks/docs/requirements/ARCHITECTURE.md
Consider: Existing src/wrapper.rs for jujutsu integration
```

### Phase 2: Docker Environment

**Prompt 2.1 - Docker Setup**:
```
Create a Docker environment for reproducible jujutsu vs git benchmarks.

Requirements:
1. Multi-stage Dockerfile with jujutsu + git
2. docker-compose.yml for orchestration
3. Setup/teardown scripts
4. Isolated test repositories (small/medium/large)

Deliverables:
- benchmarks/docker/Dockerfile
- benchmarks/docker/docker-compose.yml
- benchmarks/docker/scripts/setup.sh
- benchmarks/docker/scripts/teardown.sh

Target: Build time <5 minutes, container size <500MB
```

### Phase 3: Core Benchmarks

**Prompt 3.1 - VCS Operation Benchmarks**:
```
Implement comprehensive VCS operation benchmarks.

Benchmark Categories:
1. Commit: Single file (1KB, 1MB, 10MB), 100 files, 1000 files
2. Branch: Create/delete/switch (1, 10, 100 branches)
3. Merge: Fast-forward, 3-way, with conflicts
4. Rebase: Linear, branched, interactive
5. Conflict Resolution: 2-way, 3-way, complex

Implementation:
- benchmarks/src/vcs/commit.rs
- benchmarks/src/vcs/branch.rs
- benchmarks/src/vcs/merge.rs
- benchmarks/src/vcs/rebase.rs
- benchmarks/src/vcs/conflict.rs

Metrics: execution_time_ms, memory_usage_mb, disk_io_operations
Format: JSON output for analysis
```

**Prompt 3.2 - Worktree Benchmarks**:
```
Implement Git worktree vs jujutsu working copy benchmarks.

Scenarios:
1. Create: 1, 10, 100 worktrees/working copies
2. Switch: Context switching between 2, 10, 100 worktrees
3. Sync: Synchronize changes across worktrees
4. Delete: Cleanup 1, 10, 100 worktrees
5. Concurrent: 5, 20 parallel operations

Implementation: benchmarks/src/worktree/*.rs
Metrics: time, memory, disk_space_usage_mb
Compare: git worktree vs jj working copies
```

**Prompt 3.3 - Scalability Tests**:
```
Implement scalability benchmarks for large repositories.

Scalability Matrix:
| Dimension      | Small | Medium | Large  | XLarge  |
|----------------|-------|--------|--------|---------|
| Repo Size      | 1MB   | 100MB  | 1GB    | 10GB    |
| File Count     | 10    | 1,000  | 10,000 | 100,000 |
| History Depth  | 10    | 1,000  | 10,000 | 100,000 |
| Branch Count   | 5     | 50     | 500    | 5,000   |

Implementation: benchmarks/src/scalability/*.rs
Metrics: Linear scaling? Memory usage curves? Performance cliffs?
Visualization: Performance curves for each dimension
```

### Phase 4: Analysis & Learning

**Prompt 4.1 - Performance Profiling**:
```
Implement performance profiling and bottleneck detection.

Tools Integration:
1. cargo-flamegraph for CPU profiling
2. dhat for heap profiling
3. Custom instrumentation for hot paths

Implementation:
- benchmarks/src/optimization/profiler.rs
- benchmarks/src/optimization/hotspots.rs
- benchmarks/src/optimization/recommendations.rs

Output: benchmarks/reports/BOTTLENECKS.md
Goal: Identify top 10 bottlenecks with optimization suggestions
```

**Prompt 4.2 - AgentDB Learning Integration**:
```
Integrate AgentDB for self-learning pattern recognition.

Features:
1. Store benchmark runs as episodes
2. Recognize performance patterns
3. Predict operation costs
4. Suggest optimizations

Schema:
- session_id, operation_type, repo_size, file_count
- execution_time_ms, memory_usage_mb, success
- metadata (git_version, jj_version, system_info)

Implementation: benchmarks/src/learning/*.rs
Success: Pattern recognition accuracy >70%
```

### Phase 5: Reporting & Documentation

**Prompt 5.1 - Report Generation**:
```
Implement comprehensive report generation system.

Report Types:
1. Executive Summary (high-level comparison)
2. Detailed Analysis (per-operation breakdowns)
3. Scalability Report (performance curves)
4. Security Report (vulnerability summary)
5. Learning Report (AI insights)

Formats: Markdown, HTML (with charts), JSON, CSV
Templates: benchmarks/templates/*.hbs
Implementation: benchmarks/src/reporters/*.rs
```

**Prompt 5.2 - Documentation**:
```
Create comprehensive benchmark documentation.

Documents:
1. GETTING_STARTED.md - Quick start guide
2. USAGE_GUIDE.md - Detailed usage instructions
3. PROMPT_LIBRARY.md - AI agent prompts
4. INTERPRETATION_GUIDE.md - How to read reports
5. FAQ.md - Common questions
6. CONTRIBUTING.md - How to add benchmarks

Style: Clear, concise, with examples
Audience: AI agents and human developers
```

---

## 🎯 Milestone Execution Order

### Week 1: Foundation (Milestones 1.1-3.2)

**Day 1-2**: Specification
```bash
Task("Researcher Agent", "Create REQUIREMENTS.md and ARCHITECTURE.md", "researcher")
```

**Day 3-4**: Docker Environment
```bash
Task("Coder Agent", "Implement Docker environment with jj + git", "coder")
```

**Day 5**: Benchmark Harness
```bash
Task("Coder Agent", "Create benchmark trait system and metrics collector", "coder")
```

### Week 2: Core Benchmarks (Milestones 3.3-3.5)

**Day 6-8**: VCS Benchmarks
```bash
Task("Coder Agent 1", "Implement commit and branch benchmarks", "coder")
Task("Coder Agent 2", "Implement merge and rebase benchmarks", "coder")
```

**Day 9-10**: Worktree & Scalability
```bash
Task("Coder Agent 1", "Implement worktree benchmarks", "coder")
Task("Coder Agent 2", "Implement scalability tests", "coder")
```

### Week 3: Analysis (Milestones 4.1-4.4)

**Day 11-13**: Code Quality & Security
```bash
Task("Tester Agent", "Implement quality and security analysis", "tester")
```

**Day 14-15**: Performance & Learning
```bash
Task("Coder Agent", "Implement profiling and AgentDB integration", "coder")
```

### Week 4: Completion (Milestones 5.1-5.4)

**Day 16-17**: Reporting
```bash
Task("Reviewer Agent", "Implement report generators and templates", "reviewer")
```

**Day 18-19**: Documentation
```bash
Task("Reviewer Agent", "Create comprehensive documentation", "reviewer")
```

**Day 20**: Validation
```bash
Task("Tester Agent", "Run full validation suite", "tester")
Task("Reviewer Agent", "Final review and release prep", "reviewer")
```

---

## 🔧 Common Commands

### Build & Test

```bash
# Build benchmark suite
cd benchmarks
cargo build --release

# Run all benchmarks
cargo bench

# Run specific benchmark category
cargo bench --bench vcs_operations
cargo bench --bench worktree_operations
cargo bench --bench scalability

# Run with Docker
docker-compose up -d
docker-compose exec benchmarks cargo bench
```

### Analysis & Profiling

```bash
# Generate flamegraph
cargo flamegraph --bench vcs_operations

# Run with heap profiling
cargo run --features dhat --bench scalability

# Code coverage
cargo tarpaulin --out Html --output-dir coverage/

# Security audit
cargo audit
```

### Report Generation

```bash
# Generate all reports
cargo run --bin generate-reports

# Generate specific report
cargo run --bin generate-reports -- --type executive-summary
cargo run --bin generate-reports -- --type scalability --format html
```

---

## 📊 Expected Outputs

### Benchmark Results (JSON)

```json
{
  "benchmark_id": "commit_single_file_1kb",
  "tool": "jujutsu",
  "scenario": {
    "operation": "commit",
    "file_size": "1KB",
    "file_count": 1
  },
  "metrics": {
    "execution_time_ms": 12.5,
    "memory_usage_mb": 8.2,
    "disk_io_operations": 15
  },
  "system_info": {
    "os": "Linux",
    "cpu": "Intel i7",
    "memory_gb": 16
  }
}
```

### Comparative Report (Markdown)

```markdown
## Commit Operation Performance

| Scenario         | Git (ms) | Jujutsu (ms) | Speedup | Winner   |
|------------------|----------|--------------|---------|----------|
| 1 file (1KB)     | 10       | 12           | 0.83x   | Git      |
| 100 files (1KB)  | 50       | 45           | 1.11x   | Jujutsu  |
| 1000 files (1KB) | 500      | 420          | 1.19x   | Jujutsu  |

**Conclusion**: Jujutsu shows better scaling for multi-file commits.
```

---

## 🐛 Troubleshooting

### Issue: Jujutsu not found

```bash
# Install jujutsu
cargo install --git https://github.com/martinvonz/jj.git jj-cli

# Verify installation
jj --version
```

### Issue: Docker build fails

```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache
```

### Issue: Benchmark timeout

```bash
# Increase timeout in config
# benchmarks/config/benchmarks.yml
timeout_seconds: 600  # 10 minutes instead of default 5
```

### Issue: Out of memory

```bash
# Reduce test size
# benchmarks/config/scenarios.yml
scalability:
  max_repo_size: 1GB  # Reduced from 10GB
  max_file_count: 10000  # Reduced from 100k
```

---

## 📚 Additional Resources

### Documentation

- **Full Plan**: `docs/BENCHMARK_IMPLEMENTATION_PLAN.md`
- **Architecture**: `docs/requirements/ARCHITECTURE.md`
- **Metrics**: `docs/requirements/METRICS_DEFINITION.md`
- **Jujutsu Docs**: https://github.com/martinvonz/jj
- **Git Worktree**: https://git-scm.com/docs/git-worktree

### Tools

- **Criterion**: https://github.com/bheisler/criterion.rs
- **Plotters**: https://github.com/plotters-rs/plotters
- **Flamegraph**: https://github.com/flamegraph-rs/flamegraph
- **AgentDB**: (agentic-flow internal)

---

## 🎓 Success Criteria Checklist

### Phase 1: Specification ✅
- [ ] Requirements documented
- [ ] Architecture designed
- [ ] Metrics defined
- [ ] Scenarios specified

### Phase 2: Infrastructure ✅
- [ ] Docker environment functional
- [ ] Benchmark harness implemented
- [ ] Metrics collector working

### Phase 3: Benchmarks ✅
- [ ] VCS operations benchmarked
- [ ] Worktree operations benchmarked
- [ ] Scalability tests implemented
- [ ] All metrics collected

### Phase 4: Analysis ✅
- [ ] Code quality measured
- [ ] Security audited
- [ ] Bottlenecks identified
- [ ] AgentDB learning integrated

### Phase 5: Completion ✅
- [ ] Reports generated
- [ ] Documentation complete
- [ ] CI/CD integrated
- [ ] Validation passed

---

**Created**: 2025-11-09
**Agent**: code-goal-planner
**For**: AI Agents & Human Developers
**Status**: Ready to Use

**Next Step**: Execute Milestone 1.1 with Researcher Agent

