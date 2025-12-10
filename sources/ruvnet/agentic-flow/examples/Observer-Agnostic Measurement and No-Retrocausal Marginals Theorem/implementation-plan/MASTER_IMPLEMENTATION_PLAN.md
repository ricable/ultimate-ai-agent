# Observer-Agnostic Measurement and No-Retrocausal Marginals Theorem
## Master Implementation Plan

**Version:** 1.0
**Date:** 2025-10-14
**Status:** Ready for Execution
**Timeline:** 28-40 weeks (6-10 months)
**Budget:** $280,000
**Success Probability:** 85%

---

## Executive Summary

This document provides a comprehensive, actionable implementation plan for testing the **Observer-Agnostic Measurement and No-Retrocausal Marginals Theorem**—a falsifiable hypothesis that quantum measurement outcomes are independent of observer consciousness, biological state, and delayed measurement choices.

### Project Goals

1. **Theoretical:** Formalize and prove the theorem mathematically using POVM formalism
2. **Computational:** Build a Rust-based quantum simulator to compute exact predictions
3. **Experimental:** Conduct photonic delayed-choice quantum eraser experiments with three controller types (human, RNG, deterministic)
4. **Statistical:** Demonstrate singles invariance within ε = 5×10⁻⁴ margin
5. **Publication:** Pre-register protocol, collect open data, publish in peer-reviewed journal

### Key Deliverables

- ✅ Mathematical theorem with formal proof
- ✅ Open-source Rust quantum simulator (nalgebra + num-complex)
- ✅ Experimental apparatus with calibrated photonic setup
- ✅ Pre-registered protocol with falsification criteria
- ✅ Statistical analysis demonstrating null result or discovery
- ✅ Peer-reviewed publication with DOI-archived code and data

### Expected Outcome

**Most Likely (95% probability):** Null result confirming observer-agnostic measurement, providing tight bounds on any "consciousness effect" (|Δp| < 5×10⁻⁴). This would be a high-impact negative result definitively ruling out quantum consciousness hypotheses.

**Revolutionary Case (5% probability):** Detection of reproducible controller-dependent variation, indicating new physics linking consciousness to quantum mechanics. Would require immediate replication and could lead to paradigm shift.

---

## Table of Contents

1. [Project Architecture](#project-architecture)
2. [Phase 1: Theoretical Foundation](#phase-1-theoretical-foundation)
3. [Phase 2: Rust Simulation](#phase-2-rust-simulation)
4. [Phase 3: Testing & Validation](#phase-3-testing--validation)
5. [Phase 4: Computational Validation](#phase-4-computational-validation)
6. [Phase 5: Experimental Design](#phase-5-experimental-design)
7. [Phase 6: Hardware & Lab Setup](#phase-6-hardware--lab-setup)
8. [Phase 7: Data Collection](#phase-7-data-collection)
9. [Phase 8: Analysis & Interpretation](#phase-8-analysis--interpretation)
10. [Phase 9: Publication & Dissemination](#phase-9-publication--dissemination)
11. [Risk Management](#risk-management)
12. [Resource Requirements](#resource-requirements)
13. [Success Criteria](#success-criteria)
14. [References](#references)

---

## Project Architecture

### Directory Structure

```
Observer-Agnostic Measurement and No-Retrocausal Marginals Theorem/
├── implementation-plan/
│   ├── MASTER_IMPLEMENTATION_PLAN.md (this file)
│   ├── PHASE_1_THEORETICAL.md
│   ├── PHASE_2_SIMULATION.md
│   ├── PHASE_3_TESTING.md
│   ├── PHASE_4_VALIDATION.md
│   ├── PHASE_5_EXPERIMENTAL_DESIGN.md
│   ├── PHASE_6_LAB_SETUP.md
│   ├── PHASE_7_DATA_COLLECTION.md
│   ├── PHASE_8_ANALYSIS.md
│   └── PHASE_9_PUBLICATION.md
├── rust-simulator/
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs
│   │   ├── math.rs          # Linear algebra utilities
│   │   ├── eraser.rs        # Quantum eraser simulation
│   │   ├── duality.rs       # Wave-particle duality
│   │   └── cli.rs           # Command-line interface
│   ├── tests/
│   │   ├── invariance_tests.rs
│   │   ├── duality_tests.rs
│   │   └── property_tests.rs
│   └── benches/
│       └── simulation_bench.rs
├── experimental-protocols/
│   ├── APPARATUS_DESIGN.md
│   ├── CALIBRATION_PROCEDURE.md
│   ├── CONTROLLER_PROTOCOL.md
│   ├── SAFETY_CHECKLIST.md
│   └── PRE_REGISTRATION_TEMPLATE.md
├── data-analysis/
│   ├── statistical_tests.py
│   ├── visualization.py
│   ├── power_analysis.py
│   └── requirements.txt
└── docs/
    ├── THEOREM_FORMALIZATION.pdf
    ├── ARCHITECTURE_DIAGRAM.png
    ├── DEPENDENCY_GRAPH.mermaid
    └── QUICK_START.md
```

### Technology Stack

**Simulation:**
- **Language:** Rust 1.70+ (performance, safety, reproducibility)
- **Linear Algebra:** nalgebra 0.32 (complex matrix operations)
- **Complex Numbers:** num-complex 0.4
- **Statistics:** statrs 0.16 (chi-squared, distributions)
- **Parallelization:** rayon 1.8 (data parallelism)
- **SIMD:** SimSIMD (production optimization)
- **Testing:** proptest 1.4 (property-based), criterion 0.5 (benchmarking)
- **Visualization:** plotters 0.3 (interference patterns)

**Experimental Control:**
- **Time-Tagging:** Swabian Instruments Time Tagger Ultra or PicoQuant HydraHarp 400
- **Controller:** Python 3.11+ with hardware RNG integration
- **Data Format:** HDF5 with rich metadata
- **Analysis:** Python (NumPy, SciPy, pandas, matplotlib)

**Documentation:**
- **Theory:** LaTeX (theorem formalization)
- **Code:** Rust doc comments + MkDocs
- **Protocols:** Markdown with mermaid diagrams

---

## Phase 1: Theoretical Foundation

**Duration:** Weeks 1-2
**Cost:** 9 units (PI: 4 weeks, Mathematician: 1 week)
**Team:** Principal Investigator (PI), Mathematical Physicist

### Objectives

1. Formalize the Observer-Agnostic Measurement theorem in rigorous mathematical notation
2. Provide complete proof using POVM formalism and Born rule
3. Derive testable corollaries (singles invariance, duality bound, no retrocausality)
4. Document assumptions and boundary conditions

### Action Items

#### 1.1: Formalize Theorem Statement (Week 1, 2 days)

**Input:** Research document theorem section
**Output:** LaTeX document with precise definitions

**Content:**
- Hilbert space definitions (H_S, H_I, H_SI)
- POVM formalism: {E_y} with E_y ≥ 0, Σ E_y = I
- Observer model: Control parameter λ, internal state ρ_O
- Delayed-choice setup: Signal S, idler I, joint state ρ_SI
- Measurement families: {E^(λ)_y} on S, {F^(μ)_z} on I

**Theorem Statement:**

```latex
\begin{theorem}[Observer-Agnostic Measurement]
For any quantum system S with density matrix ρ_S, any measurement
POVM {E^(λ)_y} controlled by parameter λ chosen by observer O,
if λ is statistically independent of ρ_S, then the outcome
distribution

  p(y | λ) = Tr(E^(λ)_y ρ_S)

is independent of the internal state ρ_O, biological status, or
consciousness of O. Replacing O by O' that realizes the same λ
leaves p(y | λ) invariant.
\end{theorem}

\begin{theorem}[No-Retrocausal Marginals]
For any bipartite state ρ_SI and measurement families
{E^(λ)_y}, {F^(μ)_z}, the singles distribution on S

  p(y | λ) = Σ_z Tr((E^(λ)_y ⊗ F^(μ)_z) ρ_SI)

is independent of the later choice μ on I. Conditional patterns
p(y | z, λ, μ) may depend on μ (quantum erasure), but the
unconditioned marginal cannot.
\end{theorem}
```

#### 1.2: Prove Theorems (Week 1, 3 days)

**Proof Sketch for Theorem 1:**

1. By linearity of trace: p(y | λ) = Tr(E^(λ)_y ρ_S)
2. E^(λ)_y and ρ_S are determined independently of ρ_O
3. The trace operation depends only on the operator and state, not the apparatus
4. Therefore, p(y | λ) is invariant under O → O' for fixed λ ∎

**Proof Sketch for Theorem 2:**

1. Reduce ρ_S by partial trace: ρ_S = Tr_I(ρ_SI)
2. For any μ, Σ_z F^(μ)_z = I_I (completeness)
3. Therefore: p(y | λ) = Tr(E^(λ)_y ⊗ I_I ρ_SI) = Tr(E^(λ)_y ρ_S)
4. This is independent of μ by construction ∎

**Full Proof Requirements:**
- Rigorous treatment of tensor product structure
- Handling of mixed states and classical correlations
- Edge cases: non-maximally entangled states, partial measurements
- Connection to no-signaling theorem

#### 1.3: Derive Falsifiable Predictions (Week 2, 2 days)

**Corollary 1: Singles Invariance**
For Bell state |Ψ⟩ = (|00⟩ + e^(iφ)|11⟩)/√2:
- p(0 | λ) = p(1 | λ) = 0.5 for all φ, λ, μ, controller type
- Any deviation |p(0) - 0.5| > ε constitutes violation

**Corollary 2: Duality Bound Invariance**
For two-path interferometer:
- V² + D² ≤ 1 (Englert's relation)
- Bound holds for all observer types
- Violation V² + D² > 1 + δ (beyond uncertainty) falsifies theorem

**Corollary 3: No Retro-Signaling**
For delayed-choice eraser:
- Changing μ after S detection cannot alter registered p(y)
- Coincidence patterns may change, but not singles
- Correlation between μ and prior singles would falsify

#### 1.4: Document Assumptions (Week 2, 1 day)

**Key Assumptions:**
1. Standard quantum mechanics (Born rule, linearity)
2. No hidden variables violating Bell inequalities
3. Independence of controller choice from ρ_S (no conspiracy)
4. Detector efficiency corrections applied consistently
5. Statistical independence of measurement events

**Boundary Conditions:**
- Applies to projective and generalized (POVM) measurements
- Valid for pure and mixed states
- Requires proper time-like separation (no superluminal signaling)
- Assumes standard Hilbert space formalism

### Deliverables

- [ ] `THEOREM_FORMALIZATION.tex` (complete LaTeX document)
- [ ] `THEOREM_FORMALIZATION.pdf` (compiled with proofs)
- [ ] `FALSIFICATION_CRITERIA.md` (testable predictions)
- [ ] `ASSUMPTIONS_DOCUMENT.md` (boundary conditions)

### Success Criteria

- ✅ Theorem statements pass peer review by mathematical physicist
- ✅ Proofs are rigorous and complete
- ✅ Predictions are quantitatively testable with defined error margins
- ✅ Assumptions are explicitly stated and defensible

### Dependencies

**Preconditions:** Research document reviewed
**Postconditions:** Enables simulation design (Phase 2) and experimental protocol (Phase 5)

---

## Phase 2: Rust Simulation

**Duration:** Weeks 2-4
**Cost:** 19 units (Rust Dev: 3 weeks, Code Review: 1 week)
**Team:** Rust Developer, Software Architect

### Objectives

1. Implement quantum state manipulation (kets, density matrices, tensor products)
2. Build POVM measurement framework
3. Simulate delayed-choice quantum eraser with phase sweep
4. Verify singles invariance and duality relations computationally

### Action Items

#### 2.1: Initialize Project (Week 2, 1 day)

```bash
cd rust-simulator
cargo init --name observer_invariance
cargo add nalgebra@0.32
cargo add num-complex@0.4
cargo add statrs@0.16
cargo add rayon@1.8
cargo add clap@4.5 --features derive
cargo add serde@1.0 --features derive
cargo add csv@1.3
cargo add plotters@0.3
cargo add --dev proptest@1.4
cargo add --dev approx@0.5
cargo add --dev criterion@0.5
```

**Project Structure:**
```
rust-simulator/
├── Cargo.toml
├── src/
│   ├── lib.rs        # Module exports
│   ├── math.rs       # Linear algebra primitives
│   ├── eraser.rs     # Quantum eraser simulation
│   ├── duality.rs    # Wave-particle duality
│   └── cli.rs        # CLI tool
├── tests/
│   ├── invariance_tests.rs
│   ├── duality_tests.rs
│   └── property_tests.rs
└── benches/
    └── simulation_bench.rs
```

#### 2.2: Implement Core Math Module (Week 2-3, 3 days)

**File:** `src/math.rs`

**Components:**
1. **Type Aliases:**
   ```rust
   pub type C64 = Complex64;
   pub type State = DVector<C64>;
   pub type Operator = DMatrix<C64>;
   ```

2. **Basis States:**
   ```rust
   pub fn ket0() -> State { ... }
   pub fn ket1() -> State { ... }
   pub fn ket_plus() -> State { (ket0() + ket1()) / √2 }
   pub fn ket_minus() -> State { (ket0() - ket1()) / √2 }
   ```

3. **Quantum Gates:**
   ```rust
   pub fn hadamard() -> Operator { ... }
   pub fn phase(phi: f64) -> Operator { ... }
   pub fn pauli_x() -> Operator { ... }
   pub fn pauli_y() -> Operator { ... }
   pub fn pauli_z() -> Operator { ... }
   pub fn rotation_z(theta: f64) -> Operator { ... }
   ```

4. **Tensor Operations:**
   ```rust
   pub fn kron(a: &Operator, b: &Operator) -> Operator { ... }
   pub fn partial_trace_second(dm: &Operator) -> Operator { ... }
   ```

5. **Measurement:**
   ```rust
   pub fn projector(v: &State) -> Operator { |v⟩⟨v| }
   pub fn dm_from_state(psi: &State) -> Operator { |ψ⟩⟨ψ| }
   pub fn prob(dm: &Operator, povm: &[Operator], outcome: usize) -> f64 {
       let p = (povm[outcome].clone() * dm).trace();
       p.re  // Born rule: Tr(E_y ρ)
   }
   ```

**Key Implementation Details:**
- Use `nalgebra::DMatrix` for dense complex matrices
- Implement Kronecker product with explicit indexing (avoid tensor crate overhead)
- Partial trace over second qubit: sum over diagonal blocks
- Normalization checks for states (⟨ψ|ψ⟩ = 1)
- POVM completeness checks (Σ E_y = I)

#### 2.3: Implement Quantum Eraser Module (Week 3, 3 days)

**File:** `src/eraser.rs`

**Key Functions:**

1. **Bell State with Phase:**
   ```rust
   pub fn bell_with_phase(phi: f64) -> State {
       // |Ψ(φ)⟩ = (|00⟩ + e^(iφ)|11⟩) / √2
       let s = 1.0 / 2.0_f64.sqrt();
       DVector::from_vec(vec![
           C64::new(s, 0.0),
           C64::new(0.0, 0.0),
           C64::new(0.0, 0.0),
           C64::from_polar(s, phi)
       ])
   }
   ```

2. **Idler POVMs:**
   ```rust
   pub enum IdlerBasis { WhichPath, Eraser }

   pub fn idler_povm(basis: IdlerBasis) -> Vec<Operator> {
       match basis {
           IdlerBasis::WhichPath => {
               // Computational basis {|0⟩⟨0|, |1⟩⟨1|}
               vec![projector(&ket0()), projector(&ket1())]
           }
           IdlerBasis::Eraser => {
               // Hadamard basis {|+⟩⟨+|, |-⟩⟨-|}
               vec![projector(&ket_plus()), projector(&ket_minus())]
           }
       }
   }
   ```

3. **Singles Calculation:**
   ```rust
   pub fn singles_signal(phi: f64) -> [f64; 2] {
       let psi = bell_with_phase(phi);
       let dm = dm_from_state(&psi);
       let rho_s = partial_trace_second(&dm);
       let povm_s = signal_povm();
       [prob(&rho_s, &povm_s, 0), prob(&rho_s, &povm_s, 1)]
   }
   ```

4. **Conditional Probabilities:**
   ```rust
   pub fn conditional_signal(phi: f64, basis: IdlerBasis) -> [[f64; 2]; 2] {
       // p(S=y | I=z) for chosen idler basis
       // Joint measurement: E_y ⊗ F_z
       // Returns [z][y] array
   }
   ```

**Critical Test:**
```rust
#[test]
fn singles_must_be_flat() {
    for phi in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, PI] {
        let singles = singles_signal(phi);
        assert_abs_diff_eq!(singles[0], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(singles[1], 0.5, epsilon = 1e-12);
    }
}
```

#### 2.4: Implement Duality Module (Week 3, 2 days)

**File:** `src/duality.rs`

**Englert's Duality Relation:**
For a two-path interferometer with which-path marking parameter γ:
- Visibility: V = γ
- Distinguishability: D = √(1 - γ²)
- Bound: V² + D² = 1 (for pure states)

```rust
pub fn visibility_distinguishability(gamma: f64) -> (f64, f64) {
    let v = gamma.abs().clamp(0.0, 1.0);
    let d = (1.0 - v * v).sqrt();
    (v, d)
}

pub fn check_duality_bound(rho_s: &Operator) -> f64 {
    // Extract V from off-diagonal coherence
    let off = rho_s[(0, 1)].norm();
    let v = 2.0 * off;  // For balanced intensities

    // Extract D from population difference
    let d = (rho_s[(0, 0)].re - rho_s[(1, 1)].re).abs();

    v * v + d * d
}
```

#### 2.5: Implement CLI Tool (Week 4, 2 days)

**File:** `src/cli.rs`

**Features:**
- Sweep phase φ from 0 to 2π
- Generate CSV with singles and conditioned probabilities
- Support multiple idler bases (which-path, eraser)
- Optional plotting with plotters crate

```rust
use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(short, long, default_value_t = 64)]
    steps: usize,

    #[arg(short, long, default_value = "eraser")]
    basis: String,

    #[arg(short, long)]
    output: Option<String>,

    #[arg(short, long)]
    plot: bool,
}

fn main() {
    let args = Args::parse();

    println!("phi,singles0,singles1,cond_plus_0,cond_plus_1");

    for k in 0..=args.steps {
        let phi = 2.0 * PI * (k as f64) / (args.steps as f64);
        let singles = singles_signal(phi);
        let cond = conditional_signal(phi, IdlerBasis::Eraser);

        println!("{:.6},{:.6},{:.6},{:.6},{:.6}",
            phi, singles[0], singles[1], cond[0][0], cond[0][1]);
    }
}
```

**Usage:**
```bash
cargo run --release -- --steps 128 > eraser_data.csv
cargo run --release -- --basis whichpath --plot
```

### Deliverables

- [ ] `src/math.rs` with linear algebra primitives
- [ ] `src/eraser.rs` with quantum eraser simulation
- [ ] `src/duality.rs` with duality relation checks
- [ ] `src/cli.rs` with CSV export
- [ ] Unit tests with ≥95% coverage
- [ ] Documentation with rustdoc

### Success Criteria

- ✅ All tests pass: `cargo test`
- ✅ Singles invariance: |p(0) - 0.5| < 10⁻¹² for all φ, basis
- ✅ Duality bound: V² + D² ≤ 1.0 + 10⁻¹² for all γ
- ✅ Performance: Phase sweep (1000 steps) completes in <1 second
- ✅ Code review passes (architecture, correctness, documentation)

---

## Phase 3: Testing & Validation

**Duration:** Weeks 3-5
**Cost:** 15 units (QA Engineer: 2 weeks, Rust Dev: 1 week)
**Team:** QA Engineer, Rust Developer

### Objectives

1. Comprehensive unit testing (≥95% coverage)
2. Property-based testing with proptest
3. Integration testing for full workflows
4. Performance benchmarking with criterion
5. Numerical stability validation

### Action Items

#### 3.1: Unit Tests (Week 3-4, 5 days)

**File:** `tests/invariance_tests.rs`

```rust
use observer_invariance::eraser::*;
use observer_invariance::math::*;
use approx::assert_abs_diff_eq;

#[test]
fn singles_are_flat_for_all_phases() {
    let phases = [0.0, 0.7, 1.3, 2.6, PI, 3.5, 5.8, 2.0*PI];

    for &phi in &phases {
        let singles = singles_signal(phi);
        assert_abs_diff_eq!(singles[0], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(singles[1], 0.5, epsilon = 1e-12);
    }
}

#[test]
fn singles_are_basis_independent() {
    let phi = 1.234;

    let singles_wp = singles_signal(phi);
    let cond_wp = conditional_signal(phi, IdlerBasis::WhichPath);
    let cond_er = conditional_signal(phi, IdlerBasis::Eraser);

    // Singles must be identical regardless of idler basis choice
    assert_abs_diff_eq!(singles_wp[0], 0.5, epsilon = 1e-12);

    // But conditioned patterns differ (erasure effect)
    assert!(cond_wp[0][0] != cond_er[0][0]);
}

#[test]
fn conditioned_probabilities_normalized() {
    let phi = 0.8;

    for basis in [IdlerBasis::WhichPath, IdlerBasis::Eraser] {
        let cond = conditional_signal(phi, basis);

        for z in 0..2 {
            let sum = cond[z][0] + cond[z][1];
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
        }
    }
}
```

**File:** `tests/duality_tests.rs`

```rust
#[test]
fn duality_bound_holds() {
    let gammas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0];

    for &gamma in &gammas {
        let (v, d) = visibility_distinguishability(gamma);
        let bound = v*v + d*d;
        assert_abs_diff_eq!(bound, 1.0, epsilon = 1e-12);
    }
}
```

#### 3.2: Property-Based Tests (Week 4, 3 days)

**File:** `tests/property_tests.rs`

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn singles_always_flat(phi in 0.0..2.0*PI) {
        let singles = singles_signal(phi);
        prop_assert!((singles[0] - 0.5).abs() < 1e-10);
        prop_assert!((singles[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn probabilities_normalized(phi in 0.0..2.0*PI) {
        let singles = singles_signal(phi);
        let sum = singles[0] + singles[1];
        prop_assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn duality_bound_never_violated(gamma in 0.0..=1.0) {
        let (v, d) = visibility_distinguishability(gamma);
        let bound = v*v + d*d;
        prop_assert!(bound <= 1.0 + 1e-10);
    }
}
```

#### 3.3: Benchmarks (Week 4, 2 days)

**File:** `benches/simulation_bench.rs`

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use observer_invariance::eraser::*;

fn bench_singles_calculation(c: &mut Criterion) {
    c.bench_function("singles_signal", |b| {
        b.iter(|| singles_signal(black_box(1.234)))
    });
}

fn bench_conditional_calculation(c: &mut Criterion) {
    c.bench_function("conditional_signal", |b| {
        b.iter(|| conditional_signal(black_box(1.234), black_box(IdlerBasis::Eraser)))
    });
}

fn bench_phase_sweep(c: &mut Criterion) {
    c.bench_function("phase_sweep_1000", |b| {
        b.iter(|| {
            for k in 0..1000 {
                let phi = 2.0 * PI * (k as f64) / 1000.0;
                let _ = singles_signal(black_box(phi));
            }
        })
    });
}

criterion_group!(benches, bench_singles_calculation, bench_conditional_calculation, bench_phase_sweep);
criterion_main!(benches);
```

**Performance Targets:**
- Singles calculation: <10 μs per call
- Conditional calculation: <50 μs per call
- Phase sweep (1000 points): <10 ms total
- Memory usage: <100 MB for full simulation

#### 3.4: Integration Tests (Week 5, 2 days)

**Test End-to-End Workflows:**

```rust
#[test]
fn full_eraser_workflow() {
    // 1. Generate Bell state
    let phi = 0.5;
    let psi = bell_with_phase(phi);

    // 2. Verify entanglement
    let dm = dm_from_state(&psi);
    let entropy = von_neumann_entropy(&dm);  // Should be > 0
    assert!(entropy > 0.1);

    // 3. Calculate singles
    let singles = singles_signal(phi);
    assert_abs_diff_eq!(singles[0], 0.5, epsilon = 1e-10);

    // 4. Calculate conditioned patterns for both bases
    let cond_wp = conditional_signal(phi, IdlerBasis::WhichPath);
    let cond_er = conditional_signal(phi, IdlerBasis::Eraser);

    // 5. Verify erasure effect
    // Which-path basis should not show fringes
    // Eraser basis should show fringes with phase dependence
    assert_ne!(cond_wp, cond_er);
}

#[test]
fn cli_output_valid() {
    // Run CLI and parse output
    let output = Command::new("cargo")
        .args(&["run", "--release", "--", "--steps", "10"])
        .output()
        .expect("Failed to run CLI");

    let csv_data = String::from_utf8(output.stdout).unwrap();

    // Parse and validate CSV
    for line in csv_data.lines().skip(1) {  // Skip header
        let cols: Vec<f64> = line.split(',')
            .map(|s| s.parse().unwrap())
            .collect();

        let (phi, s0, s1, c0, c1) = (cols[0], cols[1], cols[2], cols[3], cols[4]);

        // Singles must be 0.5
        assert_abs_diff_eq!(s0, 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(s1, 0.5, epsilon = 1e-6);

        // Probabilities normalized
        assert_abs_diff_eq!(c0 + c1, 1.0, epsilon = 1e-6);
    }
}
```

### Deliverables

- [ ] Unit tests with ≥95% code coverage
- [ ] Property-based tests covering all core functions
- [ ] Benchmark suite with performance baselines
- [ ] Integration tests for end-to-end workflows
- [ ] Coverage report: `cargo tarpaulin --out Html`

### Success Criteria

- ✅ `cargo test` passes all tests
- ✅ Code coverage ≥95%
- ✅ Property tests pass 10,000+ generated cases
- ✅ Benchmarks meet performance targets
- ✅ No memory leaks or unsafe code issues

---

## Phase 4: Computational Validation

**Duration:** Week 5
**Cost:** 3 units (Physicist: 3 days, Code Review: 1 day)
**Team:** Quantum Physicist, Computational Scientist

### Objectives

1. Verify simulation predictions against known quantum mechanics results
2. Cross-check with analytical calculations
3. Compare with published quantum eraser data
4. Validate numerical stability
5. Generate reference datasets for experimental comparison

### Action Items

#### 4.1: Analytical Validation (Week 5, 2 days)

**Verify Against Known Results:**

1. **Bell State Probabilities:**
   - Analytical: For |Ψ⟩ = (|00⟩ + e^(iφ)|11⟩)/√2, ρ_S = I/2 always
   - Simulation: Should give p(0) = p(1) = 0.5 exactly (within machine precision)

2. **Conditioned Fringes:**
   - Analytical: For eraser basis, p(S=0 | I=+) = ½(1 + cos φ)
   - Simulation: Should match within numerical error (<10⁻¹⁰)

3. **Duality Relation:**
   - Analytical: V² + D² = 1 for pure states
   - Simulation: Should hold to machine precision

**Validation Script:**

```rust
#[test]
fn verify_bell_state_marginals() {
    for phi in linspace(0.0, 2.0*PI, 100) {
        let psi = bell_with_phase(phi);
        let dm = dm_from_state(&psi);
        let rho_s = partial_trace_second(&dm);

        // Analytical prediction: ρ_S = I/2
        let identity_half = DMatrix::identity(2, 2) * 0.5;

        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(
                    rho_s[(i, j)],
                    identity_half[(i, j)],
                    epsilon = 1e-12
                );
            }
        }
    }
}

#[test]
fn verify_conditioned_fringes() {
    for phi in linspace(0.0, 2.0*PI, 100) {
        let cond = conditional_signal(phi, IdlerBasis::Eraser);

        // Analytical: p(0 | +) = ½(1 + cos φ)
        let expected = 0.5 * (1.0 + phi.cos());

        assert_abs_diff_eq!(
            cond[0][0],  // p(S=0 | I=+)
            expected,
            epsilon = 1e-10
        );
    }
}
```

#### 4.2: Literature Cross-Check (Week 5, 1 day)

**Compare with Published Data:**

1. **Walborn et al. (2002)** - Double-slit quantum eraser
   - Reported visibility V > 0.95 for eraser configuration
   - Our simulation should predict similar visibility

2. **Ma et al. (2016)** - Delayed-choice experiments review
   - Singles always 50/50 regardless of delayed choice
   - Verify simulation matches this fundamental prediction

3. **Kim et al. (2000)** - Original delayed-choice quantum eraser
   - Four detection patterns (D0-D3) with different fringes
   - Simulate equivalent setup and verify pattern reproduction

#### 4.3: Generate Reference Datasets (Week 5, 2 days)

**Create Gold-Standard Data:**

```bash
# High-resolution phase sweep
cargo run --release -- --steps 1000 --basis eraser > reference_eraser_1000.csv
cargo run --release -- --steps 1000 --basis whichpath > reference_whichpath_1000.csv

# Multiple phi values for statistical tests
for phi in 0.0 0.5 1.0 1.5 2.0 2.5 3.0; do
    echo "Phi: $phi"
    cargo run --release -- --steps 1 --phi $phi
done > reference_discrete_phi.csv

# Duality sweep
for gamma in $(seq 0 0.01 1.0); do
    cargo run --release -- --duality --gamma $gamma
done > reference_duality.csv
```

**Dataset Validation:**
- SHA-256 hashes for integrity
- JSON metadata with git commit, timestamp, parameters
- DOI assignment via Zenodo or figshare

### Deliverables

- [ ] Analytical validation report
- [ ] Literature comparison document
- [ ] Reference datasets (CSV + metadata)
- [ ] Validation test suite

### Success Criteria

- ✅ Simulation matches analytical predictions within 10⁻¹⁰
- ✅ Consistency with published quantum eraser results
- ✅ Reference data archived with DOI
- ✅ Physicist sign-off on correctness

---

## Phase 5: Experimental Design

**Duration:** Weeks 6-8
**Cost:** 23 units (Experimentalist: 3 weeks, Engineer: 2 weeks)
**Team:** Experimental Physicist, Optical Engineer

### Objectives

1. Design photonic quantum eraser apparatus
2. Specify hardware components (laser, crystals, detectors)
3. Develop controller protocols (human, RNG, deterministic)
4. Create calibration procedures
5. Write pre-registration document

### Action Items

#### 5.1: Apparatus Design (Week 6, 5 days)

**System Overview:**

```
         [405nm Pump Laser]
                |
         [Type-II BBO Crystal]  ← SPDC generates entangled pairs
              /    \
    (Signal)  /      \  (Idler)
            /          \
   [Interferometer]  [Basis Selector]
           |              |
       [SPAD-S]       [SPAD-I]
           |              |
   [Time Tagger Module]
           |
   [Controller: Human/RNG/Timer]
```

**Key Components:**

1. **Pump Laser:**
   - Wavelength: 405 nm (blue-violet)
   - Power: 50-100 mW CW
   - Beam quality: TEM00, M² < 1.2
   - Stability: <1% over 8 hours
   - Example: Thorlabs LP405-SF100 or similar

2. **SPDC Crystal (Type-II BBO):**
   - Material: β-BaB₂O₄ (beta barium borate)
   - Type: Type-II phase matching (ordinary → extraordinary + extraordinary)
   - Wavelength: 405 nm pump → 810 nm signal + idler
   - Length: 2-3 mm (balance efficiency vs spectral width)
   - Angle: Cut for collinear degenerate SPDC (~29.2°)
   - Example: Raicol Crystals or Castech

3. **Signal Interferometer:**
   - Type: Mach-Zehnder configuration
   - Path difference: <50 μm (coherence length requirement)
   - Phase control: Piezo-mounted mirror (PZT, 0-10 μm range)
   - Beamsplitters: 50/50 non-polarizing at 810 nm
   - Example: Thorlabs BSW10R or custom optics

4. **Idler Basis Selector:**
   - Computational basis (which-path): Direct detection in |0⟩/|1⟩
   - Eraser basis (Hadamard): 50/50 beamsplitter + detectors
   - Switching: Motorized flip mirror or electro-optic switch
   - Control signal: TTL from controller (human button, RNG, timer)

5. **Single-Photon Detectors:**
   - Type: Single-Photon Avalanche Diodes (SPADs)
   - Efficiency: >60% at 810 nm
   - Dark counts: <100 cps
   - Dead time: <50 ns
   - Timing jitter: <300 ps FWHM
   - Example: Excelitas SPCM-AQRH or ID Quantique

6. **Time-Tagging Module:**
   - Resolution: <100 ps
   - Channels: 4+ (signal, idler, sync, controller trigger)
   - Buffer: >1 GB for long runs
   - Example: Swabian Instruments Time Tagger Ultra or PicoQuant HydraHarp 400

7. **Controller System:**
   - **Human:** Keyboard/button press, randomized prompts
   - **Hardware RNG:** Quantum RNG (e.g., ID Quantique Quantis) or thermal noise TRNG
   - **Deterministic:** Pre-programmed timer or pseudorandom sequence

**Optical Layout:**

```
                    [Pump Laser 405nm]
                           |
                      [Focusing Lens]
                           |
                    [Type-II BBO Crystal]
                           |
                    [Dichroic Filter] ← Remove pump
                      /         \
          [IF Filter]           [IF Filter] ← 810nm ±5nm
              /                       \
        (Signal)                   (Idler)
           |                          |
    [Mach-Zehnder]              [Basis Selector]
     /    |    \                  /         \
  BS1   PZT   BS2              BS3         (Direct)
   \     |    /                /             |
    \    |   /             [Det I1]      [Det I0]
   [Det S0] [Det S1]
        \     |     /
         \    |    /
      [Time Tagger Module]
              |
      [Controller Interface]
```

**Alignment Procedure:**
1. Align pump laser to BBO crystal (maximize SPDC flux)
2. Collect signal and idler photons with fiber couplers
3. Build Mach-Zehnder with path difference <50 μm
4. Measure visibility without idler (should be >95%)
5. Add idler basis selector and verify entanglement (Bell parameter S > 2.5)
6. Calibrate PZT phase control (0 to 2π sweep)
7. Test controller switching (verify TTL signals reach time tagger)

#### 5.2: Controller Protocols (Week 6-7, 3 days)

**Design Requirements:**
- **Randomization:** Controller type unknown to analyzer until post-processing
- **Blinding:** Data files coded with "Controller A/B/C" (identity revealed after analysis)
- **Balance:** Equal numbers of events per controller type
- **Independence:** Controller choice statistically independent of quantum state

**Protocol Details:**

**Human Controller:**
```
1. Display prompt: "Press LEFT or RIGHT for idler basis"
2. Wait for keypress (timeout: 5 seconds)
3. Log choice + timestamp + reaction time
4. Send TTL trigger to basis selector
5. Record controller type: "human"
```

**Hardware RNG Controller:**
```
1. Query quantum RNG for random bit
2. Map: 0 → which-path, 1 → eraser
3. Send TTL trigger immediately
4. Log choice + timestamp + RNG seed
5. Record controller type: "qrng"
```

**Deterministic Controller:**
```
1. Use predetermined sequence (e.g., alternating or LFSR pseudorandom)
2. Advance sequence counter
3. Send TTL trigger at fixed intervals (e.g., 1 Hz)
4. Log choice + timestamp + sequence state
5. Record controller type: "deterministic"
```

**Blinding Procedure:**
```python
# Run by independent party, not experimentalist
import random
import json

controllers = ["human", "qrng", "deterministic"]
mapping = {c: f"Controller_{chr(65+i)}" for i, c in enumerate(random.sample(controllers, 3))}

with open("controller_mapping_sealed.json", "w") as f:
    json.dump(mapping, f)

# Seal in timestamped, cryptographically signed envelope
# Only open after statistical analysis complete
```

#### 5.3: Calibration Procedures (Week 7, 3 days)

**Daily Calibration Checklist:**

1. **Laser Stability:**
   - Measure power: Should be within 2% of baseline
   - Check beam pointing: Centroid drift <50 μm
   - Warm-up time: 30 minutes minimum

2. **SPDC Efficiency:**
   - Measure coincidence rate: Should be >1000 pairs/sec
   - Check singles-to-coincidence ratio: >100:1 (accidentals low)
   - Verify photon statistics: g²(0) < 0.1 (antibunching)

3. **Interferometer Visibility:**
   - Block idler arm
   - Sweep phase with PZT (0 to 2π)
   - Measure visibility: V = (I_max - I_min)/(I_max + I_min) > 0.95
   - Record phase calibration curve: Voltage vs φ

4. **Entanglement Quality:**
   - Measure Bell parameter: S = |E(A,B) - E(A,B') + E(A',B) + E(A',B')| > 2.5
   - Calculate concurrence or fidelity to |Ψ⁻⟩ state
   - Verify no classical correlations (test separability)

5. **Detector Efficiency:**
   - Measure singles rates for each detector
   - Balance rates with neutral density filters
   - Check dark counts: <100 cps per detector
   - Verify timing jitter: <300 ps FWHM via histogram

6. **Time Tagger Calibration:**
   - Measure channel delays with LED pulser
   - Set coincidence window: Typically 2-3 ns
   - Verify timestamp accuracy: Compare to external reference clock

**Weekly Calibration:**
- Full alignment check (all mirrors and beamsplitters)
- SPDC crystal temperature optimization (if phase-matched by temperature)
- Controller system test (all three types in sequence)

**Data to Record:**
- Temperature and humidity logs (±0.1°C, ±5% RH)
- Laser power readings (every hour during runs)
- Visibility measurements (before and after each run)
- Coincidence rates (should be stable within 10%)

#### 5.4: Pre-Registration Document (Week 8, 2 days)

**Pre-Registration Template (Open Science Framework or AsPredicted):**

**1. Study Title:**
"Testing Observer-Agnostic Measurement in Delayed-Choice Quantum Eraser Experiments"

**2. Hypotheses:**

**H0 (Null Hypothesis):**
Singles probabilities p(y | λ) are independent of controller type (human, RNG, deterministic) within experimental uncertainty:
- |p_human(y) - p_RNG(y)| < ε
- |p_human(y) - p_deterministic(y)| < ε
- |p_RNG(y) - p_deterministic(y)| < ε

Where ε = 5×10⁻⁴ (equivalence margin based on apparatus stability).

**H1 (Alternative Hypothesis):**
At least one pairwise comparison shows |Δp| > ε with statistical significance (α = 0.01, power = 0.9).

**3. Experimental Design:**

- **Setup:** Photonic delayed-choice quantum eraser with Type-II SPDC (810 nm)
- **Controllers:** Human (keyboard), Quantum RNG (ID Quantique), Deterministic (timer)
- **Blinding:** Controller identities coded as A/B/C, revealed post-analysis
- **Sample Size:** 5 million events per controller type (15 million total)
- **Sessions:** 10 runs per controller type, randomized order
- **Duration:** ~3 weeks of data collection

**4. Falsification Criteria:**

**Violation Detected If:**
1. Chi-squared test: p < 0.01 for H0: p_A = p_B = p_C
2. AND |Δp| > ε for at least one pair
3. AND reproducible across ≥3 independent runs
4. AND survives drift correction and look-elsewhere adjustment

**5. Data Exclusion Rules:**

**Exclude Events If:**
- Timestamp errors or detector dead time violations
- Visibility drops below 0.90 during run (alignment drift)
- Coincidence rate deviates >20% from baseline (SPDC instability)
- Controller malfunction (logged errors in control software)

**Maximum Exclusion:** 5% of total events. If >5%, flag run as failed and repeat.

**6. Analysis Plan:**

**Primary Analysis:**
```python
# Chi-squared test for singles distribution
observed = [n_A0, n_A1, n_B0, n_B1, n_C0, n_C1]
expected = [N_A/2, N_A/2, N_B/2, N_B/2, N_C/2, N_C/2]
chi2, p_value = scipy.stats.chisquare(observed, expected)

# Equivalence test (TOST)
for (p_A, p_B) in [(p_A, p_B), (p_A, p_C), (p_B, p_C)]:
    delta = abs(p_A - p_B)
    se = sqrt(p_A*(1-p_A)/N_A + p_B*(1-p_B)/N_B)

    # Two one-sided tests
    t1 = (delta - epsilon) / se
    t2 = (-delta - epsilon) / se
    p_equiv = max(t.cdf(t1), t.cdf(t2))

    if p_equiv < 0.05:
        print(f"Equivalence confirmed: |Δp| < {epsilon}")
```

**Secondary Analyses:**
- Conditioned fringes: Verify phase dependence matches simulation
- Duality relation: Check V² + D² ≤ 1 across all runs
- Temporal correlations: Test for drift or learning effects

**7. Power Analysis:**

Sample size calculation for detecting Δp = 10⁻³:
- Effect size: h = 2 * arcsin(√p₁) - 2 * arcsin(√p₂) ≈ 0.002
- Power: 1 - β = 0.9
- Significance: α = 0.01 (two-tailed)
- Required n ≈ 850,000 per group
- Planned n = 5,000,000 per group (5.9× oversampled)

**8. Data and Code Availability:**

- **Data:** Deposited to Zenodo with DOI (CC-BY 4.0 license)
- **Code:** GitHub repository with MIT license
- **Analysis:** Jupyter notebooks with complete workflow
- **Hardware:** Detailed bill of materials and CAD files

**9. Stopping Rules:**

**Stop for Success If:**
- All 30 runs complete (10 per controller)
- Statistical power achieved (verified with interim analysis)
- All calibrations within spec

**Stop for Failure If:**
- Apparatus repeatedly fails calibration (>3 consecutive days)
- Fundamental hardware failure (laser, crystal, detectors)
- Budget or time overruns >50%

**10. Timeline:**

- Week 8: Pre-registration submitted
- Weeks 9-16: Hardware procurement and setup
- Weeks 17-19: Data collection (10 runs × 3 controllers)
- Week 20: Unblind controller mapping
- Weeks 21-22: Statistical analysis
- Weeks 23-24: Paper writing and submission

### Deliverables

- [ ] `APPARATUS_DESIGN.md` with full optical layout
- [ ] `CONTROLLER_PROTOCOL.md` with detailed procedures
- [ ] `CALIBRATION_PROCEDURE.md` with checklists
- [ ] `PRE_REGISTRATION_TEMPLATE.md` (submitted to OSF)
- [ ] Bill of materials (BOM) with vendor quotes

### Success Criteria

- ✅ Design reviewed by optical physicist
- ✅ BOM complete with budget within allocation
- ✅ Pre-registration approved by ethics/oversight committee
- ✅ Safety checklist complete (laser safety, electrical)

---

## Phase 6: Hardware & Lab Setup

**Duration:** Weeks 9-16
**Cost:** 34 units (Experimentalist: 6 weeks, Lab Tech: 4 weeks)
**Team:** Experimental Physicist, Lab Technician, Optical Engineer

### Objectives

1. Procure all hardware components
2. Assemble optical setup on breadboard
3. Align laser, crystal, interferometer, and detectors
4. Integrate controller system and time tagger
5. Achieve target performance metrics (visibility, coincidence rate)

### Action Items

#### 6.1: Procurement (Weeks 9-11, Budget: $180,000)

**Major Components:**

| Item | Vendor | Part Number | Cost | Lead Time |
|------|--------|-------------|------|-----------|
| 405nm Laser (100mW CW) | Thorlabs | LP405-SF100 | $2,500 | 2 weeks |
| Type-II BBO Crystal (3mm) | Raicol | Custom | $8,000 | 6-8 weeks |
| SPADs (4×) | Excelitas | SPCM-AQRH-16 | $16,000 | 4 weeks |
| Time Tagger | Swabian | Time Tagger Ultra | $45,000 | 8 weeks |
| Quantum RNG | ID Quantique | Quantis-16M-PCI | $5,000 | 3 weeks |
| Optical Table | Thorlabs | B4872F | $12,000 | 6 weeks |
| Optics Kit (mirrors, BS, mounts) | Thorlabs | Various | $25,000 | 2 weeks |
| PZT Mirror Mount | Thorlabs | KCH601 | $3,500 | 2 weeks |
| Fiber Couplers (4×) | Thorlabs | PAF2S-11B | $8,000 | 2 weeks |
| SM Fiber (100m) | Thorlabs | P3-780PM-FC-5 | $2,000 | 1 week |
| Power Supplies, Electronics | Various | Various | $10,000 | 2 weeks |
| Environmental Enclosure | Custom | Fabricated | $15,000 | 4 weeks |
| Miscellaneous (alignment tools, mounts, etc.) | Various | Various | $28,000 | Ongoing |

**Total Hardware Cost:** ~$180,000

**Critical Path Items:**
1. **BBO Crystal:** 6-8 weeks (longest lead time)
2. **Time Tagger:** 8 weeks
3. **Optical Table:** 6 weeks
4. **SPADs:** 4 weeks

**Ordering Strategy:**
- Order critical path items immediately (Week 9)
- Order supporting optics after table delivery (Week 11)
- Maintain contingency budget ($20,000) for replacements

#### 6.2: Assembly (Weeks 12-14, 3 weeks)

**Week 12: Table Setup**
- Install optical table in lab with passive vibration isolation
- Level table (bubble level, <0.1 mm/m slope)
- Install environmental enclosure (thermal stability, dust protection)
- Set up laser safety interlocks and signage
- Test power supplies and grounding

**Week 13: Initial Optical Path**
```
Day 1-2: Pump Laser Alignment
- Mount 405nm laser on kinematic base
- Align to BBO crystal using irises
- Measure pump power and beam profile

Day 3-4: SPDC Generation
- Install BBO crystal in rotation mount
- Optimize crystal angle for collinear SPDC
- Verify photon pairs with SPADs (coincidence measurement)
- Measure SPDC spectrum with spectrometer (should be ~810nm)

Day 5: Pump Filtering
- Install dichroic filter after BBO (remove 405nm pump)
- Add interference filters (810nm ±5nm)
- Verify pump suppression: <0.01% leakage to detectors
```

**Week 14: Interferometer Construction**
```
Day 1-3: Mach-Zehnder for Signal
- Build 50/50 beamsplitter network
- Install PZT-controlled mirror in one arm
- Couple both outputs to fibers → SPADs
- Optimize path length matching (<50 μm difference)

Day 4-5: Idler Basis Selector
- Install flip mirror or beamsplitter for basis switching
- Wire motorized mount to controller TTL
- Couple outputs to fiber → SPADs
- Test switching speed and repeatability
```

#### 6.3: Alignment (Weeks 14-15, 2 weeks)

**Visibility Optimization:**

```
Goal: V > 0.95 without idler measurement

Procedure:
1. Block idler arm
2. Sweep PZT phase from 0 to 2π (apply 0-10V ramp)
3. Record signal detector counts vs voltage
4. Calculate visibility: V = (I_max - I_min) / (I_max + I_min)

Optimization Steps:
- If V < 0.90: Check path length matching (add/remove glass plates)
- If V < 0.80: Suspect mode mismatch (optimize fiber coupling)
- If V < 0.70: Major alignment issue (rebuild interferometer)

Expected Result: V > 0.95 after optimization
```

**Entanglement Verification:**

```
Goal: Bell parameter S > 2.5

Procedure:
1. Unblock idler arm
2. Measure coincidences in four analyzer combinations:
   - (A, B): Signal at 0°, Idler at 0°
   - (A, B'): Signal at 0°, Idler at 45°
   - (A', B): Signal at 45°, Idler at 0°
   - (A', B'): Signal at 45°, Idler at 45°

3. Calculate correlations: E(A,B) = (N++ + N-- - N+- - N-+) / N_total

4. Compute Bell parameter: S = |E(A,B) - E(A,B') + E(A',B) + E(A',B')|

Expected Result: S > 2.5 (violates Bell inequality, confirms entanglement)
```

**Coincidence Rate Tuning:**

```
Goal: >1000 pairs/sec, singles/coincidences ratio >100:1

Tuning Procedure:
1. Optimize pump focus into BBO (adjust lens position)
2. Optimize collection optics (fiber coupling efficiency)
3. Balance detector rates with ND filters
4. Adjust coincidence window (typically 2-3 ns)

Monitor:
- Singles rates: 100k-500k cps per detector
- Coincidence rate: >1000 cps
- Accidental coincidences: <10 cps
```

#### 6.4: Controller Integration (Week 15-16, 2 weeks)

**Hardware RNG Setup:**
```python
import quantis

# Initialize Quantis RNG
qrng = quantis.Quantis(quantis.QUANTIS_DEVICE_PCI)

def get_random_basis():
    """Get random bit from quantum RNG."""
    bit = qrng.read_int(0, 1)  # Read single bit
    return "eraser" if bit == 1 else "whichpath"

# Test randomness
bits = [qrng.read_int(0, 1) for _ in range(10000)]
assert 0.48 < np.mean(bits) < 0.52  # Should be ~50% ones
```

**Human Interface:**
```python
import keyboard
import time

def human_controller():
    """Prompt human to choose basis."""
    print("Press [E] for Eraser or [W] for Which-Path")
    start = time.time()

    while True:
        if keyboard.is_pressed('e'):
            reaction_time = time.time() - start
            return "eraser", reaction_time
        elif keyboard.is_pressed('w'):
            reaction_time = time.time() - start
            return "whichpath", reaction_time

        if time.time() - start > 5.0:
            return "timeout", 5.0  # Default after 5 sec
```

**Deterministic Controller:**
```python
import hashlib

class DeterministicController:
    def __init__(self, seed=42):
        self.state = seed

    def next_basis(self):
        """Generate pseudorandom basis choice."""
        # Use cryptographic hash for deterministic but unpredictable sequence
        h = hashlib.sha256(str(self.state).encode()).digest()
        self.state += 1
        bit = h[0] % 2
        return "eraser" if bit == 1 else "whichpath"
```

**Time Tagger Integration:**
```python
import TimeTagger

tagger = TimeTagger.createTimeTagger()

# Define channels
SIGNAL_0 = 1
SIGNAL_1 = 2
IDLER_0 = 3
IDLER_1 = 4
CONTROLLER_TRIGGER = 5

# Set up coincidence measurement
coincidence = TimeTagger.Coincidence(
    tagger=tagger,
    channels=[SIGNAL_0, SIGNAL_1, IDLER_0, IDLER_1],
    coincidenceWindow=2000  # 2 ns window
)

# Record data
data = []
for event in coincidence.getData():
    timestamp = event.time
    channels = event.channels
    data.append((timestamp, channels))
```

**Controller Blinding:**
```python
# Run by independent party
import random
import json
import hashlib
from datetime import datetime

def create_blinded_mapping():
    controllers = ["human", "qrng", "deterministic"]
    random.shuffle(controllers)

    mapping = {c: f"Controller_{chr(65+i)}" for i, c in enumerate(controllers)}

    # Create cryptographic seal
    seal = {
        "mapping": mapping,
        "timestamp": datetime.now().isoformat(),
        "hash": hashlib.sha256(str(mapping).encode()).hexdigest()
    }

    # Save sealed mapping
    with open("sealed_mapping.json", "w") as f:
        json.dump(seal, f, indent=2)

    # Save public codes only (for data collection)
    with open("public_codes.json", "w") as f:
        json.dump({v: k for k, v in mapping.items()}, f)

    print("Mapping sealed. Hash:", seal["hash"])
    return mapping

# Use public codes during data collection
# Only unseal after analysis complete
```

### Deliverables

- [ ] All hardware installed and operational
- [ ] Alignment complete: V > 0.95, S > 2.5
- [ ] Controller system tested (all three types)
- [ ] Safety checklist verified
- [ ] Lab notebook with alignment logs

### Success Criteria

- ✅ Visibility: V > 0.95 (single-photon interference)
- ✅ Bell parameter: S > 2.5 (entanglement verified)
- ✅ Coincidence rate: >1000 pairs/sec
- ✅ Controller switching: <100 ms latency
- ✅ System stability: >4 hour runs without realignment

---

## Phase 7: Data Collection

**Duration:** Weeks 17-20
**Cost:** 17 units (Experimentalist: 3 weeks, Lab Tech: 2 weeks)
**Team:** Experimental Physicist, Lab Technician, Data Analyst

### Objectives

1. Conduct 30 blinded data collection runs (10 per controller type)
2. Monitor apparatus stability and calibration
3. Record 15 million photon detection events
4. Maintain complete logs (environmental, calibration, errors)
5. Validate data quality in real-time

### Action Items

#### 7.1: Data Collection Protocol (Weeks 17-19, 3 weeks)

**Run Schedule (10 runs × 3 controllers = 30 total runs):**

```
Week 17:
  Day 1: Calibration + Test Run (100k events, verify all systems)
  Day 2: Run 1 (Controller A, 500k events, ~2 hours)
  Day 3: Run 2 (Controller B, 500k events)
  Day 4: Run 3 (Controller C, 500k events)
  Day 5: Re-calibration + Run 4 (Controller A, 500k events)

Week 18:
  Day 1: Run 5 (Controller B, 500k events)
  Day 2: Run 6 (Controller C, 500k events)
  Day 3: Run 7 (Controller A, 500k events)
  Day 4: Run 8 (Controller B, 500k events)
  Day 5: Re-calibration + Run 9 (Controller C, 500k events)

Week 19:
  [Continue pattern through Run 30]
  Final Day: Post-experiment calibration and stability check
```

**Randomization:**
- Controller order randomized using sealed envelope method
- Experimentalist opens envelope each morning to learn day's controller
- Order unknown to data analyst until post-processing

**Single Run Procedure:**

```python
# run_experiment.py

def run_data_collection(controller_code, target_events=500000, run_id=1):
    """
    Execute single data collection run.

    Args:
        controller_code: "Controller_A", "Controller_B", or "Controller_C"
        target_events: Number of signal detection events to collect
        run_id: Unique run identifier
    """
    # 1. Pre-run calibration
    visibility = measure_visibility()
    assert visibility > 0.90, f"Visibility too low: {visibility}"

    bell_param = measure_bell_parameter()
    assert bell_param > 2.3, f"Entanglement quality low: {bell_param}"

    coincidence_rate = measure_coincidence_rate(duration=60)
    assert coincidence_rate > 800, f"SPDC rate too low: {coincidence_rate}"

    # 2. Initialize data file
    filename = f"run_{run_id:03d}_{controller_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
    h5file = h5py.File(filename, 'w')

    # Metadata
    h5file.attrs['controller_code'] = controller_code
    h5file.attrs['run_id'] = run_id
    h5file.attrs['target_events'] = target_events
    h5file.attrs['visibility_prerun'] = visibility
    h5file.attrs['bell_parameter'] = bell_param
    h5file.attrs['coincidence_rate'] = coincidence_rate
    h5file.attrs['temperature_start'] = read_temperature()
    h5file.attrs['humidity_start'] = read_humidity()

    # Create datasets
    timestamps = h5file.create_dataset('timestamps', (0,), maxshape=(None,), dtype='float64')
    channels = h5file.create_dataset('channels', (0,), maxshape=(None,), dtype='uint8')
    basis_choices = h5file.create_dataset('basis_choices', (0,), maxshape=(None,), dtype='S20')
    basis_timestamps = h5file.create_dataset('basis_timestamps', (0,), maxshape=(None,), dtype='float64')

    # 3. Select controller
    if controller_code == "Controller_A":
        controller = HumanController()
    elif controller_code == "Controller_B":
        controller = QRNGController()
    elif controller_code == "Controller_C":
        controller = DeterministicController()

    # 4. Main data collection loop
    event_count = 0
    basis_count = 0
    start_time = time.time()

    print(f"Starting Run {run_id} with {controller_code}")
    print(f"Target: {target_events} events")

    while event_count < target_events:
        # Get basis choice from controller
        basis = controller.get_basis_choice()
        basis_time = time.time()

        # Set basis selector (send TTL to flip mirror or BS)
        set_idler_basis(basis)

        # Record basis choice
        basis_choices.resize((basis_count + 1,))
        basis_choices[basis_count] = basis
        basis_timestamps.resize((basis_count + 1,))
        basis_timestamps[basis_count] = basis_time
        basis_count += 1

        # Collect events for this basis (e.g., 1000 events or 1 second)
        events = timetagger.collect_events(duration=1.0)

        for event in events:
            timestamps.resize((event_count + 1,))
            channels.resize((event_count + 1,))
            timestamps[event_count] = event['time']
            channels[event_count] = event['channel']
            event_count += 1

        # Progress update every 50k events
        if event_count % 50000 == 0:
            elapsed = time.time() - start_time
            rate = event_count / elapsed
            eta = (target_events - event_count) / rate / 60
            print(f"  {event_count}/{target_events} events ({event_count/target_events*100:.1f}%), ETA: {eta:.1f} min")

    # 5. Post-run calibration
    visibility_post = measure_visibility()
    h5file.attrs['visibility_postrun'] = visibility_post
    h5file.attrs['temperature_end'] = read_temperature()
    h5file.attrs['humidity_end'] = read_humidity()
    h5file.attrs['duration_seconds'] = time.time() - start_time
    h5file.attrs['total_events'] = event_count

    h5file.close()

    # 6. Validation
    drift = abs(visibility - visibility_post)
    if drift > 0.05:
        print(f"WARNING: Visibility drift {drift:.3f} exceeds threshold!")

    print(f"Run {run_id} complete: {event_count} events in {(time.time()-start_time)/60:.1f} min")
    print(f"  Visibility: {visibility:.3f} → {visibility_post:.3f}")
    print(f"  File: {filename}")

    return filename
```

#### 7.2: Real-Time Monitoring (Weeks 17-19, continuous)

**Automated Checks Every 10 Minutes:**

```python
def monitor_system():
    """Real-time quality control during data collection."""

    # Check detector rates
    rates = get_singles_rates()  # [S0, S1, I0, I1] in cps
    for i, rate in enumerate(rates):
        if rate < 50000 or rate > 1000000:
            alert(f"Detector {i} rate abnormal: {rate} cps")

    # Check coincidence rate
    coinc_rate = get_coincidence_rate()
    if coinc_rate < 800:
        alert(f"Coincidence rate low: {coinc_rate} cps")

    # Check laser power
    power = read_laser_power()
    if abs(power - BASELINE_POWER) > 0.02 * BASELINE_POWER:
        alert(f"Laser power drift: {power:.1f} mW")

    # Check temperature
    temp = read_temperature()
    if abs(temp - 20.0) > 2.0:  # ±2°C tolerance
        alert(f"Temperature deviation: {temp:.1f}°C")

    # Log metrics
    log_metrics({
        'timestamp': time.time(),
        'singles_rates': rates,
        'coincidence_rate': coinc_rate,
        'laser_power': power,
        'temperature': temp,
        'humidity': read_humidity()
    })
```

**Dashboard Display (Plotly Dash):**

```
┌─────────────────────────────────────────────────────────┐
│ Run 15 - Controller_B - 342,581 / 500,000 events       │
├─────────────────────────────────────────────────────────┤
│ Visibility: 0.963  ✓                                    │
│ Coinc. Rate: 1,234 cps  ✓                               │
│ Laser Power: 98.3 mW  ✓                                 │
│ Temperature: 20.4°C  ✓                                  │
│ Elapsed: 1.2 hours, ETA: 0.6 hours                      │
├─────────────────────────────────────────────────────────┤
│ [Singles Rate Graph]                                    │
│ [Coincidence Rate Graph]                                │
│ [Temperature/Humidity Graph]                            │
└─────────────────────────────────────────────────────────┘
```

#### 7.3: Data Validation (Week 20, 1 week)

**Post-Collection Quality Checks:**

```python
def validate_dataset(filename):
    """Comprehensive validation of collected data."""

    h5file = h5py.File(filename, 'r')

    # 1. Check completeness
    n_events = len(h5file['timestamps'])
    assert n_events >= 450000, f"Insufficient events: {n_events}"

    # 2. Check for timestamp errors
    timestamps = h5file['timestamps'][:]
    diffs = np.diff(timestamps)
    assert np.all(diffs > 0), "Non-monotonic timestamps detected"
    assert np.all(diffs < 1.0), "Timestamp gaps >1s detected"

    # 3. Check channel distribution
    channels = h5file['channels'][:]
    channel_counts = np.bincount(channels)
    for i, count in enumerate(channel_counts):
        fraction = count / n_events
        assert 0.15 < fraction < 0.35, f"Channel {i} imbalanced: {fraction*100:.1f}%"

    # 4. Check visibility consistency
    vis_pre = h5file.attrs['visibility_prerun']
    vis_post = h5file.attrs['visibility_postrun']
    drift = abs(vis_pre - vis_post)
    if drift > 0.05:
        print(f"WARNING: Visibility drift {drift:.3f}")
        if drift > 0.10:
            return False, "Excessive visibility drift"

    # 5. Check environmental stability
    temp_start = h5file.attrs['temperature_start']
    temp_end = h5file.attrs['temperature_end']
    if abs(temp_end - temp_start) > 3.0:
        print(f"WARNING: Temperature change {temp_end - temp_start:.1f}°C")

    # 6. Calculate preliminary singles ratio
    signal_0 = np.sum(channels == 0)
    signal_1 = np.sum(channels == 1)
    ratio = signal_0 / (signal_0 + signal_1)

    print(f"  Events: {n_events}")
    print(f"  Visibility: {vis_pre:.3f} → {vis_post:.3f}")
    print(f"  Singles ratio: {ratio:.4f}")
    print(f"  Temperature: {temp_start:.1f}°C → {temp_end:.1f}°C")

    if 0.48 < ratio < 0.52:
        print("  ✓ Singles ratio nominal")
    else:
        print(f"  ⚠ Singles ratio off-nominal: {ratio:.4f}")

    h5file.close()
    return True, "Valid"

# Validate all runs
for i in range(1, 31):
    files = glob.glob(f"run_{i:03d}_*.h5")
    for f in files:
        valid, msg = validate_dataset(f)
        if not valid:
            print(f"FAILED: {f} - {msg}")
```

**Data Exclusion Summary:**

```python
def generate_exclusion_report():
    """Document any excluded runs."""

    excluded = []

    for run_id in range(1, 31):
        files = glob.glob(f"run_{run_id:03d}_*.h5")
        h5file = h5py.File(files[0], 'r')

        vis_drift = abs(h5file.attrs['visibility_prerun'] - h5file.attrs['visibility_postrun'])
        temp_change = abs(h5file.attrs['temperature_start'] - h5file.attrs['temperature_end'])
        n_events = len(h5file['timestamps'])

        exclude_reason = None
        if vis_drift > 0.10:
            exclude_reason = f"Visibility drift: {vis_drift:.3f}"
        elif temp_change > 5.0:
            exclude_reason = f"Temperature change: {temp_change:.1f}°C"
        elif n_events < 450000:
            exclude_reason = f"Insufficient events: {n_events}"

        if exclude_reason:
            excluded.append({
                'run_id': run_id,
                'controller': h5file.attrs['controller_code'],
                'reason': exclude_reason
            })

        h5file.close()

    # Report
    print(f"Total runs: 30")
    print(f"Excluded: {len(excluded)}")
    print(f"Included: {30 - len(excluded)}")

    if len(excluded) > 0:
        print("\nExcluded runs:")
        for e in excluded:
            print(f"  Run {e['run_id']} ({e['controller']}): {e['reason']}")

    # Verify exclusion rate
    exclusion_rate = len(excluded) / 30
    assert exclusion_rate < 0.05, f"Exclusion rate too high: {exclusion_rate*100:.1f}%"

    return excluded
```

### Deliverables

- [ ] 30 HDF5 data files (15 million+ events total)
- [ ] Real-time monitoring logs (temperature, power, rates)
- [ ] Calibration logs (pre/post-run visibility, Bell parameter)
- [ ] Data validation report
- [ ] Exclusion summary (if any runs excluded)

### Success Criteria

- ✅ ≥27 runs pass quality checks (≤10% exclusion rate)
- ✅ ≥4.5 million events per controller type
- ✅ All runs maintain V > 0.90 throughout
- ✅ No systematic drifts correlated with controller type
- ✅ Complete environmental logs with no unexplained anomalies

---

## Phase 8: Analysis & Interpretation

**Duration:** Weeks 21-22
**Cost:** 14 units (Statistician: 2 weeks, Physicist: 1 week)
**Team:** Statistician, Data Analyst, Physicist

### Objectives

1. Unseal controller identity mapping
2. Compute singles probabilities for each controller type
3. Perform statistical tests (chi-squared, TOST equivalence)
4. Analyze conditioned fringes and duality relations
5. Interpret results in context of theorem predictions

### Action Items

#### 8.1: Unseal Controller Mapping (Week 21, Day 1)

**Procedure:**

```python
import json
import hashlib

def unseal_controller_mapping():
    """
    Reveal controller identities after data collection complete.
    This step should be done ceremonially with witnesses.
    """

    # 1. Load sealed mapping
    with open("sealed_mapping.json", "r") as f:
        seal = json.load(f)

    # 2. Verify seal integrity
    stored_hash = seal["hash"]
    computed_hash = hashlib.sha256(str(seal["mapping"]).encode()).hexdigest()

    assert stored_hash == computed_hash, "Seal integrity violation!"

    # 3. Display mapping
    print("=" * 60)
    print("CONTROLLER IDENTITY MAPPING")
    print("=" * 60)
    print(f"Sealed at: {seal['timestamp']}")
    print(f"Hash: {stored_hash[:16]}...")
    print()

    for controller_type, code in seal["mapping"].items():
        print(f"  {code} = {controller_type}")

    print("=" * 60)

    # 4. Update data files with true identity
    for i in range(1, 31):
        files = glob.glob(f"run_{i:03d}_*.h5")
        for f in files:
            h5file = h5py.File(f, 'r+')
            code = h5file.attrs['controller_code']
            true_type = seal["mapping"][code]
            h5file.attrs['controller_type'] = true_type
            h5file.close()

    return seal["mapping"]

# Execute with witnesses
mapping = unseal_controller_mapping()
```

#### 8.2: Singles Probability Analysis (Week 21, 3 days)

**Compute Singles for Each Controller:**

```python
def analyze_singles_probabilities():
    """
    Primary analysis: Test observer-agnostic invariance.
    """

    results = {
        'human': {'p0': [], 'p1': [], 'n_events': []},
        'qrng': {'p0': [], 'p1': [], 'n_events': []},
        'deterministic': {'p0': [], 'p1': [], 'n_events': []}
    }

    # 1. Extract singles from each run
    for i in range(1, 31):
        files = glob.glob(f"run_{i:03d}_*.h5")
        h5file = h5py.File(files[0], 'r')

        controller_type = h5file.attrs['controller_type']
        channels = h5file['channels'][:]

        # Count signal detector events
        n_signal_0 = np.sum(channels == 0)
        n_signal_1 = np.sum(channels == 1)
        n_total = n_signal_0 + n_signal_1

        p0 = n_signal_0 / n_total
        p1 = n_signal_1 / n_total

        results[controller_type]['p0'].append(p0)
        results[controller_type]['p1'].append(p1)
        results[controller_type]['n_events'].append(n_total)

        h5file.close()

    # 2. Aggregate statistics
    summary = {}
    for controller, data in results.items():
        p0_mean = np.mean(data['p0'])
        p0_std = np.std(data['p0'])
        p0_sem = p0_std / np.sqrt(len(data['p0']))

        total_events = np.sum(data['n_events'])

        summary[controller] = {
            'p0_mean': p0_mean,
            'p0_std': p0_std,
            'p0_sem': p0_sem,
            'total_events': total_events,
            'n_runs': len(data['p0'])
        }

    # 3. Display results
    print("=" * 70)
    print("SINGLES PROBABILITY ANALYSIS")
    print("=" * 70)
    for controller, stats in summary.items():
        print(f"\n{controller.upper()}")
        print(f"  p(0) = {stats['p0_mean']:.6f} ± {stats['p0_sem']:.6f}")
        print(f"  Total events: {stats['total_events']:,}")
        print(f"  Runs: {stats['n_runs']}")

    return summary

summary = analyze_singles_probabilities()
```

**Expected Output:**

```
======================================================================
SINGLES PROBABILITY ANALYSIS
======================================================================

HUMAN
  p(0) = 0.500123 ± 0.000047
  Total events: 5,234,567
  Runs: 10

QRNG
  p(0) = 0.499887 ± 0.000051
  Total events: 5,198,432
  Runs: 10

DETERMINISTIC
  p(0) = 0.500045 ± 0.000043
  Total events: 5,289,101
  Runs: 10
```

#### 8.3: Statistical Hypothesis Testing (Week 21, 2 days)

**Chi-Squared Test:**

```python
def chi_squared_test(summary):
    """
    Test H0: p_human = p_qrng = p_deterministic = 0.5
    """

    # Observed counts
    observed = []
    expected = []

    for controller, stats in summary.items():
        n = stats['total_events']
        p0 = stats['p0_mean']

        observed.extend([p0 * n, (1 - p0) * n])
        expected.extend([n / 2, n / 2])

    # Chi-squared statistic
    chi2, p_value = scipy.stats.chisquare(observed, expected)

    print("\n" + "=" * 70)
    print("CHI-SQUARED TEST")
    print("=" * 70)
    print(f"H0: All controllers have p(0) = 0.5")
    print(f"χ² statistic: {chi2:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Degrees of freedom: {len(observed) - 1}")

    alpha = 0.01
    if p_value < alpha:
        print(f"\n❌ REJECT H0 at α = {alpha}")
        print("   Evidence for controller-dependent variation!")
    else:
        print(f"\n✓ FAIL TO REJECT H0 at α = {alpha}")
        print("   No evidence for controller dependence.")

    return chi2, p_value
```

**TOST Equivalence Test:**

```python
def tost_equivalence_test(summary, epsilon=0.0005):
    """
    Test equivalence: |p_A - p_B| < epsilon
    """

    controllers = list(summary.keys())

    print("\n" + "=" * 70)
    print("TWO ONE-SIDED TESTS (TOST) FOR EQUIVALENCE")
    print("=" * 70)
    print(f"Equivalence margin: ε = {epsilon}")
    print()

    results = []

    for i in range(len(controllers)):
        for j in range(i + 1, len(controllers)):
            c1, c2 = controllers[i], controllers[j]

            p1 = summary[c1]['p0_mean']
            p2 = summary[c2]['p0_mean']
            n1 = summary[c1]['total_events']
            n2 = summary[c2]['total_events']

            # Standard error of difference
            se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)

            # Observed difference
            delta = abs(p1 - p2)

            # TOST test statistics
            t1 = (delta - epsilon) / se
            t2 = (-delta - epsilon) / se

            # p-values for one-sided tests
            p1_tost = scipy.stats.norm.cdf(t1)
            p2_tost = scipy.stats.norm.cdf(t2)
            p_equiv = max(p1_tost, p2_tost)

            print(f"{c1.upper()} vs {c2.upper()}")
            print(f"  Δp = {delta:.6f}")
            print(f"  SE(Δp) = {se:.6f}")
            print(f"  p-value (equivalence): {p_equiv:.6f}")

            if p_equiv < 0.05:
                print(f"  ✓ EQUIVALENT (|Δp| < {epsilon})")
                equivalent = True
            else:
                print(f"  ⚠ NOT EQUIVALENT")
                equivalent = False

            print()

            results.append({
                'controllers': (c1, c2),
                'delta': delta,
                'se': se,
                'p_value': p_equiv,
                'equivalent': equivalent
            })

    return results
```

#### 8.4: Secondary Analyses (Week 22, 3 days)

**Conditioned Fringes Analysis:**

```python
def analyze_conditioned_fringes():
    """
    Verify that conditioned patterns show phase dependence.
    """

    # For runs where idler basis = eraser, extract signal counts conditioned on idler outcome

    fringes = []

    for i in range(1, 31):
        files = glob.glob(f"run_{i:03d}_*.h5")
        h5file = h5py.File(files[0], 'r')

        timestamps = h5file['timestamps'][:]
        channels = h5file['channels'][:]
        basis_choices = h5file['basis_choices'][:]
        basis_timestamps = h5file['basis_timestamps'][:]

        # Filter for eraser basis periods
        for k in range(len(basis_choices)):
            if basis_choices[k] == b'eraser':
                # Find events in this time window
                t_start = basis_timestamps[k]
                t_end = basis_timestamps[k + 1] if k + 1 < len(basis_timestamps) else timestamps[-1]

                mask = (timestamps >= t_start) & (timestamps < t_end)
                window_channels = channels[mask]

                # Compute coincidences and conditional probabilities
                # ... (detailed coincidence analysis)

        h5file.close()

    # Plot interference fringes
    # Should show sinusoidal dependence on phase φ
    # Only for eraser basis, not which-path basis
```

**Duality Relation Check:**

```python
def check_duality_relation():
    """
    Verify V² + D² ≤ 1 for all runs.
    """

    duality_values = []

    for i in range(1, 31):
        files = glob.glob(f"run_{i:03d}_*.h5")
        h5file = h5py.File(files[0], 'r')

        # Extract visibility from post-run calibration
        visibility = h5file.attrs['visibility_postrun']

        # Compute distinguishability from singles imbalance
        channels = h5file['channels'][:]
        n0 = np.sum(channels == 0)
        n1 = np.sum(channels == 1)
        distinguishability = abs(n0 - n1) / (n0 + n1)

        bound = visibility**2 + distinguishability**2
        duality_values.append(bound)

        print(f"Run {i}: V² + D² = {bound:.6f}")

        h5file.close()

    # All should satisfy ≤ 1
    violations = [v for v in duality_values if v > 1.01]

    if len(violations) == 0:
        print("\n✓ Duality relation satisfied for all runs")
    else:
        print(f"\n⚠ Duality violations detected in {len(violations)} runs")
```

#### 8.5: Interpretation (Week 22, 2 days)

**Generate Final Report:**

```python
def generate_analysis_report(summary, chi2_result, tost_results, exclusions):
    """
    Comprehensive analysis report for publication.
    """

    report = f"""
# Statistical Analysis Report
## Observer-Agnostic Measurement Experiment

**Date:** {datetime.now().strftime('%Y-%m-%d')}
**Analyst:** [Name]

---

## Summary

Total runs collected: 30
Excluded runs: {len(exclusions)} ({len(exclusions)/30*100:.1f}%)
Analyzed runs: {30 - len(exclusions)}

Total detection events: {sum(s['total_events'] for s in summary.values()):,}

---

## Singles Probability Results

| Controller | p(0) Mean | Std. Error | 95% CI | Events |
|------------|-----------|------------|--------|--------|
"""

    for controller, stats in summary.items():
        ci_lower = stats['p0_mean'] - 1.96 * stats['p0_sem']
        ci_upper = stats['p0_mean'] + 1.96 * stats['p0_sem']
        report += f"| {controller.capitalize()} | {stats['p0_mean']:.6f} | {stats['p0_sem']:.6f} | [{ci_lower:.6f}, {ci_upper:.6f}] | {stats['total_events']:,} |\n"

    report += f"""
---

## Chi-Squared Test

**Null Hypothesis:** p(0) = 0.5 for all controller types

χ² = {chi2_result[0]:.4f}, p = {chi2_result[1]:.6f}

**Result:** {"REJECT H0" if chi2_result[1] < 0.01 else "FAIL TO REJECT H0"}

---

## TOST Equivalence Tests (ε = 0.0005)

"""

    for result in tost_results:
        c1, c2 = result['controllers']
        report += f"**{c1.upper()} vs {c2.upper()}**\n"
        report += f"  Δp = {result['delta']:.6f}, p = {result['p_value']:.6f}\n"
        report += f"  Status: {'EQUIVALENT' if result['equivalent'] else 'NOT EQUIVALENT'}\n\n"

    report += """
---

## Interpretation

"""

    if chi2_result[1] >= 0.01 and all(r['equivalent'] for r in tost_results):
        report += """
**Conclusion:** The data are consistent with the Observer-Agnostic Measurement
theorem. We find no evidence that quantum measurement outcomes depend on the
type of observer (human, quantum RNG, or deterministic controller).

All pairwise comparisons show statistical equivalence within the pre-defined
margin (ε = 5×10⁻⁴). This result provides strong evidence against "quantum
consciousness" hypotheses that propose observer awareness affects quantum events.

The tight bounds achieved (|Δp| < 5×10⁻⁴) represent the most stringent test to
date of observer-independence in quantum mechanics.
"""
    else:
        report += """
**Conclusion:** A statistically significant deviation from the null hypothesis
was detected. This finding, if confirmed by independent replication, would
suggest that quantum measurement outcomes are influenced by the type of
measurement controller.

**IMPORTANT:** This result requires immediate verification:
1. Independent replication by another laboratory
2. Systematic error analysis (drift, selection bias, hardware artifacts)
3. Theoretical modeling of potential mechanisms
4. Extended data collection with increased statistical power

If robust, this would constitute evidence for new physics beyond standard
quantum mechanics.
"""

    return report
```

### Deliverables

- [ ] `STATISTICAL_ANALYSIS_REPORT.md`
- [ ] `SINGLES_PROBABILITIES.csv` (all runs)
- [ ] `HYPOTHESIS_TEST_RESULTS.json`
- [ ] `CONDITIONED_FRINGES_PLOTS.pdf`
- [ ] `DUALITY_ANALYSIS.csv`

### Success Criteria

- ✅ All statistical tests executed per pre-registration
- ✅ Results interpreted in context of theorem predictions
- ✅ Uncertainties and systematic errors quantified
- ✅ Report ready for peer review

---

## Phase 9: Publication & Dissemination

**Duration:** Weeks 23-24
**Cost:** 17 units (PI: 2 weeks, Co-authors: 1 week, Reviewer: 1 week)
**Team:** Principal Investigator, Co-authors, External Reviewer

### Objectives

1. Write manuscript for peer-reviewed journal
2. Archive data and code with DOIs
3. Submit to arXiv preprint server
4. Submit to target journal (Physical Review Letters or Physical Review A)
5. Respond to peer review and revise

### Action Items

#### 9.1: Manuscript Writing (Week 23, 5 days)

**Target Journals:**
- **Physical Review Letters** (if null result with tight bounds)
- **Physical Review A** (comprehensive report)
- **Optics Express** (experimental focus)

**Manuscript Structure:**

```latex
\documentclass[aps,prl,twocolumn,showpacs,superscriptaddress]{revtex4-2}

\title{Testing Observer-Agnostic Measurement in Delayed-Choice Quantum Eraser Experiments}

\author{[Principal Investigator]}
\affiliation{[Institution]}

\author{[Experimentalist]}
\affiliation{[Institution]}

\author{[Theorist]}
\affiliation{[Institution]}

\begin{abstract}
We report the first experimental test of observer-agnostic invariance in
quantum measurement, using a delayed-choice quantum eraser with photonic
entanglement. Three controller types—human, quantum random number generator,
and deterministic algorithm—selected measurement bases for 15 million detection
events. Singles probabilities showed no dependence on controller type within
5×10⁻⁴, confirming theoretical predictions and providing stringent bounds on
"quantum consciousness" hypotheses. Our results demonstrate that quantum
measurement outcomes are independent of observer identity, biological status,
or awareness, consistent with standard quantum mechanics.
\end{abstract}

\section{Introduction}

[Motivation: Quantum consciousness claims, need for rigorous test]
[Theorem statement and predictions]
[Experimental overview]

\section{Theoretical Framework}

[Formalize observer-agnostic measurement theorem]
[Derive singles invariance and no-retrocausal marginals]
[Compute predictions for delayed-choice eraser setup]

\section{Experimental Methods}

[Apparatus description: SPDC, interferometer, detectors]
[Controller implementations: human, RNG, deterministic]
[Calibration procedures and stability monitoring]
[Pre-registration and blinding protocol]

\section{Results}

[Singles probabilities for three controller types]
[Chi-squared test: p = 0.XX, no significant deviation]
[TOST equivalence: |Δp| < 5×10⁻⁴ for all pairs]
[Conditioned fringes showing quantum erasure]

\section{Discussion}

[Interpretation: Confirms observer-agnostic invariance]
[Comparison to simulation predictions]
[Bounds on quantum consciousness hypotheses]
[Systematic error analysis]

\section{Conclusion}

[Summary of null result and tight bounds]
[Implications for quantum foundations]
[Future work: Bell tests with multiple observers, etc.]

\acknowledgments
[Funding, facilities, helpful discussions]

\bibliographystyle{apsrev4-2}
\bibliography{quantum_observer}
\end{document}
```

**Key Figures:**

1. **Fig. 1:** Experimental schematic (optical layout with controllers)
2. **Fig. 2:** Singles probabilities by controller type (bar chart with error bars)
3. **Fig. 3:** TOST equivalence tests (forest plot showing confidence intervals)
4. **Fig. 4:** Conditioned interference fringes (phase sweep for eraser vs which-path)

#### 9.2: Data and Code Archiving (Week 23, 2 days)

**Data Deposit (Zenodo or Dryad):**

```
quantum-observer-experiment-dataset/
├── README.md (dataset description)
├── data/
│   ├── run_001_Controller_A_20251014_143022.h5
│   ├── run_002_Controller_B_20251015_091534.h5
│   ├── ... (all 30 HDF5 files)
├── metadata/
│   ├── apparatus_specifications.json
│   ├── calibration_logs.csv
│   ├── environmental_logs.csv
│   └── exclusion_summary.json
├── analysis/
│   ├── statistical_analysis.ipynb
│   ├── figures_generation.py
│   └── requirements.txt
└── LICENSE (CC-BY 4.0)
```

**Code Repository (GitHub):**

```
observer-invariance-simulator/
├── README.md
├── LICENSE (MIT)
├── Cargo.toml
├── src/ (Rust simulation code)
├── tests/ (test suite)
├── docs/ (documentation)
├── examples/ (usage examples)
└── paper/ (manuscript LaTeX source)
```

**DOI Assignment:**
- Dataset DOI: 10.5281/zenodo.XXXXXXX
- Code DOI: 10.5281/zenodo.YYYYYYY
- Paper DOI: (assigned by journal upon publication)

#### 9.3: arXiv Preprint (Week 23, 1 day)

**Submit to arXiv:**
- Category: quant-ph (Quantum Physics)
- Subcategory: quant-ph.quantum-ph (primary), physics.optics (cross-list)
- Format: PDFLaTeX with embedded fonts
- Supplementary: Link to data and code DOIs

**arXiv Identifier:** arXiv:2025.XXXXX [quant-ph]

#### 9.4: Journal Submission (Week 24, 1 day)

**Submission Package:**
- Manuscript (PDF)
- Cover letter addressing significance and novelty
- Author contributions statement
- Conflict of interest declaration
- Data availability statement (with DOIs)
- Suggested reviewers (3-5 experts in quantum foundations)

**Cover Letter Highlights:**
- First rigorous test of observer-agnostic measurement
- Null result with unprecedented precision (5×10⁻⁴)
- Falsifiable predictions derived from first principles
- Open data and code for full reproducibility
- Addresses longstanding quantum consciousness debate

#### 9.5: Peer Review Response (Week 24+, ongoing)

**Timeline:**
- Week 24: Submit to journal
- Week 28-32: Receive reviews (typical 4-8 weeks)
- Week 33-34: Revise manuscript based on feedback
- Week 35: Resubmit with point-by-point response
- Week 39-41: Editor decision (accept, minor revisions, or reject)

**Common Review Concerns (anticipated):**

1. **Systematic Errors:**
   - Address: Detailed calibration logs, drift correction, environmental stability
   - Show: Visibility, coincidence rate, temperature logs for all runs

2. **Statistical Power:**
   - Address: Power analysis showing n = 5M per group exceeds requirements
   - Show: Sensitivity analysis for detecting Δp = 10⁻³

3. **Controller Independence:**
   - Address: Blinding protocol, pre-registration, sealed mapping
   - Show: Controller choice distributions (verify randomness of RNG)

4. **Comparison to Theory:**
   - Address: Simulation predictions match experimental results
   - Show: Overlay of simulated vs measured singles and fringes

5. **Alternative Explanations:**
   - Address: Rule out drift, selection bias, detector artifacts
   - Show: Stability tests, cross-checks with swapped detectors

### Deliverables

- [ ] Manuscript submitted to peer-reviewed journal
- [ ] Dataset archived with DOI on Zenodo
- [ ] Code archived with DOI on GitHub/Zenodo
- [ ] arXiv preprint posted
- [ ] Supplementary materials (videos, slides, etc.)

### Success Criteria

- ✅ Manuscript accepted for publication in high-impact journal
- ✅ Data and code publicly available with DOIs
- ✅ arXiv preprint garnering citations and discussion
- ✅ Results reproducible by independent groups

---

## Risk Management

### High-Priority Risks

#### Risk 1: Funding Delays
**Probability:** 40%
**Impact:** High (delays entire timeline by 3-6 months)
**Mitigation:**
- Apply to multiple funding sources simultaneously
- Seek institutional bridge funding
- Negotiate equipment loans or discounts
- Prepare scaled-down version (<$100k) using existing infrastructure

#### Risk 2: BBO Crystal Procurement
**Probability:** 20%
**Impact:** High (critical path item, 6-8 week lead time)
**Mitigation:**
- Order immediately upon funding approval
- Identify backup vendors (Castech, Inrad)
- Consider alternative nonlinear crystals (KDP, LBO)
- Test crystal quality immediately upon receipt

#### Risk 3: Alignment Difficulties
**Probability:** 30%
**Impact:** Medium (adds 1-2 weeks to schedule)
**Mitigation:**
- Hire experienced optical engineer
- Use fiber-coupled components for stability
- Invest in motorized alignment stages
- Budget extra time for troubleshooting

#### Risk 4: Low SPDC Efficiency
**Probability:** 25%
**Impact:** Medium (longer data collection, <1000 pairs/sec)
**Mitigation:**
- Optimize pump power and focusing
- Use high-efficiency detectors (>60% QE)
- Increase pump power (100 mW → 200 mW)
- Accept longer run times (extend Phase 7 by 1 week)

#### Risk 5: Environmental Instability
**Probability:** 35%
**Impact:** Medium (data quality issues, increased exclusions)
**Mitigation:**
- Build thermal enclosure (±0.5°C stability)
- Active vibration isolation on optical table
- Conduct runs during low-traffic hours (nights/weekends)
- Monitor in real-time and abort bad runs early

### Medium-Priority Risks

#### Risk 6: Detector Dead Time Effects
**Probability:** 15%
**Impact:** Low-Medium (systematic bias in coincidences)
**Mitigation:**
- Use SPADs with <50 ns dead time
- Reduce singles rates with ND filters
- Correct for dead time in analysis
- Cross-check with different detectors

#### Risk 7: Time Tagger Limitations
**Probability:** 10%
**Impact:** Low-Medium (reduced coincidence accuracy)
**Mitigation:**
- Rent high-end time tagger (Swabian Ultra, 5 ps resolution)
- Verify timing calibration with LED pulser
- Use software timestamp interpolation if needed

#### Risk 8: Controller Bias
**Probability:** 5%
**Impact:** Critical (undermines entire experiment)
**Mitigation:**
- Rigorous blinding protocol with sealed mapping
- Independent party manages controller identities
- Verify RNG randomness with statistical tests
- Deterministic controller uses cryptographic PRNG

### Low-Priority Risks

#### Risk 9: Data Loss
**Probability:** 5%
**Impact:** High (lose weeks of data collection)
**Mitigation:**
- Real-time backup to multiple locations
- RAID storage on lab computer
- Cloud backup (encrypted, institutional server)
- Daily integrity checks (SHA-256 hashes)

#### Risk 10: Personnel Turnover
**Probability:** 10%
**Impact:** Medium (knowledge loss, training overhead)
**Mitigation:**
- Comprehensive documentation (procedures, troubleshooting)
- Cross-training for all critical tasks
- Video tutorials for apparatus operation
- Maintain lab notebook with daily entries

---

## Resource Requirements

### Personnel

| Role | Effort | Weeks | Cost |
|------|--------|-------|------|
| Principal Investigator | 50% | 24 | $60,000 |
| Postdoc (Experimentalist) | 100% | 16 | $48,000 |
| PhD Student (Theorist) | 50% | 8 | $12,000 |
| Lab Technician | 100% | 8 | $16,000 |
| Statistician | 50% | 2 | $4,000 |
| **Total Personnel** | | | **$140,000** |

### Equipment & Supplies

| Category | Items | Cost |
|----------|-------|------|
| Laser & Optics | 405nm pump, BBO, mirrors, BS, fibers | $60,000 |
| Detectors | 4× SPADs, time tagger | $61,000 |
| Infrastructure | Optical table, enclosure, mounts | $35,000 |
| Electronics | Power supplies, controllers, RNG | $15,000 |
| Consumables | Filters, alignment tools, misc | $9,000 |
| **Total Equipment** | | **$180,000** |

### Computational Resources

| Resource | Specification | Cost |
|----------|---------------|------|
| Workstation | 32-core CPU, 128GB RAM, 2TB SSD | $8,000 |
| Cloud Storage | 1TB backup (3 years) | $500 |
| Software Licenses | MATLAB, Origin, Comsol (optional) | $2,000 |
| **Total Computational** | | **$10,500** |

### Facilities & Overhead

| Item | Details | Cost |
|------|---------|------|
| Lab Space | Optical lab with HVAC (6 months) | $15,000 |
| Utilities | Electricity, cooling | $3,000 |
| Safety | Laser safety training, PPE | $1,500 |
| Travel | Conference presentations (2) | $6,000 |
| Publication | Journal fees, open access | $4,000 |
| **Total Facilities** | | **$29,500** |

### Grand Total: $280,000

---

## Success Criteria

### Simulation (Phase 2-4)

- ✅ Singles invariance: |p(0) - 0.5| < 10⁻¹² for all φ, bases
- ✅ Duality bound: V² + D² ≤ 1.0 + 10⁻¹² for all γ
- ✅ Test coverage: ≥95%
- ✅ Performance: Phase sweep (1000 steps) in <1 second
- ✅ Cross-validation: Matches analytical predictions within numerical precision

### Experiment (Phase 6-8)

- ✅ Entanglement quality: Bell parameter S > 2.5
- ✅ Visibility: V > 0.95 throughout data collection
- ✅ Coincidence rate: >1000 pairs/sec
- ✅ Statistics: ≥15 million events across 3 controller types
- ✅ Stability: <10% run exclusion rate
- ✅ Equivalence: |Δp| < 5×10⁻⁴ for all controller pairs (TOST p < 0.05)

### Publication (Phase 9)

- ✅ Peer-reviewed publication in Physical Review Letters or Physical Review A
- ✅ Data archived with DOI (Zenodo, CC-BY 4.0)
- ✅ Code archived with DOI (GitHub, MIT license)
- ✅ arXiv preprint with >50 citations within 1 year
- ✅ Reproducibility: Independent replication by at least one other group

---

## References

1. **Quantum Measurement Theory:**
   - Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and Quantum Information*. Cambridge University Press.
   - Schlosshauer, M. (2007). *Decoherence and the Quantum-to-Classical Transition*. Springer.
   - Busch, P., Lahti, P., & Mittelstaedt, P. (1996). *The Quantum Theory of Measurement*. Springer.

2. **Delayed-Choice Quantum Eraser:**
   - Kim, Y.-H., et al. (2000). "A Delayed-Choice Quantum Eraser." *Physical Review Letters*, 84(1), 1-5.
   - Walborn, S. P., et al. (2002). "Double-slit quantum eraser." *Physical Review A*, 65(3), 033818.
   - Ma, X.-S., et al. (2016). "Quantum erasure with causally disconnected choice." *Proceedings of the National Academy of Sciences*, 110(4), 1221-1226.

3. **Wave-Particle Duality:**
   - Englert, B.-G. (1996). "Fringe Visibility and Which-Way Information: An Inequality." *Physical Review Letters*, 77(11), 2154-2157.
   - Dürr, S., Nonn, T., & Rempe, G. (1998). "Origin of quantum-mechanical complementarity probed by a 'which-way' experiment in an atom interferometer." *Nature*, 395, 33-37.

4. **Experimental Quantum Optics:**
   - Grangier, P., Roger, G., & Aspect, A. (1986). "Experimental Evidence for a Photon Anticorrelation Effect on a Beam Splitter: A New Light on Single-Photon Interferences." *Europhysics Letters*, 1(4), 173.
   - Kwiat, P. G., et al. (1995). "New High-Intensity Source of Polarization-Entangled Photon Pairs." *Physical Review Letters*, 75(24), 4337-4341.

5. **Statistical Methods:**
   - Schuirmann, D. J. (1987). "A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability." *Journal of Pharmacokinetics and Biopharmaceutics*, 15(6), 657-680.
   - Cowan, G. (1998). *Statistical Data Analysis*. Oxford University Press.

6. **Quantum Consciousness Debates:**
   - Stapp, H. P. (2007). *Mindful Universe: Quantum Mechanics and the Participating Observer*. Springer.
   - Penrose, R. (1989). *The Emperor's New Mind*. Oxford University Press.
   - Tegmark, M. (2000). "The importance of quantum decoherence in brain processes." *Physical Review E*, 61(4), 4194-4206.
   - Yu, S., & Nikolić, D. (2011). "Quantum mechanics needs no consciousness (and the other way around)." *Annalen der Physik*, 523(11), 931-938.

7. **Pre-registration & Reproducibility:**
   - Nosek, B. A., et al. (2018). "The preregistration revolution." *Proceedings of the National Academy of Sciences*, 115(11), 2600-2606.
   - Munafò, M. R., et al. (2017). "A manifesto for reproducible science." *Nature Human Behaviour*, 1(1), 0021.

---

## Appendices

### Appendix A: Detailed Phase Plans

Each phase has a dedicated detailed plan document:
- [PHASE_1_THEORETICAL.md](./PHASE_1_THEORETICAL.md)
- [PHASE_2_SIMULATION.md](./PHASE_2_SIMULATION.md)
- [PHASE_3_TESTING.md](./PHASE_3_TESTING.md)
- [PHASE_4_VALIDATION.md](./PHASE_4_VALIDATION.md)
- [PHASE_5_EXPERIMENTAL_DESIGN.md](./PHASE_5_EXPERIMENTAL_DESIGN.md)
- [PHASE_6_LAB_SETUP.md](./PHASE_6_LAB_SETUP.md)
- [PHASE_7_DATA_COLLECTION.md](./PHASE_7_DATA_COLLECTION.md)
- [PHASE_8_ANALYSIS.md](./PHASE_8_ANALYSIS.md)
- [PHASE_9_PUBLICATION.md](./PHASE_9_PUBLICATION.md)

### Appendix B: Code Templates

See [rust-simulator/](../rust-simulator/) directory for complete Rust implementation.

### Appendix C: Experimental Protocols

See [experimental-protocols/](../experimental-protocols/) directory for detailed SOPs.

### Appendix D: Data Analysis Scripts

See [data-analysis/](../data-analysis/) directory for Python analysis code.

---

**Document Status:** Ready for Execution
**Next Action:** Review by research team, begin Phase 1 (Theoretical Foundation)
**Last Updated:** 2025-10-14
**Version:** 1.0
