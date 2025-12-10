# Algorithm Documentation Index

This directory contains detailed pseudocode for all CRISPR-Cas13 pipeline modules.

## Files

### 1. [alignment-module.md](alignment-module.md)
**Read Alignment Module** - 868 lines

Detailed algorithms for:
- BWT Index Construction (DC3 algorithm)
- BWT Exact Match (Backward Search)
- Seed-and-Extend Alignment Strategy
- Smith-Waterman Local Alignment (with affine gaps)
- MAPQ Score Calculation
- Main Alignment Pipeline (with parallelization)

**Key Complexity**: O(N × n × m / p) where N=reads, n=read length, m=reference size, p=cores

**Performance**: 10,000-50,000 reads/second on 32-core system

---

### 2. [offtarget-module.md](offtarget-module.md)
**Off-Target Prediction Module** - 1,030 lines

Detailed algorithms for:
- Off-Target Site Discovery (k-mer indexing)
- Feature Extraction (15-20 biological features)
- Gradient Boosting Model Training (XGBoost-style)
- Off-Target Scoring & Ranking
- RNA Secondary Structure Prediction (Vienna RNA fold)
- Binding Affinity Calculation

**Key Complexity**: O(T × L + K × n²) where T=transcriptome size, K=candidates, n=gRNA length

**Performance**: 100-1000 ms per gRNA for full prediction

---

### 3. [expression-module.md](expression-module.md)
**Differential Expression Analysis Module** - 928 lines

Detailed algorithms for:
- DESeq2 Normalization (Size Factor Estimation)
- Dispersion Estimation (Negative Binomial GLM)
- GLM Fitting (IRLS algorithm)
- Differential Expression Testing (Wald Test)
- Benjamini-Hochberg FDR Correction
- Log Fold Change Shrinkage (apeglm-style)
- Immune Pathway Enrichment Analysis

**Key Complexity**: O(G × S × I × C²) where G=genes, S=samples, I=iterations, C=coefficients

**Performance**: 5-30 minutes for 20,000 genes × 50 samples

---

### 4. [reporting-module.md](reporting-module.md)
**Reporting Engine** - 1,135 lines

Detailed algorithms for:
- Report Data Aggregation
- Visualization Generation (10+ plot types)
  - Volcano plots
  - MA plots
  - Heatmaps with hierarchical clustering
  - Pathway dot plots
  - PCA plots
- HTML Report Generation (interactive)
- PDF Export (headless browser rendering)
- Data Export (VCF, BED, CSV, JSON)

**Key Complexity**: O(G × S + G² log G) for heatmap with clustering

**Performance**: 30 seconds to 5 minutes depending on dataset size

---

## Algorithm Design Principles

All algorithms follow SPARC methodology standards:

1. **Language-Agnostic**: Written in clear pseudocode, not language-specific
2. **Comprehensive**: Include main algorithms and all subroutines
3. **Complexity-Analyzed**: Time and space complexity for each algorithm
4. **Error-Handled**: Explicit error handling and validation
5. **Performance-Optimized**: Consider parallelization and caching strategies

## Usage

Each module documentation includes:
- Algorithm overview and purpose
- Detailed pseudocode with clear control flow
- Complexity analysis (time and space)
- Error handling strategies
- Performance optimization notes
- Testing strategy
- References to academic papers

## Related Documentation

- [Main Pseudocode Overview](../PSEUDOCODE.md)
- [Flowcharts](../flowcharts/)
- [Architecture Documentation](../ARCHITECTURE.md)
- [Testing Guide](../TESTING_GUIDE.md)

## Contributing

When adding new algorithms:
1. Follow SPARC pseudocode standards
2. Include complexity analysis
3. Document error handling
4. Provide performance characteristics
5. Add corresponding flowchart
6. Include unit test pseudocode

---

**Total Lines**: 3,961 lines of detailed algorithmic documentation
**Total Algorithms**: 26 major algorithms + 40+ subroutines
