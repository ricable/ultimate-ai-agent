# Differential Expression Analysis Module - Detailed Pseudocode

## Overview

The Differential Expression Analysis Module identifies differentially expressed genes (DEGs) using DESeq2-style statistical methodology with negative binomial modeling, followed by immune pathway enrichment analysis specific to CRISPR-Cas13 effects.

---

## Algorithm 1: Count Normalization

**Purpose**: Normalize raw RNA-seq read counts using size factor estimation to account for sequencing depth differences.

```
ALGORITHM: NormalizeCountsDESeq2
INPUT: rawCounts (matrix: genes × samples), sampleMetadata (DataFrame)
OUTPUT: normalizedCounts (matrix), sizeFactors (array)

BEGIN
    numGenes ← NumRows(rawCounts)
    numSamples ← NumColumns(rawCounts)

    // Step 1: Calculate geometric means for each gene across samples
    geoMeans ← []
    FOR geneIdx ← 0 TO numGenes - 1 DO
        geneCounts ← rawCounts[geneIdx, :]

        // Filter zero counts for geometric mean calculation
        nonZeroCounts ← [count for count in geneCounts WHERE count > 0]

        IF length(nonZeroCounts) == 0 THEN
            geoMeans.append(0)
        ELSE
            // Geometric mean = exp(mean(log(counts)))
            logSum ← SUM(LOG(count) for count in nonZeroCounts)
            geoMean ← EXP(logSum / length(nonZeroCounts))
            geoMeans.append(geoMean)
        END IF
    END FOR

    // Step 2: Calculate size factors for each sample
    sizeFactors ← []
    FOR sampleIdx ← 0 TO numSamples - 1 DO
        sampleCounts ← rawCounts[:, sampleIdx]

        // Calculate ratio to geometric mean for each gene
        ratios ← []
        FOR geneIdx ← 0 TO numGenes - 1 DO
            IF geoMeans[geneIdx] > 0 AND sampleCounts[geneIdx] > 0 THEN
                ratio ← sampleCounts[geneIdx] / geoMeans[geneIdx]
                ratios.append(ratio)
            END IF
        END FOR

        IF length(ratios) == 0 THEN
            LogError("Sample {sampleIdx} has no valid ratios")
            sizeFactors.append(1.0)
        ELSE
            // Size factor = median of ratios
            sizeFactor ← MEDIAN(ratios)
            sizeFactors.append(sizeFactor)
        END IF
    END FOR

    LogInfo("Size factors: {sizeFactors}")

    // Step 3: Normalize counts by dividing by size factors
    normalizedCounts ← CREATE matrix numGenes × numSamples
    FOR geneIdx ← 0 TO numGenes - 1 DO
        FOR sampleIdx ← 0 TO numSamples - 1 DO
            normalizedCounts[geneIdx][sampleIdx] ← rawCounts[geneIdx][sampleIdx] / sizeFactors[sampleIdx]
        END FOR
    END FOR

    RETURN {
        normalizedCounts: normalizedCounts,
        sizeFactors: sizeFactors,
        geometricMeans: geoMeans
    }
END
```

**Complexity Analysis**:
- **Time**: O(G × S) where G = genes, S = samples
- **Space**: O(G × S) for count matrix
- **Typical Performance**: 1-5 seconds for 20,000 genes × 100 samples

---

## Algorithm 2: Dispersion Estimation

**Purpose**: Estimate gene-wise dispersion parameters for negative binomial distribution.

```
ALGORITHM: EstimateDispersion
INPUT: normalizedCounts (matrix), sampleMetadata (DataFrame), design (formula)
OUTPUT: dispersionEstimates (array of dispersion values per gene)

CONSTANTS:
    MIN_DISPERSION = 1e-8
    MAX_DISPERSION = 10
    MAX_ITERATIONS = 100
    CONVERGENCE_THRESHOLD = 1e-6

BEGIN
    numGenes ← NumRows(normalizedCounts)
    numSamples ← NumColumns(normalizedCounts)

    // Step 1: Parse design formula and create design matrix
    designMatrix ← CreateDesignMatrix(sampleMetadata, design)
    numCoefficients ← NumColumns(designMatrix)

    // Step 2: Fit GLM for each gene to get initial dispersion estimates
    geneDispersions ← []
    geneMeans ← []

    FOR geneIdx ← 0 TO numGenes - 1 DO
        geneCounts ← normalizedCounts[geneIdx, :]

        // Fit negative binomial GLM
        glmResult ← FitNegativeBinomialGLM(
            counts = geneCounts,
            designMatrix = designMatrix,
            initialDispersion = 0.1
        )

        geneDispersions.append(glmResult.dispersion)
        geneMeans.append(glmResult.mu)
    END FOR

    // Step 3: Fit dispersion-mean relationship (parametric fit)
    // Model: log(dispersion) = a / mu + b
    dispersionModel ← FitDispersionTrend(geneMeans, geneDispersions)

    LogInfo("Dispersion trend: a = {dispersionModel.a}, b = {dispersionModel.b}")

    // Step 4: Shrink gene-wise dispersions toward trend
    finalDispersions ← []
    FOR geneIdx ← 0 TO numGenes - 1 DO
        geneMean ← geneMeans[geneIdx]
        geneDispersion ← geneDispersions[geneIdx]

        // Predicted dispersion from trend
        trendDispersion ← dispersionModel.a / geneMean + dispersionModel.b

        // Empirical Bayes shrinkage
        // Weight depends on number of samples and variance
        shrinkageWeight ← CalculateShrinkageWeight(
            numSamples,
            geneDispersion,
            trendDispersion
        )

        shrunkDispersion ← shrinkageWeight × trendDispersion +
                          (1 - shrinkageWeight) × geneDispersion

        // Clamp to reasonable range
        shrunkDispersion ← MAX(MIN_DISPERSION, MIN(MAX_DISPERSION, shrunkDispersion))

        finalDispersions.append(shrunkDispersion)
    END FOR

    RETURN finalDispersions
END

SUBROUTINE: FitNegativeBinomialGLM
INPUT: counts (array), designMatrix (matrix), initialDispersion (float)
OUTPUT: glmResult (object with coefficients and dispersion)

BEGIN
    numSamples ← length(counts)
    numCoefficients ← NumColumns(designMatrix)

    // Initialize coefficients
    beta ← [0] × numCoefficients
    dispersion ← initialDispersion

    // Iteratively reweighted least squares (IRLS)
    FOR iteration ← 0 TO MAX_ITERATIONS - 1 DO
        // Calculate predicted means
        mu ← []
        FOR i ← 0 TO numSamples - 1 DO
            linearPredictor ← DOT_PRODUCT(designMatrix[i, :], beta)
            mu.append(EXP(linearPredictor))
        END FOR

        // Calculate weights (Fisher information)
        weights ← []
        FOR i ← 0 TO numSamples - 1 DO
            // Variance = mu + dispersion * mu^2 (negative binomial)
            variance ← mu[i] + dispersion × mu[i]²
            weight ← mu[i]² / variance
            weights.append(weight)
        END FOR

        // Weighted least squares update
        weightedDesign ← SQRT(weights) * designMatrix
        weightedCounts ← SQRT(weights) * (counts - mu + designMatrix * beta)

        betaNew ← SOLVE(weightedDesign^T * weightedDesign,
                       weightedDesign^T * weightedCounts)

        // Check convergence
        IF MAX(ABS(betaNew - beta)) < CONVERGENCE_THRESHOLD THEN
            beta ← betaNew
            BREAK
        END IF

        beta ← betaNew
    END FOR

    // Estimate dispersion using maximum likelihood
    residuals ← counts - mu
    pearsonResiduals ← residuals / SQRT(mu + dispersion × mu²)

    dispersionEstimate ← EstimateDispersionML(counts, mu, dispersion)

    RETURN {
        coefficients: beta,
        mu: mu,
        dispersion: dispersionEstimate,
        converged: (iteration < MAX_ITERATIONS - 1)
    }
END

SUBROUTINE: FitDispersionTrend
INPUT: means (array), dispersions (array)
OUTPUT: model (object with parameters)

BEGIN
    // Remove outliers (robust fit)
    validIndices ← []
    FOR i ← 0 TO length(means) - 1 DO
        IF means[i] > 1 AND dispersions[i] > 0 AND dispersions[i] < 10 THEN
            validIndices.append(i)
        END IF
    END FOR

    filteredMeans ← [means[i] for i in validIndices]
    filteredDispersions ← [dispersions[i] for i in validIndices]

    // Fit: log(dispersion) = a / mu + b
    // Transform: y = log(disp), x = 1/mu
    x ← [1/mu for mu in filteredMeans]
    y ← [LOG(disp) for disp in filteredDispersions]

    // Weighted least squares (weight by mean)
    weights ← filteredMeans

    // Calculate weighted coefficients
    sumW ← SUM(weights)
    sumWX ← SUM(weights[i] × x[i] for i in 0 to length(x)-1)
    sumWY ← SUM(weights[i] × y[i] for i in 0 to length(y)-1)
    sumWXX ← SUM(weights[i] × x[i]² for i in 0 to length(x)-1)
    sumWXY ← SUM(weights[i] × x[i] × y[i] for i in 0 to length(x)-1)

    // Solve for a and b
    a ← (sumW × sumWXY - sumWX × sumWY) / (sumW × sumWXX - sumWX²)
    b ← (sumWY - a × sumWX) / sumW

    RETURN {
        a: a,
        b: b,
        fitType: "parametric"
    }
END

SUBROUTINE: CalculateShrinkageWeight
INPUT: numSamples, geneDispersion, trendDispersion
OUTPUT: weight (float in [0, 1])

BEGIN
    // More samples → more confidence in gene-wise estimate → less shrinkage
    // Larger difference from trend → less shrinkage

    sampleWeight ← numSamples / (numSamples + 10)

    dispersionDiff ← ABS(LOG(geneDispersion) - LOG(trendDispersion))
    variabilityWeight ← 1 / (1 + dispersionDiff)

    weight ← 1 - (sampleWeight × variabilityWeight)

    // Clamp to [0.1, 0.9] to avoid complete shrinkage
    weight ← MAX(0.1, MIN(0.9, weight))

    RETURN weight
END
```

**Complexity Analysis**:
- **Time**: O(G × S × I × C²) where I = iterations, C = coefficients
- **Space**: O(G × S) for count matrix
- **Typical Performance**: 1-5 minutes for 20,000 genes

---

## Algorithm 3: Differential Expression Testing

**Purpose**: Perform Wald test for statistical significance of differential expression.

```
ALGORITHM: TestDifferentialExpression
INPUT: normalizedCounts (matrix), sampleMetadata (DataFrame),
       dispersions (array), design (formula), contrast (string)
OUTPUT: results (DataFrame with gene statistics)

BEGIN
    numGenes ← NumRows(normalizedCounts)

    // Step 1: Parse contrast (e.g., "treated_vs_control")
    contrastVector ← ParseContrast(contrast, design)

    // Step 2: Fit full model for each gene
    geneResults ← []

    FOR geneIdx ← 0 TO numGenes - 1 DO
        geneCounts ← normalizedCounts[geneIdx, :]
        geneDispersion ← dispersions[geneIdx]

        // Fit negative binomial GLM
        glmResult ← FitNegativeBinomialGLM(
            counts = geneCounts,
            designMatrix = designMatrix,
            initialDispersion = geneDispersion
        )

        // Calculate log2 fold change
        log2FC ← DOT_PRODUCT(glmResult.coefficients, contrastVector) / LOG(2)

        // Calculate standard error
        // SE = sqrt(contrast^T * Cov(beta) * contrast)
        covarianceMatrix ← CalculateCovarianceMatrix(glmResult, designMatrix)
        standardError ← SQRT(DOT_PRODUCT(contrastVector,
                                        covarianceMatrix * contrastVector))

        // Wald test statistic
        waldStatistic ← log2FC / standardError

        // P-value from standard normal distribution
        pValue ← 2 × (1 - NormalCDF(ABS(waldStatistic)))

        // Calculate base mean
        baseMean ← MEAN(glmResult.mu)

        geneResults.append({
            geneId: GetGeneId(geneIdx),
            baseMean: baseMean,
            log2FoldChange: log2FC,
            lfcSE: standardError,
            stat: waldStatistic,
            pvalue: pValue,
            dispersion: geneDispersion
        })
    END FOR

    // Step 3: Multiple testing correction (Benjamini-Hochberg)
    adjustedPValues ← BenjaminiHochbergCorrection(
        [result.pvalue for result in geneResults]
    )

    FOR i ← 0 TO length(geneResults) - 1 DO
        geneResults[i].padj ← adjustedPValues[i]
    END FOR

    // Step 4: Add annotations
    FOR i ← 0 TO length(geneResults) - 1 DO
        geneResults[i].geneSymbol ← GetGeneSymbol(geneResults[i].geneId)
        geneResults[i].biotype ← GetBiotype(geneResults[i].geneId)
    END FOR

    // Convert to DataFrame and sort by adjusted p-value
    resultsDF ← CreateDataFrame(geneResults)
    resultsDF ← SortBy(resultsDF, column = "padj", ascending = true)

    RETURN resultsDF
END

SUBROUTINE: CalculateCovarianceMatrix
INPUT: glmResult (object), designMatrix (matrix)
OUTPUT: covMatrix (matrix)

BEGIN
    numSamples ← NumRows(designMatrix)
    numCoefficients ← NumColumns(designMatrix)

    // Calculate Fisher information matrix
    // I = X^T * W * X
    // where W = diag(weights)

    weights ← []
    FOR i ← 0 TO numSamples - 1 DO
        mu ← glmResult.mu[i]
        dispersion ← glmResult.dispersion
        variance ← mu + dispersion × mu²
        weight ← mu² / variance
        weights.append(weight)
    END FOR

    weightMatrix ← DiagonalMatrix(weights)

    informationMatrix ← TRANSPOSE(designMatrix) * weightMatrix * designMatrix

    // Covariance matrix = inverse of information matrix
    TRY
        covMatrix ← INVERT(informationMatrix)
    CATCH SingularMatrixError AS e DO
        LogWarning("Singular information matrix, using pseudo-inverse")
        covMatrix ← PSEUDO_INVERT(informationMatrix)
    END TRY

    RETURN covMatrix
END

SUBROUTINE: BenjaminiHochbergCorrection
INPUT: pValues (array of p-values)
OUTPUT: adjustedPValues (array of adjusted p-values)

BEGIN
    numTests ← length(pValues)

    // Create array of (pValue, originalIndex) pairs
    indexedPValues ← [(pValues[i], i) for i in 0 to numTests-1]

    // Sort by p-value
    sortedPValues ← Sort(indexedPValues, by = pValue, ascending = true)

    // Calculate adjusted p-values
    adjustedPValues ← [0] × numTests
    minAdjusted ← 1.0

    FOR i ← numTests - 1 DOWN TO 0 DO
        pValue ← sortedPValues[i].pValue
        originalIndex ← sortedPValues[i].originalIndex

        // BH adjustment: p * numTests / rank
        adjusted ← pValue × numTests / (i + 1)

        // Enforce monotonicity
        adjusted ← MIN(adjusted, minAdjusted)
        adjusted ← MIN(adjusted, 1.0)

        adjustedPValues[originalIndex] ← adjusted
        minAdjusted ← adjusted
    END FOR

    RETURN adjustedPValues
END
```

**Complexity Analysis**:
- **Time**: O(G × S × C²) for GLM fitting, O(G log G) for sorting
- **Space**: O(G × S) for count matrix
- **Typical Performance**: 2-10 minutes for 20,000 genes

---

## Algorithm 4: Log Fold Change Shrinkage

**Purpose**: Shrink log2 fold changes to reduce noise in low-count genes.

```
ALGORITHM: ShrinkLogFoldChanges
INPUT: deResults (DataFrame), normalizedCounts (matrix)
OUTPUT: shrunkResults (DataFrame)

BEGIN
    // Use adaptive shrinkage (apeglm-style) based on precision

    FOR EACH gene IN deResults DO
        baseMean ← gene.baseMean
        log2FC ← gene.log2FoldChange
        lfcSE ← gene.lfcSE

        // Calculate precision (inverse variance)
        precision ← 1 / lfcSE²

        // Prior distribution parameters (learned from all genes)
        priorMean ← 0  // Assume null hypothesis centered at 0
        priorPrecision ← EstimatePriorPrecision(deResults)

        // Posterior mean (shrunk log fold change)
        posteriorPrecision ← precision + priorPrecision
        posteriorMean ← (precision × log2FC + priorPrecision × priorMean) /
                       posteriorPrecision

        // Posterior standard error
        posteriorSE ← SQRT(1 / posteriorPrecision)

        // Update results
        gene.log2FoldChange ← posteriorMean
        gene.lfcSE ← posteriorSE

        // Recalculate test statistic with shrunk values
        gene.stat ← posteriorMean / posteriorSE
        gene.pvalue ← 2 × (1 - NormalCDF(ABS(gene.stat)))
    END FOR

    // Recalculate adjusted p-values
    pValues ← [gene.pvalue for gene in deResults]
    adjustedPValues ← BenjaminiHochbergCorrection(pValues)

    FOR i ← 0 TO length(deResults) - 1 DO
        deResults[i].padj ← adjustedPValues[i]
    END FOR

    RETURN deResults
END

SUBROUTINE: EstimatePriorPrecision
INPUT: deResults (DataFrame)
OUTPUT: priorPrecision (float)

BEGIN
    // Estimate prior precision from distribution of log fold changes
    // Use genes with high confidence (low p-value)

    highConfidenceGenes ← [gene for gene in deResults
                          WHERE gene.pvalue < 0.01 AND gene.baseMean > 10]

    IF length(highConfidenceGenes) < 100 THEN
        LogWarning("Few high-confidence genes, using default prior")
        RETURN 10.0  // Default prior precision
    END IF

    log2FCs ← [gene.log2FoldChange for gene in highConfidenceGenes]

    // Estimate variance (excluding outliers)
    q25 ← Percentile(log2FCs, 25)
    q75 ← Percentile(log2FCs, 75)
    iqr ← q75 - q25

    filteredLog2FCs ← [lfc for lfc in log2FCs
                      WHERE lfc >= q25 - 1.5×iqr AND lfc <= q75 + 1.5×iqr]

    variance ← Variance(filteredLog2FCs)
    priorPrecision ← 1 / variance

    LogInfo("Estimated prior precision: {priorPrecision}")

    RETURN priorPrecision
END
```

**Complexity Analysis**:
- **Time**: O(G) for shrinkage
- **Space**: O(G) for gene results
- **Typical Performance**: 1-5 seconds

---

## Algorithm 5: Immune Pathway Enrichment

**Purpose**: Identify enriched immune pathways in differentially expressed genes.

```
ALGORITHM: ImmunePathwayEnrichment
INPUT: deResults (DataFrame), pathwayDatabase (PathwayDB),
       significanceThreshold (float, default = 0.05)
OUTPUT: enrichmentResults (DataFrame)

CONSTANTS:
    IMMUNE_PATHWAYS = [
        "Interferon Signaling",
        "Cytokine Signaling",
        "T Cell Activation",
        "B Cell Activation",
        "Inflammasome",
        "Complement Cascade",
        "Antigen Presentation",
        "NK Cell Cytotoxicity"
    ]

BEGIN
    // Step 1: Extract significant DEGs
    significantGenes ← [gene.geneId for gene in deResults
                       WHERE gene.padj < significanceThreshold]

    upregulatedGenes ← [gene.geneId for gene in deResults
                       WHERE gene.padj < significanceThreshold AND gene.log2FoldChange > 0]

    downregulatedGenes ← [gene.geneId for gene in deResults
                         WHERE gene.padj < significanceThreshold AND gene.log2FoldChange < 0]

    LogInfo("Significant DEGs: {length(significantGenes)} " +
           "({length(upregulatedGenes)} up, {length(downregulatedGenes)} down)")

    // Step 2: Filter to immune-related pathways
    immunePathways ← [pathway for pathway in pathwayDatabase.pathways
                     WHERE pathway.category IN IMMUNE_PATHWAYS]

    // Step 3: Perform hypergeometric test for each pathway
    enrichmentResults ← []
    totalGenes ← pathwayDatabase.totalGenes

    FOR EACH pathway IN immunePathways DO
        pathwayGenes ← pathway.geneSet

        // Genes in pathway AND significant
        overlap ← SET_INTERSECTION(significantGenes, pathwayGenes)
        overlapSize ← length(overlap)

        // Hypergeometric test
        // P(X >= k) where k = overlap size
        pValue ← HypergeometricTest(
            populationSize = totalGenes,
            successStatesInPopulation = length(pathwayGenes),
            numberOfDraws = length(significantGenes),
            numberOfSuccesses = overlapSize
        )

        // Calculate enrichment fold change
        expectedOverlap ← (length(significantGenes) × length(pathwayGenes)) / totalGenes
        foldEnrichment ← overlapSize / expectedOverlap

        // Calculate overlap statistics
        overlapUpregulated ← length(SET_INTERSECTION(upregulatedGenes, pathwayGenes))
        overlapDownregulated ← length(SET_INTERSECTION(downregulatedGenes, pathwayGenes))

        enrichmentResults.append({
            pathway: pathway.name,
            pathwayId: pathway.id,
            category: pathway.category,
            pValue: pValue,
            overlapSize: overlapSize,
            pathwaySize: length(pathwayGenes),
            foldEnrichment: foldEnrichment,
            overlapGenes: overlap,
            upregulated: overlapUpregulated,
            downregulated: overlapDownregulated
        })
    END FOR

    // Step 4: Multiple testing correction
    pValues ← [result.pValue for result in enrichmentResults]
    adjustedPValues ← BenjaminiHochbergCorrection(pValues)

    FOR i ← 0 TO length(enrichmentResults) - 1 DO
        enrichmentResults[i].padjusted ← adjustedPValues[i]
    END FOR

    // Step 5: Sort by adjusted p-value
    enrichmentResults ← Sort(enrichmentResults, by = padjusted, ascending = true)

    // Step 6: Calculate combined immune score
    immuneScore ← CalculateImmuneScore(enrichmentResults, deResults)

    LogInfo("Top enriched pathway: {enrichmentResults[0].pathway} " +
           "(FDR = {enrichmentResults[0].padjusted})")
    LogInfo("Immune activation score: {immuneScore}")

    RETURN {
        pathways: enrichmentResults,
        immuneScore: immuneScore
    }
END

SUBROUTINE: HypergeometricTest
INPUT: populationSize, successStatesInPopulation, numberOfDraws, numberOfSuccesses
OUTPUT: pValue (float)

BEGIN
    // P(X >= k) = sum_{i=k}^{min(n,K)} C(K,i) * C(N-K, n-i) / C(N,n)
    // where N = populationSize, K = successStatesInPopulation,
    //       n = numberOfDraws, k = numberOfSuccesses

    N ← populationSize
    K ← successStatesInPopulation
    n ← numberOfDraws
    k ← numberOfSuccesses

    IF k > MIN(n, K) THEN
        RETURN 1.0  // Impossible
    END IF

    pValue ← 0
    maxI ← MIN(n, K)

    FOR i ← k TO maxI DO
        // Calculate hypergeometric probability
        prob ← (BinomialCoefficient(K, i) ×
               BinomialCoefficient(N - K, n - i)) /
               BinomialCoefficient(N, n)

        pValue ← pValue + prob
    END FOR

    RETURN pValue
END

SUBROUTINE: CalculateImmuneScore
INPUT: enrichmentResults (array), deResults (DataFrame)
OUTPUT: immuneScore (float)

BEGIN
    // Composite score based on:
    // 1. Number of significantly enriched immune pathways
    // 2. Strength of enrichment
    // 3. Direction of regulation (up vs down)

    significantPathways ← [pathway for pathway in enrichmentResults
                          WHERE pathway.padjusted < 0.05]

    IF length(significantPathways) == 0 THEN
        RETURN 0
    END IF

    // Component 1: Pathway count score (0-33 points)
    pathwayScore ← MIN(33, length(significantPathways) × 3)

    // Component 2: Enrichment strength score (0-33 points)
    avgFoldEnrichment ← MEAN([p.foldEnrichment for p in significantPathways])
    enrichmentScore ← MIN(33, avgFoldEnrichment × 10)

    // Component 3: Immune activation direction score (0-34 points)
    avgUpregRatio ← MEAN([p.upregulated / p.overlapSize
                         for p in significantPathways
                         WHERE p.overlapSize > 0])

    // Score higher for predominantly upregulated immune genes
    // (indicates immune activation)
    IF avgUpregRatio > 0.6 THEN
        directionScore ← 34  // Strong activation
    ELSE IF avgUpregRatio > 0.4 THEN
        directionScore ← 17  // Mixed response
    ELSE
        directionScore ← 0   // Immune suppression
    END IF

    immuneScore ← pathwayScore + enrichmentScore + directionScore

    RETURN immuneScore
END
```

**Complexity Analysis**:
- **Time**: O(P × G) where P = pathways, G = significant genes
- **Space**: O(P × G) for pathway overlap storage
- **Typical Performance**: 1-10 seconds for 200 pathways

---

## Algorithm 6: Main Expression Analysis Pipeline

**Purpose**: Orchestrate full differential expression workflow.

```
ALGORITHM: AnalyzeDifferentialExpression
INPUT: countMatrixFile (path), metadataFile (path), config (AnalysisConfig)
OUTPUT: analysisResults (DifferentialExpressionResults)

BEGIN
    startTime ← GetCurrentTime()

    // Step 1: Load data
    LogInfo("Loading count matrix and metadata...")
    rawCounts ← LoadCountMatrix(countMatrixFile)
    sampleMetadata ← LoadMetadata(metadataFile)

    ValidateSampleMatching(rawCounts, sampleMetadata)

    // Step 2: Quality control filtering
    LogInfo("Performing quality control filtering...")
    filteredCounts ← FilterLowCounts(rawCounts, minCount = 10, minSamples = 3)

    numFiltered ← NumRows(rawCounts) - NumRows(filteredCounts)
    LogInfo("Filtered {numFiltered} low-count genes")

    // Step 3: Normalization
    LogInfo("Normalizing counts...")
    normResult ← NormalizeCountsDESeq2(filteredCounts, sampleMetadata)
    normalizedCounts ← normResult.normalizedCounts
    sizeFactors ← normResult.sizeFactors

    // Step 4: Dispersion estimation
    LogInfo("Estimating dispersions...")
    dispersions ← EstimateDispersion(
        normalizedCounts,
        sampleMetadata,
        design = config.designFormula
    )

    // Step 5: Differential expression testing
    LogInfo("Testing for differential expression...")
    deResults ← TestDifferentialExpression(
        normalizedCounts,
        sampleMetadata,
        dispersions,
        design = config.designFormula,
        contrast = config.contrast
    )

    // Step 6: Log fold change shrinkage
    LogInfo("Shrinking log fold changes...")
    shrunkResults ← ShrinkLogFoldChanges(deResults, normalizedCounts)

    // Step 7: Immune pathway enrichment
    LogInfo("Performing immune pathway enrichment...")
    enrichmentResults ← ImmunePathwayEnrichment(
        shrunkResults,
        config.pathwayDatabase,
        significanceThreshold = config.fdrThreshold
    )

    // Step 8: Generate summary statistics
    numSignificant ← COUNT(gene for gene in shrunkResults
                          WHERE gene.padj < config.fdrThreshold)

    numUpregulated ← COUNT(gene for gene in shrunkResults
                          WHERE gene.padj < config.fdrThreshold AND
                                gene.log2FoldChange > 0)

    numDownregulated ← COUNT(gene for gene in shrunkResults
                            WHERE gene.padj < config.fdrThreshold AND
                                  gene.log2FoldChange < 0)

    processingTime ← GetCurrentTime() - startTime

    statistics ← {
        totalGenesTested: NumRows(shrunkResults),
        significantGenes: numSignificant,
        upregulatedGenes: numUpregulated,
        downregulatedGenes: numDownregulated,
        significantPathways: COUNT(p for p in enrichmentResults.pathways
                                  WHERE p.padjusted < config.fdrThreshold),
        immuneScore: enrichmentResults.immuneScore,
        processingTime: processingTime
    }

    // Step 9: Package results
    results ← {
        deResults: shrunkResults,
        enrichment: enrichmentResults,
        normalizedCounts: normalizedCounts,
        sizeFactors: sizeFactors,
        dispersions: dispersions,
        statistics: statistics,
        config: config,
        timestamp: GetCurrentTime()
    }

    LogInfo("Differential expression analysis complete")
    LogInfo("Significant DEGs: {numSignificant} ({numUpregulated} up, {numDownregulated} down)")
    LogInfo("Immune activation score: {enrichmentResults.immuneScore}/100")

    RETURN results
END
```

**Complexity Analysis**:
- **Time**: O(G × S × I) for full pipeline
- **Space**: O(G × S) for count matrices
- **End-to-End Performance**: 5-30 minutes for typical datasets

---

## Error Handling

### 1. Input Validation

```
FUNCTION: ValidateCountMatrix
INPUT: countMatrix (matrix)
OUTPUT: boolean (valid or not)

BEGIN
    // Check for negative counts
    FOR EACH cell IN countMatrix DO
        IF cell < 0 THEN
            LogError("Negative count detected: {cell}")
            RETURN false
        END IF
    END FOR

    // Check for all-zero genes
    FOR geneIdx ← 0 TO NumRows(countMatrix) - 1 DO
        geneSum ← SUM(countMatrix[geneIdx, :])
        IF geneSum == 0 THEN
            LogWarning("Gene {geneIdx} has zero counts across all samples")
        END IF
    END FOR

    // Check for all-zero samples
    FOR sampleIdx ← 0 TO NumColumns(countMatrix) - 1 DO
        sampleSum ← SUM(countMatrix[:, sampleIdx])
        IF sampleSum < 1000 THEN
            LogWarning("Sample {sampleIdx} has very low total counts: {sampleSum}")
        END IF
    END FOR

    RETURN true
END
```

### 2. Convergence Monitoring

```
FUNCTION: FitGLMWithMonitoring
INPUT: counts, designMatrix, maxIterations
OUTPUT: glmResult or error

BEGIN
    TRY
        glmResult ← FitNegativeBinomialGLM(counts, designMatrix, maxIterations)

        IF NOT glmResult.converged THEN
            LogWarning("GLM did not converge, results may be unreliable")
        END IF

        RETURN glmResult

    CATCH NumericalError AS e DO
        LogError("Numerical instability in GLM fit: {e.message}")
        // Try with more conservative parameters
        RETURN FitGLMRobust(counts, designMatrix)
    END TRY
END
```

---

## References

1. Love, M. I., Huber, W., & Anders, S. (2014). Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2. *Genome Biology*.
2. Robinson, M. D., McCarthy, D. J., & Smyth, G. K. (2010). edgeR: a Bioconductor package for differential expression analysis of digital gene expression data. *Bioinformatics*.
3. Zhu, A., Ibrahim, J. G., & Love, M. I. (2019). Heavy-tailed prior distributions for sequence count data: removing the noise and preserving large differences. *Bioinformatics*.

---

**Next Module**: Reporting Engine
