# Differential Expression Analysis Module - Flowchart

## Main Expression Analysis Workflow

```mermaid
flowchart TD
    Start([Start: Count Matrix]) --> LoadData[Load Count Matrix<br/>& Sample Metadata]

    LoadData --> ValidateMatch[Validate Sample<br/>Matching]

    ValidateMatch --> QCFilter[Quality Control<br/>Filter Low Counts]

    QCFilter --> LogFiltered[Log Filtered<br/>Gene Count]

    LogFiltered --> Normalize[DESeq2 Normalization<br/>Size Factor Estimation]

    Normalize --> EstimateDisp[Estimate Gene-wise<br/>Dispersions]

    EstimateDisp --> FitTrend[Fit Dispersion-Mean<br/>Trend]

    FitTrend --> ShrinkDisp[Shrink Dispersions<br/>Empirical Bayes]

    ShrinkDisp --> TestDE[Differential Expression<br/>Testing Wald Test]

    TestDE --> AdjustPValues[Multiple Testing<br/>Correction BH FDR]

    AdjustPValues --> ShrinkLFC[Shrink Log2 Fold<br/>Changes apeglm]

    ShrinkLFC --> ExtractSig[Extract Significant<br/>DEGs FDR < 0.05]

    ExtractSig --> PathwayEnrich[Immune Pathway<br/>Enrichment Analysis]

    PathwayEnrich --> CalcImmuneScore[Calculate Immune<br/>Activation Score]

    CalcImmuneScore --> GenStats[Generate Summary<br/>Statistics]

    GenStats --> End([End: DE Results])

    style Start fill:#e1f5e1
    style End fill:#ffe1e1
    style Normalize fill:#e1f0ff
    style EstimateDisp fill:#e1f0ff
    style TestDE fill:#e1f0ff
    style PathwayEnrich fill:#e1f0ff
```

## DESeq2 Normalization Detail

```mermaid
flowchart TD
    StartNorm([Start: Raw Counts]) --> CalcGeoMeans[Calculate Geometric<br/>Mean per Gene]

    CalcGeoMeans --> LoopSamples{For Each<br/>Sample}

    LoopSamples --> CalcRatios[Calculate Ratios<br/>Count / GeoMean]

    CalcRatios --> FilterZeros[Filter Out<br/>Zero GeoMeans]

    FilterZeros --> MedianRatio[Median of Ratios<br/>= Size Factor]

    MedianRatio --> StoreSF[Store Size<br/>Factor]

    StoreSF --> NextSample[Next Sample]
    NextSample --> LoopSamples

    LoopSamples -->|Done| DivideBy SF[Divide Counts by<br/>Size Factors]

    DivideBySF --> LogNormCounts[Log Normalized<br/>Counts]

    LogNormCounts --> EndNorm([Return: Normalized<br/>Counts & SFs])

    style StartNorm fill:#e1f5e1
    style EndNorm fill:#ffe1e1
    style LoopSamples fill:#fff4e1
```

## Dispersion Estimation Workflow

```mermaid
flowchart TD
    StartDisp([Start: Normalized<br/>Counts]) --> ParseDesign[Parse Design Formula<br/>Create Design Matrix]

    ParseDesign --> LoopGenes{For Each<br/>Gene}

    LoopGenes --> FitGLM[Fit Negative Binomial<br/>GLM IRLS]

    FitGLM --> CheckConverge{GLM<br/>Converged?}

    CheckConverge -->|No| LogWarning[Log Convergence<br/>Warning]
    CheckConverge -->|Yes| ExtractDisp

    LogWarning --> ExtractDisp[Extract Gene<br/>Dispersion]

    ExtractDisp --> ExtractMu[Extract Gene<br/>Mean μ]

    ExtractMu --> NextGene[Next Gene]
    NextGene --> LoopGenes

    LoopGenes -->|Done| FitTrend[Fit Parametric Trend<br/>log disp = a/μ + b]

    FitTrend --> LogTrend[Log Trend<br/>Parameters]

    LogTrend --> LoopShrink{For Each<br/>Gene}

    LoopShrink --> CalcWeight[Calculate Shrinkage<br/>Weight]

    CalcWeight --> ShrinkToTrend[Shrink Toward<br/>Trend Value]

    ShrinkToTrend --> ClampDisp[Clamp to Valid<br/>Range 1e-8 to 10]

    ClampDisp --> StoreDisp[Store Final<br/>Dispersion]

    StoreDisp --> NextGene2[Next Gene]
    NextGene2 --> LoopShrink

    LoopShrink -->|Done| EndDisp([Return: Dispersion<br/>Estimates])

    style StartDisp fill:#e1f5e1
    style EndDisp fill:#ffe1e1
    style LoopGenes fill:#fff4e1
    style CheckConverge fill:#fff4e1
    style LoopShrink fill:#fff4e1
    style FitGLM fill:#e1f0ff
```

## Negative Binomial GLM Fitting (IRLS)

```mermaid
flowchart TD
    StartGLM([Start: Gene Counts]) --> InitBeta[Initialize Coefficients<br/>β = 0]

    InitBeta --> InitDisp[Initialize Dispersion<br/>α = 0.1]

    InitDisp --> IterLoop{Iteration < Max<br/>100?}

    IterLoop -->|Yes| CalcMu[Calculate μ<br/>= exp Xβ]

    CalcMu --> CalcVar[Calculate Variance<br/>= μ + α×μ²]

    CalcVar --> CalcWeights[Calculate Weights<br/>= μ² / variance]

    CalcWeights --> WeightedLS[Weighted Least Squares<br/>Update β]

    WeightedLS --> CheckDelta{|Δβ| <<br/>Threshold?}

    CheckDelta -->|Yes| EstimateAlpha[Estimate α<br/>Maximum Likelihood]
    CheckDelta -->|No| NextIter

    EstimateAlpha --> SetConverged[Set Converged<br/>= True]
    SetConverged --> EndGLM

    NextIter[Iteration++] --> IterLoop

    IterLoop -->|No| SetNotConverged[Set Converged<br/>= False]

    SetNotConverged --> EndGLM([Return: GLM Result<br/>β, μ, α])

    style StartGLM fill:#e1f5e1
    style EndGLM fill:#ffe1e1
    style IterLoop fill:#fff4e1
    style CheckDelta fill:#fff4e1
    style WeightedLS fill:#e1f0ff
```

## Differential Expression Testing

```mermaid
flowchart TD
    StartTest([Start: Normalized<br/>Counts]) --> ParseContrast[Parse Contrast<br/>treated vs control]

    ParseContrast --> LoopGenes{For Each<br/>Gene}

    LoopGenes --> FitFullModel[Fit Full Model<br/>Negative Binomial GLM]

    FitFullModel --> ExtractCoef[Extract Coefficients<br/>β]

    ExtractCoef --> CalcLFC[Calculate log2FC<br/>β^T × contrast / log 2]

    CalcLFC --> CalcCov[Calculate Covariance<br/>Matrix Cov β]

    CalcCov --> CalcSE[Calculate Standard Error<br/>√ contrast^T Cov contrast]

    CalcSE --> WaldStat[Wald Statistic<br/>= log2FC / SE]

    WaldStat --> CalcPValue[P-value from<br/>Normal Distribution]

    CalcPValue --> CalcBaseMean[Calculate<br/>Base Mean]

    CalcBaseMean --> StoreResult[Store Gene<br/>Result]

    StoreResult --> NextGene[Next Gene]
    NextGene --> LoopGenes

    LoopGenes -->|Done| SortResults[Sort by<br/>P-value]

    SortResults --> BHCorrection[Benjamini-Hochberg<br/>Correction]

    BHCorrection --> AddAnnotations[Add Gene<br/>Annotations]

    AddAnnotations --> EndTest([Return: DE Results<br/>DataFrame])

    style StartTest fill:#e1f5e1
    style EndTest fill:#ffe1e1
    style LoopGenes fill:#fff4e1
    style FitFullModel fill:#e1f0ff
```

## Benjamini-Hochberg FDR Correction

```mermaid
flowchart TD
    StartBH([Start: P-values]) --> CreateIndex[Create Indexed<br/>P-value Array]

    CreateIndex --> SortAsc[Sort by P-value<br/>Ascending]

    SortAsc --> LoopDesc{Loop from End<br/>to Start}

    LoopDesc --> CalcAdjusted[Adjusted =<br/>p × N / rank]

    CalcAdjusted --> CheckMonotone{Adjusted <<br/>minAdjusted?}

    CheckMonotone -->|Yes| UseAdjusted[Use Calculated<br/>Value]
    CheckMonotone -->|No| UseMin[Use minAdjusted<br/>Maintain Monotonicity]

    UseAdjusted --> ClampOne
    UseMin --> ClampOne[Clamp to<br/>Maximum 1.0]

    ClampOne --> StoreAdj[Store Adjusted<br/>P-value]

    StoreAdj --> UpdateMin[Update<br/>minAdjusted]

    UpdateMin --> PrevRank[Previous Rank]
    PrevRank --> LoopDesc

    LoopDesc -->|Done| ReorderOrig[Reorder to Original<br/>Indices]

    ReorderOrig --> EndBH([Return: Adjusted<br/>P-values])

    style StartBH fill:#e1f5e1
    style EndBH fill:#ffe1e1
    style LoopDesc fill:#fff4e1
    style CheckMonotone fill:#fff4e1
```

## Pathway Enrichment Analysis

```mermaid
flowchart TD
    StartPathway([Start: DEG List]) --> FilterSig[Filter Significant<br/>DEGs FDR < 0.05]

    FilterSig --> SplitDirection[Split Upregulated<br/>& Downregulated]

    SplitDirection --> LoadPathways[Load Immune<br/>Pathway Database]

    LoadPathways --> FilterImmune[Filter to Immune<br/>Categories]

    FilterImmune --> LoopPathways{For Each<br/>Pathway}

    LoopPathways --> CalcOverlap[Calculate Overlap<br/>DEGs ∩ Pathway]

    CalcOverlap --> HyperTest[Hypergeometric Test<br/>P X >= k]

    HyperTest --> CalcExpected[Calculate Expected<br/>Overlap]

    CalcExpected --> CalcFoldEnrich[Fold Enrichment<br/>= Observed / Expected]

    CalcFoldEnrich --> CountDirection[Count Up/Down<br/>Regulated in Overlap]

    CountDirection --> StorePathway[Store Pathway<br/>Result]

    StorePathway --> NextPathway[Next Pathway]
    NextPathway --> LoopPathways

    LoopPathways -->|Done| BHAdjust[Benjamini-Hochberg<br/>Adjustment]

    BHAdjust --> SortByFDR[Sort by<br/>Adjusted P-value]

    SortByFDR --> CalcScore[Calculate Immune<br/>Score 0-100]

    CalcScore --> EndPathway([Return: Enrichment<br/>Results])

    style StartPathway fill:#e1f5e1
    style EndPathway fill:#ffe1e1
    style LoopPathways fill:#fff4e1
    style HyperTest fill:#e1f0ff
```

## Immune Score Calculation

```mermaid
flowchart TD
    StartScore([Start: Enriched<br/>Pathways]) --> FilterSigPath[Filter Pathways<br/>FDR < 0.05]

    FilterSigPath --> CheckCount{Significant<br/>Pathways?}

    CheckCount -->|No| ReturnZero[Return Score = 0]

    CheckCount -->|Yes| CalcPathwayScore[Pathway Score<br/>= Count × 3 max 33]

    CalcPathwayScore --> CalcEnrichScore[Enrichment Score<br/>= AvgFold × 10 max 33]

    CalcEnrichScore --> CalcUpregRatio[Calculate Upreg<br/>Ratio per Pathway]

    CalcUpregRatio --> AvgUpregRatio[Average Upreg<br/>Ratio]

    AvgUpregRatio --> ClassifyDirection{Upreg Ratio<br/>> 0.6?}

    ClassifyDirection -->|Yes| HighActivation[Direction Score = 34<br/>Strong Activation]
    ClassifyDirection -->|Between| MixedResponse[Direction Score = 17<br/>Mixed Response]
    ClassifyDirection -->|No| Suppression[Direction Score = 0<br/>Suppression]

    HighActivation --> SumScores
    MixedResponse --> SumScores
    Suppression --> SumScores[Sum All Components<br/>Max 100]

    SumScores --> EndScore([Return: Immune<br/>Score])

    ReturnZero --> EndScore

    style StartScore fill:#e1f5e1
    style EndScore fill:#ffe1e1
    style CheckCount fill:#fff4e1
    style ClassifyDirection fill:#fff4e1
```

## Log Fold Change Shrinkage

```mermaid
flowchart TD
    StartShrink([Start: DE Results]) --> EstimatePrior[Estimate Prior<br/>Precision from Data]

    EstimatePrior --> LoopGenes{For Each<br/>Gene}

    LoopGenes --> GetObserved[Get Observed<br/>LFC & SE]

    GetObserved --> CalcPrecision[Calculate Precision<br/>= 1 / SE²]

    CalcPrecision --> PosteriorPrec[Posterior Precision<br/>= Obs + Prior]

    PosteriorPrec --> PosteriorMean[Posterior Mean<br/>Weighted Average]

    PosteriorMean --> PosteriorSE[Posterior SE<br/>= √ 1/PostPrec]

    PosteriorSE --> UpdateLFC[Update Gene<br/>LFC & SE]

    UpdateLFC --> RecalcStat[Recalculate<br/>Wald Statistic]

    RecalcStat --> RecalcPValue[Recalculate<br/>P-value]

    RecalcPValue --> NextGene[Next Gene]
    NextGene --> LoopGenes

    LoopGenes -->|Done| BHAdjust[Benjamini-Hochberg<br/>Re-adjustment]

    BHAdjust --> EndShrink([Return: Shrunk<br/>Results])

    style StartShrink fill:#e1f5e1
    style EndShrink fill:#ffe1e1
    style LoopGenes fill:#fff4e1
```

## Error Handling & Quality Control

```mermaid
flowchart TD
    Operation[Execute<br/>Expression Analysis] --> ValidateInput{Valid Input<br/>Data?}

    ValidateInput -->|No| CheckNegative{Negative<br/>Counts?}
    ValidateInput -->|Yes| CheckSamples

    CheckNegative -->|Yes| ErrorNeg[Error: Invalid<br/>Count Data]
    CheckNegative -->|No| CheckZero

    CheckZero{All-Zero<br/>Genes/Samples?} -->|Many| WarnFilter[Warning: Heavy<br/>Filtering Needed]
    CheckZero -->|Few| CheckSamples

    WarnFilter --> CheckSamples{Sample Count<br/>Adequate?}

    CheckSamples -->|< 3 per group| ErrorSample[Error: Insufficient<br/>Samples]
    CheckSamples -->|>= 3| TryOperation

    TryOperation{Try GLM<br/>Fitting} -->|Success| Success([Success])

    TryOperation -->|Convergence Error| RobustFit[Switch to Robust<br/>Fitting]
    TryOperation -->|Singular Matrix| PseudoInverse[Use Pseudo-inverse<br/>for Covariance]
    TryOperation -->|Numerical Error| LogError[Log Error<br/>Skip Gene]

    RobustFit --> Success
    PseudoInverse --> Success
    LogError --> WarnUser[Warn User About<br/>Failed Genes]

    WarnUser --> PartialSuccess([Partial Success])

    ErrorNeg --> Fail([Fail])
    ErrorSample --> Fail

    style Success fill:#e1f5e1
    style PartialSuccess fill:#fff4e1
    style Fail fill:#ffe1e1
    style ValidateInput fill:#fff4e1
    style CheckNegative fill:#fff4e1
    style CheckZero fill:#fff4e1
    style CheckSamples fill:#fff4e1
    style TryOperation fill:#fff4e1
```

---

## Complexity Summary

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Size Factor Estimation | O(G × S) | O(G × S) |
| Dispersion Estimation | O(G × S × I × C²) | O(G × S) |
| GLM Fitting per Gene | O(S × I × C²) | O(S × C) |
| Wald Testing | O(G × S × C²) | O(G × S) |
| BH Correction | O(G log G) | O(G) |
| Pathway Enrichment | O(P × G) | O(P × G) |
| Full Pipeline | O(G × S × I × C²) | O(G × S) |

Where:
- G = number of genes (20,000)
- S = number of samples (10-100)
- I = IRLS iterations (typically 5-20)
- C = number of coefficients (2-10)
- P = number of pathways (100-500)
