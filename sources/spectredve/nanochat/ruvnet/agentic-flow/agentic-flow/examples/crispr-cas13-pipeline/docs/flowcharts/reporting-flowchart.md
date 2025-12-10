# Reporting Engine - Flowchart

## Main Reporting Workflow

```mermaid
flowchart TD
    Start([Start: Analysis<br/>Results]) --> AggregateData[Aggregate Data from<br/>All Modules]

    AggregateData --> ExtractMetadata[Extract Project<br/>Metadata]

    ExtractMetadata --> CalcAlignStats[Calculate Alignment<br/>Statistics]

    CalcAlignStats --> CalcOffTargetStats[Calculate Off-Target<br/>Statistics]

    CalcOffTargetStats --> CalcExprStats[Calculate Expression<br/>Statistics]

    CalcExprStats --> CalcQualityMetrics[Calculate Quality<br/>Metrics]

    CalcQualityMetrics --> GenVisualizations[Generate<br/>Visualizations]

    GenVisualizations --> ParallelPlots{Parallel Plot<br/>Generation}

    ParallelPlots --> PlotGroup1[Group 1: Alignment<br/>Pie, Histogram]
    ParallelPlots --> PlotGroup2[Group 2: Off-Target<br/>Violin, Bar Charts]
    ParallelPlots --> PlotGroup3[Group 3: Volcano Plot<br/>MA Plot, Heatmap]
    ParallelPlots --> PlotGroup4[Group 4: Pathway Dots<br/>PCA, Browser Tracks]

    PlotGroup1 --> MergePlots[Merge All<br/>Visualizations]
    PlotGroup2 --> MergePlots
    PlotGroup3 --> MergePlots
    PlotGroup4 --> MergePlots

    MergePlots --> GenHTML[Generate HTML<br/>Report]

    GenHTML --> WriteHTML[Write HTML<br/>to File]

    WriteHTML --> CheckPDF{Generate<br/>PDF?}

    CheckPDF -->|Yes| ConvertPDF[Convert HTML<br/>to PDF]
    CheckPDF -->|No| ExportData

    ConvertPDF --> ExportData[Export Data Files<br/>VCF, BED, CSV]

    ExportData --> WriteSummary[Write Summary<br/>Text File]

    WriteSummary --> CalcTiming[Calculate Total<br/>Processing Time]

    CalcTiming --> PackageBundle[Package Report<br/>Bundle]

    PackageBundle --> End([End: Report<br/>Bundle])

    style Start fill:#e1f5e1
    style End fill:#ffe1e1
    style ParallelPlots fill:#fff4e1
    style CheckPDF fill:#fff4e1
    style GenHTML fill:#e1f0ff
```

## Data Aggregation Detail

```mermaid
flowchart TD
    StartAgg([Start: Module<br/>Results]) --> InitReport[Initialize Report<br/>Data Structure]

    InitReport --> AddMetadata[Add Project<br/>Metadata]

    AddMetadata --> ProcessAlignment[Process Alignment<br/>Results]

    ProcessAlignment --> ExtractReads[Extract Read<br/>Statistics]
    ExtractReads --> CalcCoverage[Calculate Coverage<br/>Statistics]
    CalcCoverage --> CalcMAPQ[Calculate Average<br/>MAPQ]

    CalcMAPQ --> CheckOffTarget{Off-Target<br/>Available?}

    CheckOffTarget -->|Yes| ProcessOffTarget[Process Off-Target<br/>Results]
    CheckOffTarget -->|No| CheckExpression

    ProcessOffTarget --> ExtractGRNAs[Extract gRNA<br/>Statistics]
    ExtractGRNAs --> ExtractCandidates[Extract Candidate<br/>Counts]
    ExtractCandidates --> ExtractTopGenes[Extract Top<br/>Off-Target Genes]

    ExtractTopGenes --> CheckExpression{Expression<br/>Available?}

    CheckExpression -->|Yes| ProcessExpression[Process Expression<br/>Results]
    CheckExpression -->|No| CalcQuality

    ProcessExpression --> ExtractDEGs[Extract DEG<br/>Statistics]
    ExtractDEGs --> ExtractPathways[Extract Pathway<br/>Enrichment]
    ExtractPathways --> ExtractImmuneScore[Extract Immune<br/>Score]

    ExtractImmuneScore --> CalcQuality[Calculate Quality<br/>Metrics]

    CalcQuality --> CalcCorrelations[Calculate Sample<br/>Correlations]
    CalcCorrelations --> DetectBatch[Detect Batch<br/>Effects]

    DetectBatch --> AddProcStats[Add Processing<br/>Statistics]

    AddProcStats --> EndAgg([Return: Aggregated<br/>Report Data])

    style StartAgg fill:#e1f5e1
    style EndAgg fill:#ffe1e1
    style CheckOffTarget fill:#fff4e1
    style CheckExpression fill:#fff4e1
```

## Volcano Plot Generation

```mermaid
flowchart TD
    StartVolcano([Start: Expression<br/>Data]) --> InitPlot[Initialize Plot<br/>Structure]

    InitPlot --> LoopGenes{For Each<br/>Gene}

    LoopGenes --> ExtractLFC[Extract log2<br/>Fold Change]

    ExtractLFC --> CalcY[Calculate Y<br/>= -log10 padj]

    CalcY --> CheckSig{|LFC| >= threshold<br/>AND padj < 0.05?}

    CheckSig -->|Yes, Up| SetRed[Color = Red<br/>Size = 8]
    CheckSig -->|Yes, Down| SetBlue[Color = Blue<br/>Size = 8]
    CheckSig -->|No| SetGray[Color = Gray<br/>Size = 4]

    SetRed --> AddPoint
    SetBlue --> AddPoint
    SetGray --> AddPoint[Add Point to<br/>Plot]

    AddPoint --> CheckRank{Gene Rank<br/><= 10?}

    CheckRank -->|Yes| AddLabel[Add Gene Symbol<br/>Annotation]
    CheckRank -->|No| NextGene

    AddLabel --> NextGene[Next Gene]
    NextGene --> LoopGenes

    LoopGenes -->|Done| AddThresholds[Add Threshold<br/>Lines]

    AddThresholds --> AddVLine[Add Vertical Lines<br/>at ±LFC threshold]

    AddVLine --> AddHLine[Add Horizontal Line<br/>at padj threshold]

    AddHLine --> SetLabels[Set Axis Labels<br/>& Title]

    SetLabels --> EndVolcano([Return: Volcano<br/>Plot])

    style StartVolcano fill:#e1f5e1
    style EndVolcano fill:#ffe1e1
    style LoopGenes fill:#fff4e1
    style CheckSig fill:#fff4e1
    style CheckRank fill:#fff4e1
```

## Heatmap with Hierarchical Clustering

```mermaid
flowchart TD
    StartHeatmap([Start: Top DEGs]) --> ExtractMatrix[Extract Expression<br/>Matrix]

    ExtractMatrix --> ZScoreNorm[Z-score Normalization<br/>Per Gene]

    ZScoreNorm --> CheckCluster{Clustering<br/>Enabled?}

    CheckCluster -->|No| CreateHeatmap
    CheckCluster -->|Yes| ClusterGenes[Hierarchical Clustering<br/>Genes Euclidean]

    ClusterGenes --> ClusterSamples[Hierarchical Clustering<br/>Samples Correlation]

    ClusterSamples --> GetGeneOrder[Get Dendrogram<br/>Gene Order]

    GetGeneOrder --> GetSampleOrder[Get Dendrogram<br/>Sample Order]

    GetSampleOrder --> ReorderMatrix[Reorder Matrix<br/>by Dendrograms]

    ReorderMatrix --> ReorderLabels[Reorder Row/Col<br/>Labels]

    ReorderLabels --> CreateHeatmap[Create Heatmap<br/>with Color Scheme]

    CreateHeatmap --> AddDendro{Add<br/>Dendrograms?}

    AddDendro -->|Yes| AttachTrees[Attach Gene &<br/>Sample Trees]
    AddDendro -->|No| AddColorBar

    AttachTrees --> AddColorBar[Add Color Bar<br/>Z-score Scale]

    AddColorBar --> AddLabels[Add Row & Column<br/>Labels]

    AddLabels --> EndHeatmap([Return: Heatmap<br/>Plot])

    style StartHeatmap fill:#e1f5e1
    style EndHeatmap fill:#ffe1e1
    style CheckCluster fill:#fff4e1
    style AddDendro fill:#fff4e1
    style ClusterGenes fill:#e1f0ff
```

## Pathway Dot Plot Generation

```mermaid
flowchart TD
    StartDot([Start: Pathway<br/>Results]) --> SortPathways[Sort by Adjusted<br/>P-value]

    SortPathways --> TakeTop[Take Top N<br/>Pathways]

    TakeTop --> InitPlot[Initialize Dot<br/>Plot]

    InitPlot --> LoopPathways{For Each<br/>Pathway}

    LoopPathways --> CalcGeneRatio[Calculate Gene Ratio<br/>Overlap / Pathway Size]

    CalcGeneRatio --> CalcColor[Calculate Color<br/>= -log10 FDR]

    CalcColor --> CalcSize[Calculate Size<br/>= Overlap × 2]

    CalcSize --> AddDot[Add Dot to Plot<br/>Y = Pathway Name]

    AddDot --> AddTooltip[Add Tooltip<br/>with Details]

    AddTooltip --> NextPathway[Next Pathway]
    NextPathway --> LoopPathways

    LoopPathways -->|Done| ConfigColorScale[Configure Color Scale<br/>Viridis]

    ConfigColorScale --> ConfigSizeScale[Configure Size Scale<br/>5 to 20]

    ConfigSizeScale --> SetAxes[Set Axis Labels<br/>& Title]

    SetAxes --> AddLegend[Add Color & Size<br/>Legends]

    AddLegend --> EndDot([Return: Dot<br/>Plot])

    style StartDot fill:#e1f5e1
    style EndDot fill:#ffe1e1
    style LoopPathways fill:#fff4e1
```

## HTML Report Generation

```mermaid
flowchart TD
    StartHTML([Start: Report Data<br/>& Plots]) --> WriteHeader[Write HTML Header<br/>& CSS Styles]

    WriteHeader --> WriteTitle[Write Title &<br/>Metadata Section]

    WriteTitle --> WriteExecSummary[Write Executive<br/>Summary Cards]

    WriteExecSummary --> WriteTOC[Write Table of<br/>Contents]

    WriteTOC --> WriteSection1[Write Section 1:<br/>Alignment Analysis]

    WriteSection1 --> EmbedAlignPlots[Embed Alignment<br/>Visualizations]

    EmbedAlignPlots --> CheckOffTarget{Off-Target<br/>Section?}

    CheckOffTarget -->|Yes| WriteSection2[Write Section 2:<br/>Off-Target Prediction]
    CheckOffTarget -->|No| CheckExpression

    WriteSection2 --> EmbedOffTargetPlots[Embed Off-Target<br/>Visualizations]

    EmbedOffTargetPlots --> CheckExpression{Expression<br/>Section?}

    CheckExpression -->|Yes| WriteSection3[Write Section 3:<br/>Differential Expression]
    CheckExpression -->|No| WriteQuality

    WriteSection3 --> EmbedExprPlots[Embed Expression<br/>Visualizations]

    EmbedExprPlots --> WriteSection4[Write Section 4:<br/>Pathway Enrichment]

    WriteSection4 --> EmbedPathwayPlots[Embed Pathway<br/>Visualizations]

    EmbedPathwayPlots --> WriteQuality[Write Section 5:<br/>Quality Metrics]

    WriteQuality --> WriteMethods[Write Section 6:<br/>Methods]

    WriteMethods --> WriteFooter[Write Footer<br/>& Closing Tags]

    WriteFooter --> EndHTML([Return: HTML<br/>String])

    style StartHTML fill:#e1f5e1
    style EndHTML fill:#ffe1e1
    style CheckOffTarget fill:#fff4e1
    style CheckExpression fill:#fff4e1
```

## Executive Summary Card Generation

```mermaid
flowchart TD
    StartCard([Start: Report<br/>Statistics]) --> InitGrid[Initialize Summary<br/>Grid Container]

    InitGrid --> CreateAlignCard[Create Alignment<br/>Metrics Card]

    CreateAlignCard --> AddAlignValue[Add Alignment Rate<br/>Large Value]
    AddAlignValue --> AddAlignLabel[Add Descriptive<br/>Label]
    AddAlignLabel --> AddAlignDetail[Add Read Count<br/>Details]

    AddAlignDetail --> CheckOffTarget{Off-Target<br/>Available?}

    CheckOffTarget -->|Yes| CreateOffTargetCard[Create Off-Target<br/>Metrics Card]
    CheckOffTarget -->|No| CheckExpression

    CreateOffTargetCard --> AddOffTargetValue[Add High-Confidence<br/>Count]
    AddOffTargetValue --> DetermineRisk[Determine Risk<br/>Class Color]
    DetermineRisk --> AddOffTargetDetail[Add gRNA Count<br/>Details]

    AddOffTargetDetail --> CheckExpression{Expression<br/>Available?}

    CheckExpression -->|Yes| CreateExprCard[Create Expression<br/>Metrics Card]
    CheckExpression -->|No| WrapGrid

    CreateExprCard --> AddDEGValue[Add Significant<br/>DEG Count]
    AddDEGValue --> AddUpDown[Add Up/Down<br/>Regulation Counts]

    AddUpDown --> CreateImmuneCard[Create Immune Score<br/>Card]

    CreateImmuneCard --> AddScoreValue[Add Immune Score<br/>0-100]
    AddScoreValue --> DetermineScoreClass[Determine Score<br/>Class Color]
    DetermineScoreClass --> AddInterpretation[Add Score<br/>Interpretation]

    AddInterpretation --> WrapGrid[Wrap Summary<br/>Grid]

    WrapGrid --> EndCard([Return: Summary<br/>HTML])

    style StartCard fill:#e1f5e1
    style EndCard fill:#ffe1e1
    style CheckOffTarget fill:#fff4e1
    style CheckExpression fill:#fff4e1
```

## Data Export (VCF, BED, CSV)

```mermaid
flowchart TD
    StartExport([Start: Report<br/>Data]) --> InitFiles[Initialize Export<br/>File List]

    InitFiles --> CheckOffTarget{Off-Target<br/>Data?}

    CheckOffTarget -->|Yes| ExportVCF[Export VCF Format<br/>Off-Target Sites]
    CheckOffTarget -->|No| ExportBED

    ExportVCF --> WriteVCFHeader[Write VCF Header<br/>with Metadata]
    WriteVCFHeader --> LoopgRNAs{For Each<br/>gRNA}

    LoopgRNAs --> LoopSites{For Each<br/>Off-Target Site}

    LoopSites --> WriteVCFLine[Write VCF Line<br/>CHROM, POS, INFO]
    WriteVCFLine --> NextSite[Next Site]
    NextSite --> LoopSites

    LoopSites -->|Done| NextgRNA[Next gRNA]
    NextgRNA --> LoopgRNAs

    LoopgRNAs -->|Done| CloseVCF[Close VCF File]

    CloseVCF --> ExportBED[Export BED Format<br/>Coverage Regions]

    ExportBED --> WriteBEDHeader[Write BED Track<br/>Definition]
    WriteBEDHeader --> LoopRegions{For Each<br/>Coverage Region}

    LoopRegions --> WriteBEDLine[Write BED Line<br/>chrom, start, end]
    WriteBEDLine --> NextRegion[Next Region]
    NextRegion --> LoopRegions

    LoopRegions -->|Done| CloseBED[Close BED File]

    CloseBED --> CheckExpression{Expression<br/>Data?}

    CheckExpression -->|Yes| ExportCSV[Export CSV Format<br/>DE Results]
    CheckExpression -->|No| ExportJSON

    ExportCSV --> WriteCSVHeader[Write CSV Header<br/>Column Names]
    WriteCSVHeader --> LoopGenes{For Each<br/>Gene}

    LoopGenes --> WriteCSVLine[Write CSV Line<br/>All Columns]
    WriteCSVLine --> NextGene[Next Gene]
    NextGene --> LoopGenes

    LoopGenes -->|Done| CloseCSV[Close CSV File]

    CloseCSV --> ExportJSON[Export JSON Format<br/>Full Report Data]

    ExportJSON --> WriteJSON[Write JSON with<br/>Pretty Formatting]

    WriteJSON --> EndExport([Return: Exported<br/>File Paths])

    style StartExport fill:#e1f5e1
    style EndExport fill:#ffe1e1
    style CheckOffTarget fill:#fff4e1
    style CheckExpression fill:#fff4e1
    style LoopgRNAs fill:#fff4e1
    style LoopSites fill:#fff4e1
    style LoopRegions fill:#fff4e1
    style LoopGenes fill:#fff4e1
```

## PDF Generation Workflow

```mermaid
flowchart TD
    StartPDF([Start: HTML<br/>Report]) --> CheckRenderer{PDF Renderer<br/>Available?}

    CheckRenderer -->|WeasyPrint| UseWeasy[Use WeasyPrint<br/>HTML → PDF]
    CheckRenderer -->|Puppeteer| UsePuppeteer[Use Puppeteer<br/>Headless Chrome]
    CheckRenderer -->|None| ErrorNoPDF[Error: No PDF<br/>Renderer]

    UseWeasy --> ConfigWeasy[Configure Page Size<br/>A4 Margins]
    UsePuppeteer --> LaunchBrowser[Launch Headless<br/>Browser]

    ConfigWeasy --> RenderWeasy[Render HTML<br/>to PDF]
    LaunchBrowser --> LoadHTML[Load HTML<br/>Content]

    LoadHTML --> WaitRender[Wait for<br/>Rendering]
    WaitRender --> ConfigPuppeteer[Configure PDF<br/>Options]
    ConfigPuppeteer --> RenderPuppeteer[Generate PDF<br/>File]
    RenderPuppeteer --> CloseBrowser[Close Browser<br/>Instance]

    RenderWeasy --> WritePDF
    CloseBrowser --> WritePDF[Write PDF<br/>to File]

    WritePDF --> VerifyFile{PDF File<br/>Created?}

    VerifyFile -->|Yes| LogSuccess[Log Success<br/>& File Path]
    VerifyFile -->|No| ErrorWrite[Error: Write<br/>Failed]

    LogSuccess --> EndPDF([Return: PDF<br/>File Path])

    ErrorNoPDF --> Fail([Return: null])
    ErrorWrite --> Fail

    style StartPDF fill:#e1f5e1
    style EndPDF fill:#ffe1e1
    style Fail fill:#ffe1e1
    style CheckRenderer fill:#fff4e1
    style VerifyFile fill:#fff4e1
```

## Error Handling in Reporting

```mermaid
flowchart TD
    Operation[Execute Reporting<br/>Operation] --> CheckData{All Required<br/>Data Available?}

    CheckData -->|No| CreatePlaceholder[Create Placeholder<br/>Sections]
    CheckData -->|Yes| TryVisualization

    CreatePlaceholder --> LogMissing[Log Missing<br/>Data Warning]

    LogMissing --> TryVisualization{Try Generate<br/>Visualizations}

    TryVisualization -->|Success| TryHTML
    TryVisualization -->|Plot Error| CatchPlotError[Catch Visualization<br/>Error]

    CatchPlotError --> UsePlaceholderPlot[Use Placeholder<br/>Plot]
    UsePlaceholderPlot --> LogPlotError[Log Plot<br/>Error]
    LogPlotError --> TryHTML

    TryHTML{Try Generate<br/>HTML} -->|Success| TryPDF
    TryHTML -->|Error| CatchHTMLError[Catch HTML<br/>Error]

    CatchHTMLError --> LogHTMLError[Log HTML Error<br/>& Stack Trace]
    LogHTMLError --> Fail

    TryPDF{Try Generate<br/>PDF} -->|Success| Success
    TryPDF -->|Error| CatchPDFError[Catch PDF<br/>Error]
    TryPDF -->|Skipped| Success

    CatchPDFError --> LogPDFError[Log PDF Error<br/>Continue Without PDF]
    LogPDFError --> Success([Success with<br/>Warnings])

    Fail([Partial Failure])

    style Success fill:#e1f5e1
    style Fail fill:#ffe1e1
    style CheckData fill:#fff4e1
    style TryVisualization fill:#fff4e1
    style TryHTML fill:#fff4e1
    style TryPDF fill:#fff4e1
```

---

## Complexity Summary

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Data Aggregation | O(G + S) | O(G + S) |
| Volcano Plot | O(G) | O(G) |
| Heatmap (no clustering) | O(G × S) | O(G × S) |
| Heatmap (with clustering) | O(G² log G + S² log S) | O(G × S) |
| Dot Plot | O(P) | O(P) |
| HTML Generation | O(S × P) | O(H) |
| PDF Rendering | O(N × P) | O(H) |
| VCF Export | O(K) | O(K) |
| CSV Export | O(G) | O(G) |
| Full Pipeline | O(G × S + G² log G) | O(G × S + H) |

Where:
- G = number of genes
- S = number of samples
- P = number of plots/pathways
- K = number of off-target sites
- H = HTML size
- N = number of pages
