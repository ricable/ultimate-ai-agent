# Reporting Engine - Detailed Pseudocode

## Overview

The Reporting Engine aggregates results from all pipeline modules and generates comprehensive visualizations, interactive reports, and data exports in standard bioinformatics formats.

---

## Algorithm 1: Report Data Aggregation

**Purpose**: Collect and structure data from all pipeline components.

```
ALGORITHM: AggregateReportData
INPUT: alignmentResults, offTargetResults, expressionResults, config
OUTPUT: aggregatedData (ReportData object)

BEGIN
    reportData ← {
        metadata: {},
        alignment: {},
        offTarget: {},
        expression: {},
        qualityMetrics: {},
        timestamp: GetCurrentTime()
    }

    // Step 1: Extract metadata
    reportData.metadata ← {
        projectName: config.projectName,
        analysisDate: FormatDate(GetCurrentTime()),
        pipelineVersion: config.version,
        referenceGenome: config.referenceGenome,
        organism: config.organism,
        experimentType: config.experimentType
    }

    // Step 2: Aggregate alignment statistics
    reportData.alignment ← {
        totalReads: alignmentResults.statistics.totalReads,
        alignedReads: alignmentResults.statistics.alignedReads,
        alignmentRate: alignmentResults.statistics.alignmentRate × 100,
        unmappedReads: alignmentResults.statistics.unmappedReads,
        multiMappedReads: alignmentResults.statistics.multiMappedReads,
        averageAlignmentScore: CalculateAverageAlignmentScore(alignmentResults),
        averageMAPQ: CalculateAverageMAPQ(alignmentResults),
        coverageStatistics: CalculateCoverageStatistics(alignmentResults)
    }

    // Step 3: Aggregate off-target predictions
    IF offTargetResults != null THEN
        reportData.offTarget ← {
            totalGRNAs: length(offTargetResults.gRNAs),
            totalCandidateSites: SUM(length(g.candidates) for g in offTargetResults.gRNAs),
            totalPredictedOffTargets: SUM(length(g.predicted) for g in offTargetResults.gRNAs),
            highConfidenceOffTargets: COUNT_HIGH_CONFIDENCE(offTargetResults),
            topOffTargetGenes: ExtractTopOffTargetGenes(offTargetResults, topN = 20),
            averageOffTargetScore: CalculateAverageOffTargetScore(offTargetResults)
        }
    END IF

    // Step 4: Aggregate expression analysis
    IF expressionResults != null THEN
        reportData.expression ← {
            totalGenesTested: expressionResults.statistics.totalGenesTested,
            significantDEGs: expressionResults.statistics.significantGenes,
            upregulatedGenes: expressionResults.statistics.upregulatedGenes,
            downregulatedGenes: expressionResults.statistics.downregulatedGenes,
            topDEGs: ExtractTopDEGs(expressionResults, topN = 50),
            enrichedPathways: expressionResults.enrichment.pathways,
            immuneScore: expressionResults.enrichment.immuneScore,
            pathwayCategories: GroupPathwaysByCategory(expressionResults.enrichment)
        }
    END IF

    // Step 5: Calculate quality metrics
    reportData.qualityMetrics ← {
        dataQuality: AssessDataQuality(alignmentResults, expressionResults),
        sampleCorrelations: CalculateSampleCorrelations(expressionResults),
        replicateConsistency: CalculateReplicateConsistency(expressionResults),
        batchEffects: DetectBatchEffects(expressionResults)
    }

    // Step 6: Add processing statistics
    reportData.processingStats ← {
        alignmentTime: alignmentResults.statistics.processingTime,
        offTargetTime: offTargetResults.statistics.processingTime,
        expressionTime: expressionResults.statistics.processingTime,
        totalTime: alignmentResults.statistics.processingTime +
                   offTargetResults.statistics.processingTime +
                   expressionResults.statistics.processingTime,
        memoryPeak: GetPeakMemoryUsage()
    }

    LogInfo("Report data aggregation complete")

    RETURN reportData
END

SUBROUTINE: CalculateCoverageStatistics
INPUT: alignmentResults
OUTPUT: coverageStats (object)

BEGIN
    // Calculate genome-wide coverage statistics
    coverages ← ExtractPerBaseCoverage(alignmentResults.bamFile)

    coverageStats ← {
        meanCoverage: MEAN(coverages),
        medianCoverage: MEDIAN(coverages),
        coverage10x: PERCENTAGE(coverages >= 10),
        coverage30x: PERCENTAGE(coverages >= 30),
        coverage100x: PERCENTAGE(coverages >= 100),
        stdDev: STDEV(coverages),
        uniformity: CalculateUniformity(coverages)
    }

    RETURN coverageStats
END

SUBROUTINE: ExtractTopDEGs
INPUT: expressionResults, topN
OUTPUT: topGenes (array)

BEGIN
    // Extract top genes by statistical significance and effect size
    allGenes ← expressionResults.deResults

    // Filter to significant genes
    sigGenes ← [gene for gene in allGenes WHERE gene.padj < 0.05]

    // Sort by combined score (|log2FC| × -log10(padj))
    FOR EACH gene IN sigGenes DO
        gene.combinedScore ← ABS(gene.log2FoldChange) × (-LOG10(gene.padj))
    END FOR

    sortedGenes ← Sort(sigGenes, by = combinedScore, descending = true)

    topGenes ← sortedGenes[0 : MIN(topN, length(sortedGenes))]

    RETURN topGenes
END
```

**Complexity Analysis**:
- **Time**: O(G + S) where G = genes, S = samples
- **Space**: O(G + S) for statistics storage
- **Typical Performance**: 1-5 seconds

---

## Algorithm 2: Visualization Generation

**Purpose**: Create publication-quality plots and charts.

```
ALGORITHM: GenerateVisualizations
INPUT: reportData (ReportData object), config (VisualizationConfig)
OUTPUT: visualizations (array of Plot objects)

BEGIN
    visualizations ← []

    // === Alignment Visualizations ===

    // 1. Alignment rate pie chart
    alignmentPieChart ← CreatePieChart(
        data = [
            {label: "Aligned", value: reportData.alignment.alignedReads},
            {label: "Unmapped", value: reportData.alignment.unmappedReads}
        ],
        title = "Read Alignment Distribution",
        colors = ["#4CAF50", "#F44336"]
    )
    visualizations.append(alignmentPieChart)

    // 2. Coverage histogram
    coverageHistogram ← CreateHistogram(
        data = reportData.alignment.coverageStatistics.distribution,
        title = "Genome Coverage Distribution",
        xLabel = "Coverage Depth",
        yLabel = "Frequency",
        bins = 50
    )
    visualizations.append(coverageHistogram)

    // === Off-Target Visualizations ===

    IF reportData.offTarget != null THEN
        // 3. Off-target score distribution
        offTargetScoreDist ← CreateViolin Plot(
            data = ExtractOffTargetScores(reportData.offTarget),
            title = "Off-Target Score Distribution by gRNA",
            yLabel = "Off-Target Score"
        )
        visualizations.append(offTargetScoreDist)

        // 4. Top off-target genes bar chart
        topOffTargetsBar ← CreateBarChart(
            data = reportData.offTarget.topOffTargetGenes,
            xLabel = "Gene Symbol",
            yLabel = "Off-Target Score",
            title = "Top 20 Off-Target Genes",
            orientation = "horizontal",
            colorScale = "YlOrRd"
        )
        visualizations.append(topOffTargetsBar)
    END IF

    // === Expression Visualizations ===

    IF reportData.expression != null THEN
        // 5. Volcano plot
        volcanoPlot ← GenerateVolcanoPlot(
            expressionData = reportData.expression.topDEGs,
            log2FCThreshold = 1.0,
            padjThreshold = 0.05,
            title = "Differential Gene Expression"
        )
        visualizations.append(volcanoPlot)

        // 6. MA plot
        maPlot ← GenerateMAPlot(
            expressionData = reportData.expression.allGenes,
            title = "MA Plot (Expression vs Fold Change)"
        )
        visualizations.append(maPlot)

        // 7. Heatmap of top DEGs
        degHeatmap ← GenerateHeatmap(
            genes = reportData.expression.topDEGs,
            samples = reportData.expression.samples,
            title = "Top 50 Differentially Expressed Genes",
            clustering = "hierarchical",
            colorScheme = "RdBu_r"
        )
        visualizations.append(degHeatmap)

        // 8. Pathway enrichment dot plot
        pathwayDotPlot ← GeneratePathwayDotPlot(
            pathways = reportData.expression.enrichedPathways,
            topN = 20,
            title = "Immune Pathway Enrichment"
        )
        visualizations.append(pathwayDotPlot)

        // 9. PCA plot
        pcaPlot ← GeneratePCAPlot(
            expressionData = reportData.expression.normalizedCounts,
            metadata = reportData.expression.sampleMetadata,
            title = "Principal Component Analysis"
        )
        visualizations.append(pcaPlot)
    END IF

    // === Genome Browser Tracks ===

    // 10. Generate IGV-compatible tracks
    genomeBrowserTracks ← GenerateGenomeBrowserTracks(
        alignmentData = reportData.alignment,
        offTargetData = reportData.offTarget,
        config = config
    )
    visualizations.append(genomeBrowserTracks)

    LogInfo("Generated {length(visualizations)} visualizations")

    RETURN visualizations
END

SUBROUTINE: GenerateVolcanoPlot
INPUT: expressionData, log2FCThreshold, padjThreshold, title
OUTPUT: plot (Plot object)

BEGIN
    plot ← {
        type: "scatter",
        title: title,
        xLabel: "log2 Fold Change",
        yLabel: "-log10(adjusted p-value)",
        points: [],
        annotations: []
    }

    // Process each gene
    FOR EACH gene IN expressionData DO
        x ← gene.log2FoldChange
        y ← -LOG10(gene.padj)

        // Determine point color and size based on significance
        IF ABS(x) >= log2FCThreshold AND gene.padj < padjThreshold THEN
            IF x > 0 THEN
                color ← "red"
                label ← "Upregulated"
            ELSE
                color ← "blue"
                label ← "Downregulated"
            END IF
            size ← 8
        ELSE
            color ← "gray"
            label ← "Not significant"
            size ← 4
        END IF

        plot.points.append({
            x: x,
            y: y,
            color: color,
            size: size,
            label: label,
            geneSymbol: gene.geneSymbol,
            tooltip: FormatGeneTooltip(gene)
        })

        // Annotate top genes
        IF gene.rank <= 10 THEN
            plot.annotations.append({
                x: x,
                y: y,
                text: gene.geneSymbol,
                fontSize: 10
            })
        END IF
    END FOR

    // Add threshold lines
    plot.vlines ← [
        {x: -log2FCThreshold, style: "dashed", color: "black"},
        {x: log2FCThreshold, style: "dashed", color: "black"}
    ]
    plot.hlines ← [
        {y: -LOG10(padjThreshold), style: "dashed", color: "black"}
    ]

    RETURN plot
END

SUBROUTINE: GenerateHeatmap
INPUT: genes, samples, title, clustering, colorScheme
OUTPUT: heatmap (Plot object)

BEGIN
    // Extract expression matrix
    expressionMatrix ← []
    geneLabels ← []

    FOR EACH gene IN genes DO
        row ← [gene.normalizedCounts[sample] for sample in samples]
        expressionMatrix.append(row)
        geneLabels.append(gene.geneSymbol)
    END FOR

    // Z-score normalization for better visualization
    normalizedMatrix ← []
    FOR EACH row IN expressionMatrix DO
        mean ← MEAN(row)
        std ← STDEV(row)
        normalizedRow ← [(value - mean) / std for value in row]
        normalizedMatrix.append(normalizedRow)
    END FOR

    // Perform hierarchical clustering
    IF clustering == "hierarchical" THEN
        geneTree ← HierarchicalClustering(normalizedMatrix, metric = "euclidean")
        sampleTree ← HierarchicalClustering(TRANSPOSE(normalizedMatrix), metric = "correlation")

        // Reorder based on clustering
        geneOrder ← GetDendrogram Order(geneTree)
        sampleOrder ← GetDendrogramOrder(sampleTree)

        normalizedMatrix ← ReorderMatrix(normalizedMatrix, geneOrder, sampleOrder)
        geneLabels ← [geneLabels[i] for i in geneOrder]
        samples ← [samples[i] for i in sampleOrder]
    END IF

    heatmap ← {
        type: "heatmap",
        title: title,
        data: normalizedMatrix,
        rowLabels: geneLabels,
        colLabels: samples,
        colorScheme: colorScheme,
        colorBarLabel: "Z-score",
        dendrograms: {
            row: geneTree IF clustering != null,
            col: sampleTree IF clustering != null
        }
    }

    RETURN heatmap
END

SUBROUTINE: GeneratePathwayDotPlot
INPUT: pathways, topN, title
OUTPUT: plot (Plot object)

BEGIN
    // Sort pathways by adjusted p-value
    sortedPathways ← Sort(pathways, by = padjusted, ascending = true)
    topPathways ← sortedPathways[0 : MIN(topN, length(sortedPathways))]

    plot ← {
        type: "dotplot",
        title: title,
        xLabel: "Gene Ratio",
        yLabel: "Pathway",
        points: []
    }

    FOR EACH pathway IN topPathways DO
        geneRatio ← pathway.overlapSize / pathway.pathwaySize

        plot.points.append({
            y: pathway.name,
            x: geneRatio,
            size: pathway.overlapSize × 2,  // Scale dot size by overlap
            color: -LOG10(pathway.padjusted),  // Color by significance
            tooltip: FormatPathwayTooltip(pathway)
        })
    END FOR

    plot.colorScale ← {
        label: "-log10(FDR)",
        scheme: "viridis",
        min: 0,
        max: MAX(-LOG10(p.padjusted) for p in topPathways)
    }

    plot.sizeScale ← {
        label: "Gene Count",
        min: 5,
        max: 20
    }

    RETURN plot
END

SUBROUTINE: GeneratePCAPlot
INPUT: expressionData, metadata, title
OUTPUT: plot (Plot object)

BEGIN
    // Perform PCA on normalized expression data
    // Transpose so samples are rows
    dataMatrix ← TRANSPOSE(expressionData)

    // Center and scale data
    centeredData ← CenterAndScale(dataMatrix)

    // Compute principal components
    pcaResult ← ComputePCA(centeredData, numComponents = 3)

    plot ← {
        type: "scatter",
        title: title,
        xLabel: "PC1 ({pcaResult.explainedVariance[0]}%)",
        yLabel: "PC2 ({pcaResult.explainedVariance[1]}%)",
        points: []
    }

    // Plot samples
    FOR i ← 0 TO length(metadata.samples) - 1 DO
        sample ← metadata.samples[i]

        plot.points.append({
            x: pcaResult.components[i][0],
            y: pcaResult.components[i][1],
            color: metadata.groups[i],
            shape: metadata.replicates[i],
            size: 10,
            label: sample,
            tooltip: FormatSampleTooltip(sample, metadata)
        })
    END FOR

    plot.legend ← {
        colorBy: "Treatment Group",
        shapeBy: "Replicate"
    }

    RETURN plot
END
```

**Complexity Analysis**:
- **Time**: O(G × S + G log G) for heatmap with clustering
- **Space**: O(G × S) for expression matrix
- **Typical Performance**: 5-30 seconds depending on plot complexity

---

## Algorithm 3: HTML Report Generation

**Purpose**: Create interactive HTML report with embedded visualizations.

```
ALGORITHM: GenerateHTMLReport
INPUT: reportData, visualizations, config
OUTPUT: htmlReport (string)

BEGIN
    html ← []

    // === Header Section ===
    html.append("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>CRISPR-Cas13 Analysis Report</title>
        <style>
            {CSS_STYLES}
        </style>
        <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
        <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
    </head>
    <body>
    """)

    // === Title and Summary ===
    html.append("""
    <div class="header">
        <h1>CRISPR-Cas13 Bioinformatics Analysis Report</h1>
        <p class="metadata">
            <strong>Project:</strong> {reportData.metadata.projectName}<br>
            <strong>Analysis Date:</strong> {reportData.metadata.analysisDate}<br>
            <strong>Pipeline Version:</strong> {reportData.metadata.pipelineVersion}<br>
            <strong>Reference Genome:</strong> {reportData.metadata.referenceGenome}
        </p>
    </div>
    """)

    // === Executive Summary ===
    html.append(GenerateExecutiveSummary(reportData))

    // === Table of Contents ===
    html.append("""
    <nav class="toc">
        <h2>Table of Contents</h2>
        <ul>
            <li><a href="#alignment">1. Read Alignment Analysis</a></li>
            <li><a href="#offtarget">2. Off-Target Prediction</a></li>
            <li><a href="#expression">3. Differential Expression Analysis</a></li>
            <li><a href="#pathways">4. Immune Pathway Enrichment</a></li>
            <li><a href="#quality">5. Quality Metrics</a></li>
            <li><a href="#methods">6. Methods</a></li>
        </ul>
    </nav>
    """)

    // === Section 1: Alignment ===
    html.append(GenerateAlignmentSection(reportData.alignment, visualizations))

    // === Section 2: Off-Target ===
    IF reportData.offTarget != null THEN
        html.append(GenerateOffTargetSection(reportData.offTarget, visualizations))
    END IF

    // === Section 3: Expression ===
    IF reportData.expression != null THEN
        html.append(GenerateExpressionSection(reportData.expression, visualizations))
    END IF

    // === Section 4: Pathways ===
    IF reportData.expression != null THEN
        html.append(GeneratePathwaySection(reportData.expression.enrichedPathways, visualizations))
    END IF

    // === Section 5: Quality Metrics ===
    html.append(GenerateQualitySection(reportData.qualityMetrics, visualizations))

    // === Section 6: Methods ===
    html.append(GenerateMethodsSection(config))

    // === Footer ===
    html.append("""
    <footer>
        <p>Generated by CRISPR-Cas13 Bioinformatics Pipeline v{config.version}</p>
        <p>Processing Time: {reportData.processingStats.totalTime} seconds</p>
    </footer>
    </body>
    </html>
    """)

    htmlReport ← JOIN(html, "\n")

    RETURN htmlReport
END

SUBROUTINE: GenerateExecutiveSummary
INPUT: reportData
OUTPUT: htmlSection (string)

BEGIN
    section ← []

    section.append("""
    <section class="executive-summary">
        <h2>Executive Summary</h2>
        <div class="summary-grid">
    """)

    // Alignment metrics card
    section.append("""
    <div class="metric-card">
        <h3>Read Alignment</h3>
        <div class="metric-value">{reportData.alignment.alignmentRate}%</div>
        <div class="metric-label">Alignment Rate</div>
        <p>{FormatNumber(reportData.alignment.alignedReads)} / {FormatNumber(reportData.alignment.totalReads)} reads aligned</p>
    </div>
    """)

    // Off-target metrics card
    IF reportData.offTarget != null THEN
        section.append("""
        <div class="metric-card {GetRiskClass(reportData.offTarget)}">
            <h3>Off-Target Prediction</h3>
            <div class="metric-value">{reportData.offTarget.highConfidenceOffTargets}</div>
            <div class="metric-label">High-Confidence Off-Targets</div>
            <p>{reportData.offTarget.totalGRNAs} gRNAs analyzed</p>
        </div>
        """)
    END IF

    // Expression metrics card
    IF reportData.expression != null THEN
        section.append("""
        <div class="metric-card">
            <h3>Differential Expression</h3>
            <div class="metric-value">{reportData.expression.significantDEGs}</div>
            <div class="metric-label">Significant DEGs</div>
            <p>{reportData.expression.upregulatedGenes} up, {reportData.expression.downregulatedGenes} down</p>
        </div>
        """)
    END IF

    // Immune score card
    IF reportData.expression != null THEN
        immuneScoreClass ← GetImmuneScoreClass(reportData.expression.immuneScore)
        section.append("""
        <div class="metric-card {immuneScoreClass}">
            <h3>Immune Activation</h3>
            <div class="metric-value">{reportData.expression.immuneScore}/100</div>
            <div class="metric-label">Immune Score</div>
            <p>{GetImmuneScoreInterpretation(reportData.expression.immuneScore)}</p>
        </div>
        """)
    END IF

    section.append("""
        </div>
    </section>
    """)

    RETURN JOIN(section, "\n")
END

SUBROUTINE: GenerateAlignmentSection
INPUT: alignmentData, visualizations
OUTPUT: htmlSection (string)

BEGIN
    section ← []

    section.append("""
    <section id="alignment">
        <h2>1. Read Alignment Analysis</h2>

        <div class="subsection">
            <h3>1.1 Alignment Statistics</h3>
            <table class="data-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Reads</td>
                    <td>{FormatNumber(alignmentData.totalReads)}</td>
                </tr>
                <tr>
                    <td>Aligned Reads</td>
                    <td>{FormatNumber(alignmentData.alignedReads)} ({alignmentData.alignmentRate}%)</td>
                </tr>
                <tr>
                    <td>Unmapped Reads</td>
                    <td>{FormatNumber(alignmentData.unmappedReads)}</td>
                </tr>
                <tr>
                    <td>Multi-mapped Reads</td>
                    <td>{FormatNumber(alignmentData.multiMappedReads)}</td>
                </tr>
                <tr>
                    <td>Average MAPQ</td>
                    <td>{alignmentData.averageMAPQ}</td>
                </tr>
            </table>
        </div>

        <div class="subsection">
            <h3>1.2 Coverage Statistics</h3>
            <table class="data-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Mean Coverage</td>
                    <td>{alignmentData.coverageStatistics.meanCoverage}×</td>
                </tr>
                <tr>
                    <td>Median Coverage</td>
                    <td>{alignmentData.coverageStatistics.medianCoverage}×</td>
                </tr>
                <tr>
                    <td>Coverage ≥10×</td>
                    <td>{alignmentData.coverageStatistics.coverage10x}%</td>
                </tr>
                <tr>
                    <td>Coverage ≥30×</td>
                    <td>{alignmentData.coverageStatistics.coverage30x}%</td>
                </tr>
                <tr>
                    <td>Uniformity</td>
                    <td>{alignmentData.coverageStatistics.uniformity}</td>
                </tr>
            </table>
        </div>

        <div class="subsection">
            <h3>1.3 Visualizations</h3>
            {EmbedVisualization(visualizations, "alignment_pie")}
            {EmbedVisualization(visualizations, "coverage_histogram")}
        </div>
    </section>
    """)

    RETURN JOIN(section, "\n")
END

SUBROUTINE: EmbedVisualization
INPUT: visualizations, plotId
OUTPUT: htmlDiv (string)

BEGIN
    plot ← FindPlotById(visualizations, plotId)

    IF plot == null THEN
        RETURN "<p>Visualization not available</p>"
    END IF

    // Convert plot to Plotly JSON
    plotlyJson ← ConvertToPlotlyFormat(plot)

    htmlDiv ← """
    <div class="plot-container" id="{plotId}">
        <script>
            Plotly.newPlot('{plotId}', {plotlyJson.data}, {plotlyJson.layout}, {plotlyJson.config});
        </script>
    </div>
    """

    RETURN htmlDiv
END
```

**Complexity Analysis**:
- **Time**: O(S × P) where S = sections, P = plots
- **Space**: O(H) where H = HTML size (typically 1-10 MB)
- **Generation Time**: 1-5 seconds

---

## Algorithm 4: PDF Export

**Purpose**: Generate print-friendly PDF report.

```
ALGORITHM: GeneratePDFReport
INPUT: htmlReport (string), outputPath (string)
OUTPUT: pdfFile (path)

BEGIN
    // Use headless browser for HTML→PDF conversion
    // Supports WeasyPrint, Puppeteer, or similar

    TRY
        // Option 1: WeasyPrint (Python)
        IF HasWeasyPrint() THEN
            pdf ← WeasyPrint.HTML(htmlReport).write_pdf(outputPath)

        // Option 2: Puppeteer (Node.js)
        ELSE IF HasPuppeteer() THEN
            browser ← LaunchBrowser(headless = true)
            page ← browser.newPage()
            page.setContent(htmlReport)
            page.pdf({
                path: outputPath,
                format: 'A4',
                printBackground: true,
                margin: {top: '1cm', right: '1cm', bottom: '1cm', left: '1cm'}
            })
            browser.close()

        ELSE
            RETURN error("No PDF renderer available")
        END IF

        LogInfo("PDF report generated: {outputPath}")
        RETURN outputPath

    CATCH RenderError AS e DO
        LogError("PDF generation failed: {e.message}")
        RETURN null
    END TRY
END
```

**Complexity Analysis**:
- **Time**: O(P) where P = page count (typically 5-30 seconds)
- **Space**: O(H) for HTML rendering
- **File Size**: 2-20 MB depending on plot complexity

---

## Algorithm 5: Data Export (VCF, BED, CSV)

**Purpose**: Export analysis results in standard bioinformatics formats.

```
ALGORITHM: ExportAnalysisResults
INPUT: reportData, outputDirectory, config
OUTPUT: exportedFiles (array of paths)

BEGIN
    exportedFiles ← []

    // === Export 1: VCF (Variant Call Format) for off-targets ===
    IF reportData.offTarget != null THEN
        vcfPath ← outputDirectory + "/offtargets.vcf"
        vcfFile ← OpenFile(vcfPath, mode = "write")

        // Write VCF header
        vcfFile.WriteLine("##fileformat=VCFv4.2")
        vcfFile.WriteLine("##fileDate={FormatDate(GetCurrentTime())}")
        vcfFile.WriteLine("##source=CRISPR-Cas13-Pipeline-v{config.version}")
        vcfFile.WriteLine("##reference={config.referenceGenome}")
        vcfFile.WriteLine("##INFO=<ID=GRNA,Number=1,Type=String,Description=\"Guide RNA sequence\">")
        vcfFile.WriteLine("##INFO=<ID=OTS,Number=1,Type=Float,Description=\"Off-target score\">")
        vcfFile.WriteLine("##INFO=<ID=MM,Number=1,Type=Integer,Description=\"Number of mismatches\">")
        vcfFile.WriteLine("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO")

        // Write off-target sites
        FOR EACH gRNA IN reportData.offTarget.gRNAs DO
            FOR EACH site IN gRNA.offTargetSites DO
                chrom ← site.chromosome
                pos ← site.position
                id ← site.id
                ref ← site.referenceAllele
                alt ← site.alternateAllele
                qual ← ROUND(site.offTargetScore × 100)
                filter ← IF site.offTargetScore >= 0.7 THEN "PASS" ELSE "LowScore"
                info ← "GRNA={gRNA.sequence};OTS={site.offTargetScore};MM={site.mismatches}"

                vcfFile.WriteLine("{chrom}\t{pos}\t{id}\t{ref}\t{alt}\t{qual}\t{filter}\t{info}")
            END FOR
        END FOR

        vcfFile.Close()
        exportedFiles.append(vcfPath)
        LogInfo("Exported VCF: {vcfPath}")
    END IF

    // === Export 2: BED (Browser Extensible Data) for genome browser ===
    bedPath ← outputDirectory + "/coverage.bed"
    bedFile ← OpenFile(bedPath, mode = "write")

    bedFile.WriteLine("track name=\"CRISPR_Coverage\" description=\"RNA-seq coverage\" visibility=2 color=0,0,255")

    FOR EACH region IN reportData.alignment.coverageRegions DO
        bedFile.WriteLine("{region.chrom}\t{region.start}\t{region.end}\t{region.name}\t{region.score}\t{region.strand}")
    END FOR

    bedFile.Close()
    exportedFiles.append(bedPath)
    LogInfo("Exported BED: {bedPath}")

    // === Export 3: CSV for differential expression ===
    IF reportData.expression != null THEN
        csvPath ← outputDirectory + "/differential_expression.csv"
        csvFile ← OpenFile(csvPath, mode = "write")

        // Write header
        csvFile.WriteLine("GeneID,GeneSymbol,BaseMean,Log2FoldChange,PValue,AdjustedPValue,Biotype")

        // Write gene results
        FOR EACH gene IN reportData.expression.allGenes DO
            csvFile.WriteLine(
                "{gene.geneId}," +
                "{gene.geneSymbol}," +
                "{gene.baseMean}," +
                "{gene.log2FoldChange}," +
                "{gene.pvalue}," +
                "{gene.padj}," +
                "{gene.biotype}"
            )
        END FOR

        csvFile.Close()
        exportedFiles.append(csvPath)
        LogInfo("Exported CSV: {csvPath}")
    END IF

    // === Export 4: JSON for programmatic access ===
    jsonPath ← outputDirectory + "/report_data.json"
    WriteJSON(reportData, jsonPath, indent = 2)
    exportedFiles.append(jsonPath)
    LogInfo("Exported JSON: {jsonPath}")

    RETURN exportedFiles
END
```

**Complexity Analysis**:
- **Time**: O(N) where N = total data points
- **Space**: O(N) for file buffers
- **Export Time**: 1-10 seconds depending on data size

---

## Algorithm 6: Main Reporting Pipeline

**Purpose**: Orchestrate complete reporting workflow.

```
ALGORITHM: GenerateComprehensiveReport
INPUT: alignmentResults, offTargetResults, expressionResults, config
OUTPUT: reportBundle (ReportBundle object)

BEGIN
    startTime ← GetCurrentTime()

    LogInfo("Starting comprehensive report generation...")

    // Step 1: Aggregate data
    LogInfo("Aggregating report data...")
    reportData ← AggregateReportData(
        alignmentResults,
        offTargetResults,
        expressionResults,
        config
    )

    // Step 2: Generate visualizations
    LogInfo("Generating visualizations...")
    visualizations ← GenerateVisualizations(reportData, config.visualization)

    // Step 3: Generate HTML report
    LogInfo("Generating HTML report...")
    htmlReport ← GenerateHTMLReport(reportData, visualizations, config)

    htmlPath ← config.outputDirectory + "/report.html"
    WriteFile(htmlPath, htmlReport)

    // Step 4: Generate PDF (optional)
    pdfPath ← null
    IF config.generatePDF THEN
        LogInfo("Generating PDF report...")
        pdfPath ← GeneratePDFReport(htmlReport, config.outputDirectory + "/report.pdf")
    END IF

    // Step 5: Export data files
    LogInfo("Exporting analysis results...")
    exportedFiles ← ExportAnalysisResults(reportData, config.outputDirectory, config)

    // Step 6: Generate summary statistics file
    summaryPath ← config.outputDirectory + "/summary.txt"
    WriteSummaryFile(reportData, summaryPath)
    exportedFiles.append(summaryPath)

    processingTime ← GetCurrentTime() - startTime

    reportBundle ← {
        htmlReport: htmlPath,
        pdfReport: pdfPath,
        exportedFiles: exportedFiles,
        visualizations: visualizations,
        reportData: reportData,
        generationTime: processingTime,
        timestamp: GetCurrentTime()
    }

    LogInfo("Report generation complete in {processingTime}s")
    LogInfo("HTML report: {htmlPath}")
    IF pdfPath != null THEN
        LogInfo("PDF report: {pdfPath}")
    END IF

    RETURN reportBundle
END

SUBROUTINE: WriteSummaryFile
INPUT: reportData, outputPath
OUTPUT: none

BEGIN
    file ← OpenFile(outputPath, mode = "write")

    file.WriteLine("=" × 80)
    file.WriteLine("CRISPR-Cas13 Analysis Summary")
    file.WriteLine("=" × 80)
    file.WriteLine("")

    file.WriteLine("Project: {reportData.metadata.projectName}")
    file.WriteLine("Analysis Date: {reportData.metadata.analysisDate}")
    file.WriteLine("Pipeline Version: {reportData.metadata.pipelineVersion}")
    file.WriteLine("")

    file.WriteLine("-" × 80)
    file.WriteLine("READ ALIGNMENT")
    file.WriteLine("-" × 80)
    file.WriteLine("Total Reads: {FormatNumber(reportData.alignment.totalReads)}")
    file.WriteLine("Aligned Reads: {FormatNumber(reportData.alignment.alignedReads)} ({reportData.alignment.alignmentRate}%)")
    file.WriteLine("Mean Coverage: {reportData.alignment.coverageStatistics.meanCoverage}×")
    file.WriteLine("")

    IF reportData.offTarget != null THEN
        file.WriteLine("-" × 80)
        file.WriteLine("OFF-TARGET PREDICTION")
        file.WriteLine("-" × 80)
        file.WriteLine("gRNAs Analyzed: {reportData.offTarget.totalGRNAs}")
        file.WriteLine("High-Confidence Off-Targets: {reportData.offTarget.highConfidenceOffTargets}")
        file.WriteLine("")
    END IF

    IF reportData.expression != null THEN
        file.WriteLine("-" × 80)
        file.WriteLine("DIFFERENTIAL EXPRESSION")
        file.WriteLine("-" × 80)
        file.WriteLine("Genes Tested: {FormatNumber(reportData.expression.totalGenesTested)}")
        file.WriteLine("Significant DEGs: {reportData.expression.significantDEGs}")
        file.WriteLine("Upregulated: {reportData.expression.upregulatedGenes}")
        file.WriteLine("Downregulated: {reportData.expression.downregulatedGenes}")
        file.WriteLine("Immune Score: {reportData.expression.immuneScore}/100")
        file.WriteLine("")
    END IF

    file.WriteLine("=" × 80)

    file.Close()
END
```

**Complexity Analysis**:
- **Time**: O(G × S + P) where P = plot rendering time
- **Space**: O(G × S + H) where H = HTML size
- **End-to-End Performance**: 30 seconds to 5 minutes depending on dataset size

---

## Error Handling

### 1. Missing Data Handling

```
FUNCTION: HandleMissingData
INPUT: reportData, requiredSections
OUTPUT: modifiedReportData

BEGIN
    FOR EACH section IN requiredSections DO
        IF reportData[section] == null THEN
            LogWarning("Missing data for section: {section}")
            reportData[section] ← CreatePlaceholderSection(section)
        END IF
    END FOR

    RETURN reportData
END
```

### 2. Visualization Failure Recovery

```
FUNCTION: GenerateVisualizationSafely
INPUT: plotFunction, data, config
OUTPUT: plot or placeholder

BEGIN
    TRY
        plot ← plotFunction(data, config)
        RETURN plot
    CATCH VisualizationError AS e DO
        LogError("Failed to generate plot: {e.message}")
        RETURN CreatePlaceholderPlot(e.message)
    END TRY
END
```

---

## Performance Optimization

### 1. Parallel Visualization Generation

```
FUNCTION: GenerateVisualizationsParallel
INPUT: reportData, config, numThreads
OUTPUT: visualizations

BEGIN
    plotTasks ← [
        TASK(GenerateVolcanoPlot, reportData.expression),
        TASK(GenerateHeatmap, reportData.expression),
        TASK(GeneratePathwayDotPlot, reportData.expression),
        // ... more plot tasks
    ]

    visualizations ← ParallelExecute(plotTasks, numThreads = numThreads)

    RETURN visualizations
END
```

---

## References

1. Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*.
2. Plotly Technologies Inc. (2015). Collaborative data science. https://plot.ly
3. IGV: Integrative Genomics Viewer. https://software.broadinstitute.org/software/igv/

---

**End of Pseudocode Documentation**
