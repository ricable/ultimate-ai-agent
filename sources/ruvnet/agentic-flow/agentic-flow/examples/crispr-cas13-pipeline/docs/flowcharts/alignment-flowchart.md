# Read Alignment Module - Flowchart

## Main Alignment Workflow

```mermaid
flowchart TD
    Start([Start: FASTQ Input]) --> LoadIndex{BWT Index<br/>Exists?}

    LoadIndex -->|Yes| LoadIdx[Load BWT Index]
    LoadIndex -->|No| BuildIdx[Build BWT Index<br/>from Reference]

    BuildIdx --> SaveIdx[Save Index to Disk]
    SaveIdx --> LoadIdx
    LoadIdx --> InitStats[Initialize<br/>Statistics]

    InitStats --> ReadBatch[Read Batch<br/>10,000 reads]

    ReadBatch --> ParallelAlign{Parallel<br/>Processing}

    ParallelAlign --> Thread1[Thread 1:<br/>Align Reads 1-2500]
    ParallelAlign --> Thread2[Thread 2:<br/>Align Reads 2501-5000]
    ParallelAlign --> Thread3[Thread 3:<br/>Align Reads 5001-7500]
    ParallelAlign --> Thread4[Thread 4:<br/>Align Reads 7501-10000]

    Thread1 --> Align1[Seed-and-Extend<br/>Alignment]
    Thread2 --> Align2[Seed-and-Extend<br/>Alignment]
    Thread3 --> Align3[Seed-and-Extend<br/>Alignment]
    Thread4 --> Align4[Seed-and-Extend<br/>Alignment]

    Align1 --> Merge[Merge Results]
    Align2 --> Merge
    Align3 --> Merge
    Align4 --> Merge

    Merge --> WriteBAM[Write Alignments<br/>to BAM]
    WriteBAM --> UpdateStats[Update Statistics]
    UpdateStats --> MoreReads{More Reads?}

    MoreReads -->|Yes| ReadBatch
    MoreReads -->|No| Finalize[Finalize BAM<br/>& Generate Report]

    Finalize --> End([End: BAM Output])

    style Start fill:#e1f5e1
    style End fill:#ffe1e1
    style ParallelAlign fill:#fff4e1
    style BuildIdx fill:#e1f0ff
```

## Seed-and-Extend Algorithm Detail

```mermaid
flowchart TD
    StartAlign([Start: Single Read]) --> ExtractSeeds[Extract Seeds<br/>20bp windows]

    ExtractSeeds --> BWTSearch[BWT Exact Match<br/>for Each Seed]

    BWTSearch --> CheckHits{Seeds Found?}

    CheckHits -->|No| Unmapped[Mark as<br/>Unmapped]
    CheckHits -->|Yes| ClusterSeeds[Cluster Seeds<br/>by Genomic Location]

    ClusterSeeds --> CheckCoverage{Seed Coverage<br/>>= 70%?}

    CheckCoverage -->|No| Unmapped
    CheckCoverage -->|Yes| ExtractRef[Extract Reference<br/>Regions + Flanking]

    ExtractRef --> LoopClusters{For Each<br/>Cluster}

    LoopClusters --> SmithWaterman[Smith-Waterman<br/>Local Alignment]

    SmithWaterman --> CalcScore[Calculate<br/>Alignment Score]

    CalcScore --> UpdateBest{Score > Best?}

    UpdateBest -->|Yes| SetBest[Update Best<br/>Alignment]
    UpdateBest -->|No| NextCluster

    SetBest --> NextCluster[Next Cluster]
    NextCluster --> LoopClusters

    LoopClusters -->|Done| CheckBest{Best Alignment<br/>Found?}

    CheckBest -->|No| Unmapped
    CheckBest -->|Yes| CalcMAPQ[Calculate MAPQ<br/>Score]

    CalcMAPQ --> FormatCIGAR[Format CIGAR<br/>String]

    FormatCIGAR --> EndAlign([Return: Alignment])

    Unmapped --> EndAlign

    style StartAlign fill:#e1f5e1
    style EndAlign fill:#ffe1e1
    style SmithWaterman fill:#e1f0ff
    style CheckHits fill:#fff4e1
    style CheckCoverage fill:#fff4e1
    style CheckBest fill:#fff4e1
```

## BWT Index Construction

```mermaid
flowchart TD
    StartBWT([Start: Reference<br/>Genome]) --> BuildSA[Construct Suffix Array<br/>DC3 Algorithm]

    BuildSA --> CreateBWT[Build BWT String<br/>from Suffix Array]

    CreateBWT --> BuildOcc[Build Occurrence<br/>Table O(m × Σ)]

    BuildOcc --> BuildCount[Build Count Table<br/>Cumulative Sums]

    BuildCount --> SampleSA[Sample Suffix Array<br/>Every 32 Positions]

    SampleSA --> Package[Package Index<br/>Structure]

    Package --> EndBWT([Return: BWT Index])

    style StartBWT fill:#e1f5e1
    style EndBWT fill:#ffe1e1
    style BuildSA fill:#e1f0ff
```

## Smith-Waterman Dynamic Programming

```mermaid
flowchart TD
    StartSW([Start: Query & Ref]) --> InitMatrices[Initialize DP Matrices<br/>H, E, F]

    InitMatrices --> LoopI{For i=1 to n<br/>query length}

    LoopI --> LoopJ{For j=1 to m<br/>ref length}

    LoopJ --> CalcE[Calculate E[i][j]<br/>Gap in Query]

    CalcE --> CalcF[Calculate F[i][j]<br/>Gap in Reference]

    CalcF --> CalcMatch{Query[i]<br/>== Ref[j]?}

    CalcMatch -->|Yes| AddMatch[matchScore =<br/>H[i-1][j-1] + 2]
    CalcMatch -->|No| AddMismatch[matchScore =<br/>H[i-1][j-1] - 3]

    AddMatch --> TakeMax
    AddMismatch --> TakeMax[H[i][j] = MAX(0,<br/>matchScore, E, F)]

    TakeMax --> CheckMax{H[i][j] ><br/>maxScore?}

    CheckMax -->|Yes| UpdateMax[Update maxScore<br/>& Position]
    CheckMax -->|No| NextJ

    UpdateMax --> NextJ[j++]
    NextJ --> LoopJ

    LoopJ -->|Done| NextI[i++]
    NextI --> LoopI

    LoopI -->|Done| Traceback[Traceback from<br/>maxScore Position]

    Traceback --> CompressCIGAR[Compress CIGAR<br/>String]

    CompressCIGAR --> EndSW([Return: Alignment<br/>with Score])

    style StartSW fill:#e1f5e1
    style EndSW fill:#ffe1e1
    style TakeMax fill:#e1f0ff
    style CalcMatch fill:#fff4e1
    style CheckMax fill:#fff4e1
```

## Error Handling Flow

```mermaid
flowchart TD
    Operation[Execute<br/>Alignment Operation] --> CheckError{Error<br/>Occurred?}

    CheckError -->|No| Success([Success])

    CheckError -->|Yes| ClassifyError{Error Type?}

    ClassifyError -->|Invalid Input| LogError1[Log Error<br/>Reject Read]
    ClassifyError -->|OOM| SwitchMode[Switch to<br/>Banded SW]
    ClassifyError -->|Timeout| LogWarning[Log Warning<br/>Mark Unmapped]
    ClassifyError -->|Transient| Retry{Retry Count<br/>< Max?}

    Retry -->|Yes| Backoff[Exponential<br/>Backoff]
    Retry -->|No| LogError2[Log Error<br/>Mark Failed]

    Backoff --> Operation

    LogError1 --> Fail([Fail Gracefully])
    LogError2 --> Fail
    LogWarning --> Fail

    SwitchMode --> RetryOp[Retry with<br/>Lower Memory]
    RetryOp --> Success

    style Success fill:#e1f5e1
    style Fail fill:#ffe1e1
    style ClassifyError fill:#fff4e1
    style Retry fill:#fff4e1
```

---

## Complexity Summary

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| BWT Index Construction | O(m) | O(m) |
| Backward Search | O(n) | O(1) |
| Seed-and-Extend | O(k × n × m) | O(m) |
| Smith-Waterman | O(n × m) | O(n × m) |
| Full Pipeline | O(N × n × m / p) | O(B × n) |

Where:
- m = reference genome size
- n = read length
- N = total reads
- k = seed clusters
- p = parallelism
- B = batch size
