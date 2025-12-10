# Off-Target Prediction Module - Flowchart

## Main Off-Target Prediction Workflow

```mermaid
flowchart TD
    Start([Start: gRNA Input]) --> ValidateRNA[Validate gRNA<br/>Length & Bases]

    ValidateRNA --> LoadResources[Load Transcriptome<br/>& ML Model]

    LoadResources --> BuildIndex{K-mer Index<br/>Built?}

    BuildIndex -->|No| CreateIndex[Build K-mer Index<br/>from Transcriptome]
    BuildIndex -->|Yes| ExtractSeed

    CreateIndex --> ExtractSeed[Extract Seed Region<br/>Last 12 bases]

    ExtractSeed --> FindSeeds[Find Seed Matches<br/>in K-mer Index]

    FindSeeds --> CheckMatches{Seed Matches<br/>Found?}

    CheckMatches -->|No| NoTargets[Return Empty<br/>Result Set]
    CheckMatches -->|Yes| VerifyFull[Verify Full gRNA<br/>Alignment]

    VerifyFull --> FilterMM[Filter by<br/>Max Mismatches]

    FilterMM --> CheckCandidates{Candidates > 0?}

    CheckCandidates -->|No| NoTargets
    CheckCandidates -->|Yes| ParallelFeatures{Parallel Feature<br/>Extraction}

    ParallelFeatures --> Batch1[Batch 1:<br/>Extract Features]
    ParallelFeatures --> Batch2[Batch 2:<br/>Extract Features]
    ParallelFeatures --> Batch3[Batch 3:<br/>Extract Features]

    Batch1 --> MergeFeatures[Merge Feature<br/>Vectors]
    Batch2 --> MergeFeatures
    Batch3 --> MergeFeatures

    MergeFeatures --> Predict[Gradient Boosting<br/>Prediction]

    Predict --> ScoreSites[Calculate Off-Target<br/>Scores & Probabilities]

    ScoreSites --> FilterThreshold[Filter by Score<br/>& Expression]

    FilterThreshold --> RankSites[Rank by Composite<br/>Score]

    RankSites --> Annotate[Add Gene Annotations<br/>& Pathways]

    Annotate --> GenStats[Generate Summary<br/>Statistics]

    GenStats --> End([End: Ranked<br/>Off-Targets])

    NoTargets --> End

    style Start fill:#e1f5e1
    style End fill:#ffe1e1
    style ParallelFeatures fill:#fff4e1
    style CheckMatches fill:#fff4e1
    style CheckCandidates fill:#fff4e1
    style Predict fill:#e1f0ff
```

## Feature Extraction Detail

```mermaid
flowchart TD
    StartFeat([Start: Candidate Site]) --> SeqFeatures[Sequence Features]

    SeqFeatures --> CalcMM[Calculate Mismatch<br/>Distribution]
    CalcMM --> CalcGC[Calculate GC<br/>Content]
    CalcGC --> CalcHomopoly[Calculate Homopolymer<br/>Runs]

    CalcHomopoly --> StructFeatures[Structure Features]

    StructFeatures --> PredictStruct[Predict RNA<br/>Secondary Structure]
    PredictStruct --> CalcEnergy[Calculate Free<br/>Energy]
    CalcEnergy --> CalcAccess[Calculate<br/>Accessibility]

    CalcAccess --> ExprFeatures[Expression Features]

    ExprFeatures --> GetTPM[Get Transcript<br/>TPM Value]
    GetTPM --> GetCoverage[Get Local Read<br/>Coverage]

    GetCoverage --> ChromFeatures{Chromatin Data<br/>Available?}

    ChromFeatures -->|Yes| GetChromatin[Get Chromatin<br/>Accessibility]
    ChromFeatures -->|No| DefaultChrom[Use Default<br/>Value]

    GetChromatin --> EvoFeatures
    DefaultChrom --> EvoFeatures[Evolutionary Features]

    EvoFeatures --> GetConservation[Get PhyloP<br/>Conservation Score]

    GetConservation --> DistFeatures[Distance Features]

    DistFeatures --> CalcJunction[Calculate Distance to<br/>Exon Junction]

    CalcJunction --> ThermoFeatures[Thermodynamic Features]

    ThermoFeatures --> CalcAffinity[Calculate Binding<br/>Affinity]

    CalcAffinity --> PackageVector[Package Feature<br/>Vector 15-20 dims]

    PackageVector --> EndFeat([Return: Feature<br/>Vector])

    style StartFeat fill:#e1f5e1
    style EndFeat fill:#ffe1e1
    style PredictStruct fill:#e1f0ff
    style ChromFeatures fill:#fff4e1
```

## Gradient Boosting Training

```mermaid
flowchart TD
    StartTrain([Start: Training Data]) --> InitModel[Initialize Model<br/>Base Score = Mean]

    InitModel --> InitPred[Initialize Predictions<br/>with Base Score]

    InitPred --> LoopTrees{Tree Iteration<br/>< NumTrees?}

    LoopTrees -->|Yes| CalcResidual[Calculate Residuals<br/>Label - Prediction]

    CalcResidual --> Subsample[Subsample Training<br/>Data 80%]

    Subsample --> BuildTree[Build Regression Tree<br/>Max Depth = 6]

    BuildTree --> TreeLoop{Split Node?}

    TreeLoop -->|Yes| FindBestSplit[Find Best Split<br/>Max Variance Reduction]
    FindBestSplit --> SplitNode[Split into Left/Right<br/>Children]
    SplitNode --> TreeLoop

    TreeLoop -->|No| CreateLeaf[Create Leaf Node<br/>Prediction = Mean]

    CreateLeaf --> AddTree[Add Tree to Model]

    AddTree --> UpdatePred[Update Predictions<br/>+= LR × TreePred]

    UpdatePred --> CalcLoss[Calculate Training<br/>Loss MSE]

    CalcLoss --> LogProgress{Iteration % 10<br/>== 0?}

    LogProgress -->|Yes| LogInfo[Log Training<br/>Progress]
    LogProgress -->|No| NextTree

    LogInfo --> NextTree[Tree++]
    NextTree --> LoopTrees

    LoopTrees -->|No| CalcImportance[Calculate Feature<br/>Importances]

    CalcImportance --> EndTrain([Return: Trained<br/>Model])

    style StartTrain fill:#e1f5e1
    style EndTrain fill:#ffe1e1
    style BuildTree fill:#e1f0ff
    style LoopTrees fill:#fff4e1
    style TreeLoop fill:#fff4e1
    style LogProgress fill:#fff4e1
```

## Off-Target Scoring & Ranking

```mermaid
flowchart TD
    StartScore([Start: Candidates]) --> LoopCandidates{For Each<br/>Candidate}

    LoopCandidates --> ExtractFeat[Extract Feature<br/>Vector]

    ExtractFeat --> GBPredict[Gradient Boosting<br/>Prediction]

    GBPredict --> Sigmoid[Apply Sigmoid<br/>to Get Probability]

    Sigmoid --> CalcCI[Calculate Confidence<br/>Interval]

    CalcCI --> StoreResult[Store Scored<br/>Result]

    StoreResult --> NextCandidate[Next Candidate]
    NextCandidate --> LoopCandidates

    LoopCandidates -->|Done| FilterScore{Filter by<br/>Score Threshold}

    FilterScore --> FilterExpr{Filter by<br/>Expression}

    FilterExpr --> CalcComposite[Calculate Composite<br/>Score with Expression]

    CalcComposite --> AddAnnotations[Add Gene Annotations<br/>Biotype, Pathways]

    AddAnnotations --> SortDesc[Sort by Composite<br/>Score Descending]

    SortDesc --> ApplyTopK{Limit to Top K?}

    ApplyTopK -->|Yes| TakeTopK[Take Top K<br/>Results]
    ApplyTopK -->|No| AllResults

    TakeTopK --> EndScore
    AllResults[Use All Results] --> EndScore([Return: Ranked<br/>Off-Targets])

    style StartScore fill:#e1f5e1
    style EndScore fill:#ffe1e1
    style GBPredict fill:#e1f0ff
    style FilterScore fill:#fff4e1
    style FilterExpr fill:#fff4e1
    style ApplyTopK fill:#fff4e1
```

## RNA Secondary Structure Prediction (Vienna RNA)

```mermaid
flowchart TD
    StartRNA([Start: RNA<br/>Sequence]) --> InitDP[Initialize DP Matrix<br/>n × n]

    InitDP --> LoopLength{For Length<br/>2 to n}

    LoopLength --> LoopI{For i = 0 to<br/>n - length}

    LoopI --> CalcJ[j = i + length]

    CalcJ --> CaseUnpaired[Case 1: i Unpaired<br/>dp[i][j] = dp[i+1][j]]

    CaseUnpaired --> LoopK{For k = i+1<br/>to j}

    LoopK --> CanPair{Can Pair<br/>seq[i], seq[k]?}

    CanPair -->|Yes| GetEnergy[Get Base Pair<br/>Energy]
    GetEnergy --> CalcPaired[Calculate Paired<br/>Energy]
    CalcPaired --> UpdateDP[Update dp[i][j]<br/>if Better]

    CanPair -->|No| NextK
    UpdateDP --> NextK[k++]

    NextK --> LoopK

    LoopK -->|Done| NextI[i++]
    NextI --> LoopI

    LoopI -->|Done| NextLength[length++]
    NextLength --> LoopLength

    LoopLength -->|Done| GetMFE[MFE = dp[0][n-1]]

    GetMFE --> Traceback[Traceback to Get<br/>Structure]

    Traceback --> EndRNA([Return: Structure<br/>& Free Energy])

    style StartRNA fill:#e1f5e1
    style EndRNA fill:#ffe1e1
    style LoopLength fill:#fff4e1
    style LoopI fill:#fff4e1
    style LoopK fill:#fff4e1
    style CanPair fill:#fff4e1
```

## Error Handling & Resource Management

```mermaid
flowchart TD
    Operation[Execute<br/>Off-Target Operation] --> CheckMem{Memory Usage<br/>> 90%?}

    CheckMem -->|Yes| ReduceCandidates[Reduce Candidate Set<br/>Stricter Filtering]
    CheckMem -->|No| CheckCandCount

    ReduceCandidates --> CheckCandCount{Candidates<br/>> 10,000?}

    CheckCandCount -->|Yes| FilterByExpr[Filter by Expression<br/>Keep High TPM Only]
    CheckCandCount -->|No| Proceed

    FilterByExpr --> Proceed[Proceed with<br/>Analysis]

    Proceed --> TryOperation{Try Operation}

    TryOperation -->|Success| Success([Success])

    TryOperation -->|Model Load Error| CheckFile{Model File<br/>Exists?}
    TryOperation -->|OOM Error| SwitchBatch[Reduce Batch Size]
    TryOperation -->|Invalid Input| LogReject[Log & Reject<br/>Invalid gRNA]

    CheckFile -->|No| Error1[Error: Model<br/>Not Found]
    CheckFile -->|Yes| Error2[Error: Corrupted<br/>Model]

    SwitchBatch --> RetryOp[Retry with Smaller<br/>Batches]
    RetryOp --> Success

    Error1 --> Fail([Fail])
    Error2 --> Fail
    LogReject --> Fail

    style Success fill:#e1f5e1
    style Fail fill:#ffe1e1
    style CheckMem fill:#fff4e1
    style CheckCandCount fill:#fff4e1
    style TryOperation fill:#fff4e1
    style CheckFile fill:#fff4e1
```

---

## Complexity Summary

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| K-mer Index Building | O(T × L) | O(T × L / K) |
| Candidate Discovery | O(K × log T) | O(K) |
| Feature Extraction | O(K × n²) | O(K × F) |
| Gradient Boosting Training | O(N × F × T × D) | O(T × D) |
| Prediction | O(K × T × D) | O(K × F) |
| Full Pipeline | O(T × L + K × n²) | O(T × L + K × F) |

Where:
- T = transcriptome size
- L = average transcript length
- K = number of candidates
- n = gRNA length
- F = feature dimensions
- N = training samples
- D = tree depth
