# Off-Target Prediction Module - Detailed Pseudocode

## Overview

The Off-Target Prediction Module identifies and scores potential CRISPR-Cas13 off-target sites using machine learning (gradient boosting) combined with biological feature extraction. The module predicts off-target activity based on mismatch patterns, PAM proximity, chromatin accessibility, and RNA secondary structure.

---

## Algorithm 1: Off-Target Site Discovery

**Purpose**: Identify candidate off-target sites using k-mer indexing and fuzzy matching.

```
ALGORITHM: DiscoverOffTargetSites
INPUT: gRNA (string of length 20-24), transcriptome (array of RNA sequences),
       maxMismatches (integer, default = 3)
OUTPUT: candidateSites (array of OffTargetSite objects)

CONSTANTS:
    KMER_SIZE = 10
    SEED_REGION_LENGTH = 12  // Critical seed region at 3' end
    MAX_CANDIDATES_PER_GRNA = 10000

BEGIN
    candidateSites ← []

    // Step 1: Extract seed region from gRNA (most critical for binding)
    seedRegion ← gRNA[length(gRNA) - SEED_REGION_LENGTH : length(gRNA)]

    // Step 2: Build k-mer index for fast candidate lookup
    kmerIndex ← BuildKmerIndex(transcriptome, KMER_SIZE)

    // Step 3: Find all k-mer matches in seed region
    seedKmers ← ExtractKmers(seedRegion, KMER_SIZE)
    potentialMatches ← SET()

    FOR EACH kmer IN seedKmers DO
        // Allow 1 mismatch in k-mer for fuzzy matching
        fuzzyMatches ← kmerIndex.GetFuzzyMatches(kmer, maxMismatches = 1)
        potentialMatches ← potentialMatches UNION fuzzyMatches
    END FOR

    IF length(potentialMatches) > MAX_CANDIDATES_PER_GRNA THEN
        LogWarning("Too many candidates ({length(potentialMatches)}), using stricter filtering")
        // Keep only exact seed matches
        potentialMatches ← FilterToExactSeedMatches(potentialMatches, seedRegion)
    END IF

    // Step 4: Verify full gRNA alignment for each candidate
    FOR EACH match IN potentialMatches DO
        targetSequence ← ExtractTargetSequence(match.transcriptId, match.position, length(gRNA))

        // Calculate mismatch profile
        mismatchProfile ← AlignAndCountMismatches(gRNA, targetSequence)

        IF mismatchProfile.totalMismatches <= maxMismatches THEN
            candidateSites.append({
                transcriptId: match.transcriptId,
                position: match.position,
                targetSequence: targetSequence,
                mismatchProfile: mismatchProfile,
                strand: match.strand
            })
        END IF
    END FOR

    LogInfo("Found {length(candidateSites)} candidate off-target sites for gRNA {gRNA}")

    RETURN candidateSites
END

SUBROUTINE: BuildKmerIndex
INPUT: transcriptome (array of RNA sequences), kmerSize (integer)
OUTPUT: kmerIndex (hash table: kmer → array of positions)

BEGIN
    kmerIndex ← CREATE HASH TABLE

    FOR EACH transcript IN transcriptome DO
        FOR position ← 0 TO length(transcript.sequence) - kmerSize DO
            kmer ← transcript.sequence[position : position + kmerSize]

            IF kmer NOT IN kmerIndex THEN
                kmerIndex[kmer] ← []
            END IF

            kmerIndex[kmer].append({
                transcriptId: transcript.id,
                position: position,
                strand: '+'
            })
        END FOR

        // Also index reverse complement
        rcSequence ← ReverseComplement(transcript.sequence)
        FOR position ← 0 TO length(rcSequence) - kmerSize DO
            kmer ← rcSequence[position : position + kmerSize]

            IF kmer NOT IN kmerIndex THEN
                kmerIndex[kmer] ← []
            END IF

            kmerIndex[kmer].append({
                transcriptId: transcript.id,
                position: length(transcript.sequence) - position,
                strand: '-'
            })
        END FOR
    END FOR

    LogInfo("Built k-mer index with {length(kmerIndex)} unique k-mers")

    RETURN kmerIndex
END

SUBROUTINE: AlignAndCountMismatches
INPUT: gRNA (string), target (string)
OUTPUT: mismatchProfile (object)

BEGIN
    IF length(gRNA) != length(target) THEN
        RETURN error("Length mismatch between gRNA and target")
    END IF

    totalMismatches ← 0
    seedMismatches ← 0
    mismatchPositions ← []

    seedStart ← length(gRNA) - 12  // Last 12 bases are seed region

    FOR i ← 0 TO length(gRNA) - 1 DO
        IF gRNA[i] != target[i] THEN
            totalMismatches ← totalMismatches + 1
            mismatchPositions.append(i)

            IF i >= seedStart THEN
                seedMismatches ← seedMismatches + 1
            END IF
        END IF
    END FOR

    RETURN {
        totalMismatches: totalMismatches,
        seedMismatches: seedMismatches,
        mismatchPositions: mismatchPositions,
        alignment: CreateAlignmentString(gRNA, target)
    }
END
```

**Complexity Analysis**:
- **Time**: O(T × L) for index building, O(K × log T) for k-mer lookup per gRNA
  - T = transcriptome size
  - L = average transcript length
  - K = number of k-mers in gRNA
- **Space**: O(T × L / K) for k-mer index
- **Typical Performance**: 100-1000 ms per gRNA for human transcriptome

---

## Algorithm 2: Feature Extraction

**Purpose**: Extract biological features for machine learning prediction.

```
ALGORITHM: ExtractOffTargetFeatures
INPUT: gRNA (string), candidateSite (OffTargetSite),
       alignedReads (BAM file), chromatinData (BigWig file)
OUTPUT: featureVector (array of floats)

BEGIN
    features ← []

    // === Sequence-based features ===

    // 1. Mismatch distribution features
    mismatchProfile ← candidateSite.mismatchProfile
    features.append(mismatchProfile.totalMismatches)
    features.append(mismatchProfile.seedMismatches)
    features.append(mismatchProfile.totalMismatches - mismatchProfile.seedMismatches)  // Non-seed mismatches

    // 2. Position-weighted mismatch score
    // Mismatches closer to PAM (3' end) are more important
    positionWeightedScore ← 0
    FOR EACH mismatchPos IN mismatchProfile.mismatchPositions DO
        // Weight increases from 0.5 at 5' end to 2.0 at 3' end
        weight ← 0.5 + 1.5 × (mismatchPos / length(gRNA))
        positionWeightedScore ← positionWeightedScore + weight
    END FOR
    features.append(positionWeightedScore)

    // 3. GC content features
    gRNAGC ← CalculateGCContent(gRNA)
    targetGC ← CalculateGCContent(candidateSite.targetSequence)
    features.append(gRNAGC)
    features.append(targetGC)
    features.append(ABS(gRNAGC - targetGC))  // GC content difference

    // 4. Homopolymer runs (AAAA, TTTT, etc. affect binding)
    gRNAMaxHomopolymer ← FindMaxHomopolymerLength(gRNA)
    targetMaxHomopolymer ← FindMaxHomopolymerLength(candidateSite.targetSequence)
    features.append(gRNAMaxHomopolymer)
    features.append(targetMaxHomopolymer)

    // === Structure-based features ===

    // 5. RNA secondary structure energy
    gRNAStructure ← PredictRNAStructure(gRNA)
    targetStructure ← PredictRNAStructure(candidateSite.targetSequence)
    features.append(gRNAStructure.freeEnergy)
    features.append(targetStructure.freeEnergy)
    features.append(ABS(gRNAStructure.freeEnergy - targetStructure.freeEnergy))

    // 6. Structure accessibility (unpaired probability)
    gRNAAccessibility ← CalculateAccessibility(gRNAStructure)
    targetAccessibility ← CalculateAccessibility(targetStructure)
    features.append(gRNAAccessibility)
    features.append(targetAccessibility)

    // === Expression-based features ===

    // 7. Target transcript expression level
    transcriptExpression ← GetTranscriptExpression(candidateSite.transcriptId, alignedReads)
    features.append(LOG(transcriptExpression + 1))  // Log-transformed TPM

    // 8. Local read coverage at off-target site
    localCoverage ← GetLocalCoverage(
        candidateSite.transcriptId,
        candidateSite.position,
        window = 100,
        alignedReads
    )
    features.append(LOG(localCoverage + 1))

    // === Chromatin-based features (if available) ===

    IF chromatinData != null THEN
        // 9. Chromatin accessibility (DNase-seq, ATAC-seq)
        chromatinAccessibility ← GetChromatinAccessibility(
            candidateSite.transcriptId,
            candidateSite.position,
            chromatinData
        )
        features.append(chromatinAccessibility)
    ELSE
        features.append(0)  // Default value if not available
    END IF

    // === Evolutionary conservation features ===

    // 10. PhyloP conservation score
    conservationScore ← GetConservationScore(
        candidateSite.transcriptId,
        candidateSite.position
    )
    features.append(conservationScore)

    // === Distance-based features ===

    // 11. Distance to nearest exon junction
    nearestJunctionDist ← GetDistanceToNearestJunction(
        candidateSite.transcriptId,
        candidateSite.position
    )
    features.append(MIN(nearestJunctionDist, 1000))  // Cap at 1000 bp

    // === Thermodynamic features ===

    // 12. Predicted binding affinity (simplified nearest-neighbor model)
    bindingAffinity ← CalculateBindingAffinity(gRNA, candidateSite.targetSequence)
    features.append(bindingAffinity)

    RETURN features
END

SUBROUTINE: PredictRNAStructure
INPUT: sequence (RNA string)
OUTPUT: structure (Structure object)

BEGIN
    // Use Vienna RNA fold algorithm (simplified)
    // Returns minimum free energy (MFE) structure

    n ← length(sequence)

    // Dynamic programming for MFE calculation
    dp ← CREATE 2D array n × n, initialized to 0

    FOR length ← 2 TO n DO
        FOR i ← 0 TO n - length DO
            j ← i + length

            // Case 1: i unpaired
            dp[i][j] ← dp[i+1][j]

            // Case 2: i paired with k
            FOR k ← i+1 TO j DO
                IF CanPair(sequence[i], sequence[k]) THEN
                    energy ← GetBasePairEnergy(sequence[i], sequence[k])

                    IF k == j THEN
                        dp[i][j] ← MIN(dp[i][j], dp[i+1][k-1] + energy)
                    ELSE
                        dp[i][j] ← MIN(dp[i][j], dp[i+1][k-1] + dp[k+1][j] + energy)
                    END IF
                END IF
            END FOR
        END FOR
    END FOR

    freeEnergy ← dp[0][n-1]

    RETURN {
        freeEnergy: freeEnergy,
        structure: Traceback(dp, sequence)
    }
END

SUBROUTINE: CalculateBindingAffinity
INPUT: gRNA (string), target (string)
OUTPUT: affinity (float, in kcal/mol)

BEGIN
    // Simplified nearest-neighbor thermodynamic model
    // Real implementation would use full Turner parameters

    affinity ← 0

    FOR i ← 0 TO length(gRNA) - 2 DO
        // Get dinucleotide pair
        gRNADimer ← gRNA[i : i+2]
        targetDimer ← target[i : i+2]

        // Look up nearest-neighbor parameters
        stackingEnergy ← GetStackingEnergy(gRNADimer, targetDimer)
        affinity ← affinity + stackingEnergy

        // Add penalty for mismatches
        IF gRNA[i] != target[i] THEN
            affinity ← affinity + 2.5  // Mismatch penalty (kcal/mol)
        END IF
    END FOR

    // Add terminal penalties
    affinity ← affinity + GetTerminalPenalty(gRNA[0], target[0])
    affinity ← affinity + GetTerminalPenalty(gRNA[length(gRNA)-1], target[length(target)-1])

    RETURN affinity
END
```

**Complexity Analysis**:
- **Time**: O(n²) for RNA structure prediction, O(n) for other features
- **Space**: O(n²) for structure DP matrix
- **Feature Vector Size**: 15-20 features
- **Typical Performance**: 10-50 ms per candidate site

---

## Algorithm 3: Gradient Boosting Model Training

**Purpose**: Train gradient boosting classifier to predict off-target activity.

```
ALGORITHM: TrainOffTargetModel
INPUT: trainingData (array of {features, label} pairs),
       config (ModelConfig object)
OUTPUT: trainedModel (GradientBoostingModel)

CONSTANTS:
    NUM_TREES = 100
    MAX_DEPTH = 6
    LEARNING_RATE = 0.1
    MIN_SAMPLES_SPLIT = 20
    SUBSAMPLE_RATIO = 0.8

BEGIN
    model ← {
        trees: [],
        learningRate: LEARNING_RATE,
        featureImportances: [],
        baseScore: 0
    }

    // Step 1: Initialize base prediction (mean of labels)
    model.baseScore ← MEAN(label for {features, label} in trainingData)

    // Initialize predictions with base score
    predictions ← [model.baseScore] × length(trainingData)

    // Step 2: Build trees iteratively
    FOR treeIndex ← 0 TO NUM_TREES - 1 DO
        // Calculate residuals (negative gradient for squared loss)
        residuals ← []
        FOR i ← 0 TO length(trainingData) - 1 DO
            residual ← trainingData[i].label - predictions[i]
            residuals.append(residual)
        END FOR

        // Subsample training data for stochastic gradient boosting
        subsampleSize ← FLOOR(length(trainingData) × SUBSAMPLE_RATIO)
        subsampledIndices ← RandomSample(0 to length(trainingData)-1, subsampleSize)
        subsampledData ← [trainingData[i] for i in subsampledIndices]
        subsampledResiduals ← [residuals[i] for i in subsampledIndices]

        // Build regression tree to predict residuals
        tree ← BuildRegressionTree(
            subsampledData,
            subsampledResiduals,
            maxDepth = MAX_DEPTH,
            minSamplesSplit = MIN_SAMPLES_SPLIT
        )

        model.trees.append(tree)

        // Update predictions
        FOR i ← 0 TO length(trainingData) - 1 DO
            treePrediction ← tree.Predict(trainingData[i].features)
            predictions[i] ← predictions[i] + LEARNING_RATE × treePrediction
        END FOR

        // Calculate training loss
        trainingLoss ← CalculateMSE(predictions, [data.label for data in trainingData])

        IF (treeIndex + 1) MOD 10 == 0 THEN
            LogInfo("Tree {treeIndex + 1}/{NUM_TREES}, Training Loss: {trainingLoss}")
        END IF
    END FOR

    // Step 3: Calculate feature importances
    model.featureImportances ← CalculateFeatureImportances(model.trees)

    LogInfo("Training complete. Top features: {model.featureImportances[:5]}")

    RETURN model
END

SUBROUTINE: BuildRegressionTree
INPUT: data (array), targets (array of residuals), maxDepth, minSamplesSplit
OUTPUT: tree (DecisionTree object)

BEGIN
    tree ← {
        isLeaf: false,
        splitFeature: null,
        splitThreshold: null,
        leftChild: null,
        rightChild: null,
        prediction: null
    }

    // Base case: create leaf node
    IF maxDepth == 0 OR length(data) < minSamplesSplit THEN
        tree.isLeaf ← true
        tree.prediction ← MEAN(targets)
        RETURN tree
    END IF

    // Find best split
    bestSplit ← FindBestSplit(data, targets)

    IF bestSplit == null THEN
        // No improvement possible, create leaf
        tree.isLeaf ← true
        tree.prediction ← MEAN(targets)
        RETURN tree
    END IF

    tree.splitFeature ← bestSplit.feature
    tree.splitThreshold ← bestSplit.threshold

    // Split data into left and right children
    leftIndices ← [i for i in 0 to length(data)-1
                   WHERE data[i].features[bestSplit.feature] <= bestSplit.threshold]
    rightIndices ← [i for i in 0 to length(data)-1
                    WHERE data[i].features[bestSplit.feature] > bestSplit.threshold]

    leftData ← [data[i] for i in leftIndices]
    leftTargets ← [targets[i] for i in leftIndices]
    rightData ← [data[i] for i in rightIndices]
    rightTargets ← [targets[i] for i in rightIndices]

    // Recursively build children
    tree.leftChild ← BuildRegressionTree(leftData, leftTargets, maxDepth - 1, minSamplesSplit)
    tree.rightChild ← BuildRegressionTree(rightData, rightTargets, maxDepth - 1, minSamplesSplit)

    RETURN tree
END

SUBROUTINE: FindBestSplit
INPUT: data (array), targets (array)
OUTPUT: bestSplit (object) or null

BEGIN
    bestGain ← 0
    bestSplit ← null

    numFeatures ← length(data[0].features)

    // Try splitting on each feature
    FOR featureIdx ← 0 TO numFeatures - 1 DO
        // Extract feature values
        featureValues ← [sample.features[featureIdx] for sample in data]

        // Get unique values as potential split thresholds
        sortedValues ← Sort(UNIQUE(featureValues))

        // Try each threshold
        FOR i ← 0 TO length(sortedValues) - 2 DO
            threshold ← (sortedValues[i] + sortedValues[i+1]) / 2

            // Split data
            leftIndices ← [j for j in 0 to length(data)-1
                          WHERE featureValues[j] <= threshold]
            rightIndices ← [j for j in 0 to length(data)-1
                           WHERE featureValues[j] > threshold]

            IF length(leftIndices) < 5 OR length(rightIndices) < 5 THEN
                CONTINUE  // Skip splits that create very small nodes
            END IF

            // Calculate variance reduction (information gain for regression)
            leftTargets ← [targets[j] for j in leftIndices]
            rightTargets ← [targets[j] for j in rightIndices]

            parentVariance ← Variance(targets)
            leftVariance ← Variance(leftTargets)
            rightVariance ← Variance(rightTargets)

            weightedVariance ← (length(leftIndices) × leftVariance +
                               length(rightIndices) × rightVariance) / length(data)

            gain ← parentVariance - weightedVariance

            IF gain > bestGain THEN
                bestGain ← gain
                bestSplit ← {
                    feature: featureIdx,
                    threshold: threshold,
                    gain: gain
                }
            END IF
        END FOR
    END FOR

    IF bestGain < 1e-7 THEN
        RETURN null  // No meaningful split found
    END IF

    RETURN bestSplit
END
```

**Complexity Analysis**:
- **Time**: O(T × N × F × log N) where T = trees, N = samples, F = features
- **Space**: O(T × D) where D = max depth
- **Training Time**: 1-5 minutes for 10,000 samples on CPU
- **Model Size**: 1-10 MB for 100 trees

---

## Algorithm 4: Off-Target Scoring and Ranking

**Purpose**: Predict off-target scores and rank by biological significance.

```
ALGORITHM: ScoreOffTargetSites
INPUT: candidateSites (array), trainedModel (GradientBoostingModel),
       gRNA (string), config (ScoringConfig)
OUTPUT: rankedSites (array of scored OffTargetSite objects)

BEGIN
    scoredSites ← []

    // Step 1: Extract features and predict scores
    FOR EACH candidate IN candidateSites DO
        // Extract feature vector
        features ← ExtractOffTargetFeatures(
            gRNA,
            candidate,
            config.alignedReads,
            config.chromatinData
        )

        // Predict off-target activity score
        rawScore ← PredictGradientBoosting(trainedModel, features)

        // Convert to probability using sigmoid
        probability ← 1 / (1 + EXP(-rawScore))

        // Calculate confidence interval (optional)
        confidenceInterval ← CalculateConfidenceInterval(trainedModel, features)

        scoredSites.append({
            site: candidate,
            offTargetScore: probability,
            confidenceLower: confidenceInterval.lower,
            confidenceUpper: confidenceInterval.upper,
            features: features
        })
    END FOR

    // Step 2: Apply biological filters
    filteredSites ← []
    FOR EACH scored IN scoredSites DO
        // Filter by score threshold
        IF scored.offTargetScore >= config.minScoreThreshold THEN
            // Filter by expression level
            expression ← GetTranscriptExpression(scored.site.transcriptId, config.alignedReads)
            IF expression >= config.minExpressionThreshold THEN
                filteredSites.append(scored)
            END IF
        END IF
    END FOR

    // Step 3: Rank by composite score
    FOR EACH scored IN filteredSites DO
        // Composite score = off-target probability × expression weight
        expression ← GetTranscriptExpression(scored.site.transcriptId, config.alignedReads)
        expressionWeight ← LOG(expression + 1) / 10  // Normalize

        scored.compositeScore ← scored.offTargetScore × (1 + expressionWeight)

        // Add biological context annotations
        scored.annotations ← {
            geneSymbol: GetGeneSymbol(scored.site.transcriptId),
            biotype: GetBiotype(scored.site.transcriptId),
            isEssentialGene: IsEssentialGene(scored.site.transcriptId),
            pathways: GetAssociatedPathways(scored.site.transcriptId)
        }
    END FOR

    // Step 4: Sort by composite score (descending)
    rankedSites ← Sort(filteredSites, by = compositeScore, descending = true)

    // Step 5: Apply top-k filtering if requested
    IF config.maxReturnedSites > 0 THEN
        rankedSites ← rankedSites[0 : config.maxReturnedSites]
    END IF

    LogInfo("Scored and ranked {length(rankedSites)} off-target sites")

    RETURN rankedSites
END

SUBROUTINE: PredictGradientBoosting
INPUT: model (GradientBoostingModel), features (array of floats)
OUTPUT: prediction (float)

BEGIN
    prediction ← model.baseScore

    FOR EACH tree IN model.trees DO
        treePrediction ← PredictTree(tree, features)
        prediction ← prediction + model.learningRate × treePrediction
    END FOR

    RETURN prediction
END

SUBROUTINE: PredictTree
INPUT: tree (DecisionTree), features (array)
OUTPUT: prediction (float)

BEGIN
    IF tree.isLeaf THEN
        RETURN tree.prediction
    END IF

    featureValue ← features[tree.splitFeature]

    IF featureValue <= tree.splitThreshold THEN
        RETURN PredictTree(tree.leftChild, features)
    ELSE
        RETURN PredictTree(tree.rightChild, features)
    END IF
END

SUBROUTINE: CalculateConfidenceInterval
INPUT: model (GradientBoostingModel), features (array)
OUTPUT: interval (object with lower and upper bounds)

BEGIN
    // Use bootstrap aggregation to estimate uncertainty
    NUM_BOOTSTRAP_SAMPLES ← 100
    predictions ← []

    FOR i ← 0 TO NUM_BOOTSTRAP_SAMPLES - 1 DO
        // Randomly sample trees with replacement
        sampledTrees ← RandomSampleWithReplacement(model.trees, length(model.trees))

        // Create temporary model
        tempModel ← {
            trees: sampledTrees,
            learningRate: model.learningRate,
            baseScore: model.baseScore
        }

        pred ← PredictGradientBoosting(tempModel, features)
        predictions.append(pred)
    END FOR

    // Calculate percentiles
    sortedPredictions ← Sort(predictions)
    lowerIdx ← FLOOR(NUM_BOOTSTRAP_SAMPLES × 0.025)
    upperIdx ← FLOOR(NUM_BOOTSTRAP_SAMPLES × 0.975)

    RETURN {
        lower: sortedPredictions[lowerIdx],
        upper: sortedPredictions[upperIdx],
        median: sortedPredictions[FLOOR(NUM_BOOTSTRAP_SAMPLES / 2)]
    }
END
```

**Complexity Analysis**:
- **Time**: O(K × T × D) where K = candidates, T = trees, D = depth
- **Space**: O(K × F) where F = feature dimensions
- **Inference Time**: 0.1-1 ms per candidate site

---

## Algorithm 5: Off-Target Prediction Pipeline

**Purpose**: Main pipeline orchestrating all off-target prediction steps.

```
ALGORITHM: PredictOffTargets
INPUT: gRNA (string), transcriptomeFile (path), alignedReadsFile (path),
       modelFile (path), config (PipelineConfig)
OUTPUT: predictionResults (OffTargetPredictionResults)

BEGIN
    startTime ← GetCurrentTime()

    // Step 1: Load resources
    LogInfo("Loading transcriptome and model...")
    transcriptome ← LoadTranscriptome(transcriptomeFile)
    trainedModel ← LoadModel(modelFile)
    alignedReads ← OpenBAM(alignedReadsFile)

    // Step 2: Discover candidate sites
    LogInfo("Discovering candidate off-target sites...")
    candidateSites ← DiscoverOffTargetSites(
        gRNA,
        transcriptome,
        maxMismatches = config.maxMismatches
    )

    LogInfo("Found {length(candidateSites)} candidates")

    IF length(candidateSites) == 0 THEN
        LogWarning("No candidate sites found")
        RETURN {
            gRNA: gRNA,
            offTargetSites: [],
            statistics: {totalCandidates: 0, totalPredicted: 0}
        }
    END IF

    // Step 3: Score candidates (parallelized)
    LogInfo("Scoring candidate sites...")
    rankedSites ← ScoreOffTargetSites(
        candidateSites,
        trainedModel,
        gRNA,
        config
    )

    // Step 4: Generate summary statistics
    statistics ← {
        totalCandidates: length(candidateSites),
        totalPredicted: length(rankedSites),
        highConfidenceSites: COUNT(site for site in rankedSites
                                   WHERE site.offTargetScore >= 0.7),
        mediumConfidenceSites: COUNT(site for site in rankedSites
                                     WHERE site.offTargetScore >= 0.3 AND site.offTargetScore < 0.7),
        lowConfidenceSites: COUNT(site for site in rankedSites
                                  WHERE site.offTargetScore < 0.3),
        meanScore: MEAN(site.offTargetScore for site in rankedSites),
        medianScore: MEDIAN(site.offTargetScore for site in rankedSites),
        processingTime: GetCurrentTime() - startTime
    }

    // Step 5: Create result object
    results ← {
        gRNA: gRNA,
        offTargetSites: rankedSites,
        statistics: statistics,
        modelInfo: {
            modelPath: modelFile,
            numTrees: length(trainedModel.trees),
            featureImportances: trainedModel.featureImportances
        },
        timestamp: GetCurrentTime()
    }

    LogInfo("Off-target prediction complete in {statistics.processingTime}s")
    LogInfo("Found {statistics.highConfidenceSites} high-confidence off-targets")

    RETURN results
END
```

**Complexity Analysis**:
- **Time**: O(T × L + K × log K + K × F × D) for full pipeline
- **Space**: O(T × L + K × F) where K = candidates
- **End-to-End Performance**: 1-10 seconds per gRNA

---

## Error Handling

### 1. Invalid Input Validation

```
FUNCTION: ValidateGRNA
INPUT: gRNA (string)
OUTPUT: boolean (valid or not)

BEGIN
    // Check length
    IF length(gRNA) < 18 OR length(gRNA) > 28 THEN
        LogError("Invalid gRNA length: {length(gRNA)} (expected 18-28)")
        RETURN false
    END IF

    // Check valid RNA bases
    validBases ← {'A', 'U', 'C', 'G'}
    FOR EACH base IN gRNA DO
        IF base NOT IN validBases THEN
            LogError("Invalid base '{base}' in gRNA (must be A, U, C, or G)")
            RETURN false
        END IF
    END FOR

    // Check for excessive homopolymers
    maxHomopolymer ← FindMaxHomopolymerLength(gRNA)
    IF maxHomopolymer >= 6 THEN
        LogWarning("gRNA contains homopolymer run of length {maxHomopolymer}")
    END IF

    // Check GC content
    gcContent ← CalculateGCContent(gRNA)
    IF gcContent < 0.2 OR gcContent > 0.8 THEN
        LogWarning("Extreme GC content: {gcContent × 100}%")
    END IF

    RETURN true
END
```

### 2. Model Loading Error Handling

```
FUNCTION: LoadModelSafely
INPUT: modelPath (string)
OUTPUT: model (GradientBoostingModel) or error

BEGIN
    TRY
        IF NOT FileExists(modelPath) THEN
            RETURN error("MODEL_NOT_FOUND", "Model file not found: {modelPath}")
        END IF

        model ← DeserializeModel(modelPath)

        // Validate model structure
        IF model.trees == null OR length(model.trees) == 0 THEN
            RETURN error("INVALID_MODEL", "Model has no trees")
        END IF

        IF model.learningRate <= 0 OR model.learningRate > 1 THEN
            RETURN error("INVALID_MODEL", "Invalid learning rate: {model.learningRate}")
        END IF

        LogInfo("Loaded model with {length(model.trees)} trees")
        RETURN model

    CATCH IOException AS e DO
        RETURN error("IO_ERROR", "Failed to read model file: {e.message}")
    CATCH DeserializationError AS e DO
        RETURN error("DESERIALIZATION_ERROR", "Corrupted model file: {e.message}")
    END TRY
END
```

### 3. Resource Exhaustion Handling

```
FUNCTION: DiscoverOffTargetsWithLimits
INPUT: gRNA, transcriptome, maxMismatches, memoryLimitMB
OUTPUT: candidateSites (array)

BEGIN
    candidateSites ← []
    currentMemoryMB ← GetProcessMemoryUsage()

    IF currentMemoryMB > memoryLimitMB × 0.9 THEN
        LogWarning("Approaching memory limit, using stricter filtering")
        // Reduce candidate set aggressively
        maxMismatches ← MIN(2, maxMismatches)
    END IF

    candidateSites ← DiscoverOffTargetSites(gRNA, transcriptome, maxMismatches)

    // If still too many candidates, filter by expression
    IF length(candidateSites) > 10000 THEN
        LogInfo("Too many candidates ({length(candidateSites)}), filtering by expression")
        candidateSites ← FilterByExpression(candidateSites, minExpression = 1.0)
    END IF

    RETURN candidateSites
END
```

---

## Performance Optimization

### 1. Parallel Feature Extraction

```
FUNCTION: ExtractFeaturesBatch
INPUT: candidates (array), gRNA, config, numThreads
OUTPUT: featureMatrix (2D array)

BEGIN
    batchSize ← CEILING(length(candidates) / numThreads)

    featureMatrix ← ParallelMap(
        candidates,
        FUNCTION(candidate)
            RETURN ExtractOffTargetFeatures(gRNA, candidate, config)
        END FUNCTION,
        numThreads = numThreads
    )

    RETURN featureMatrix
END
```

### 2. Caching Strategy

```
DATA STRUCTURE: TranscriptomeCache
    kmerIndex: LRU Cache (size = 10GB)
    expressionData: Hash table
    structureData: LRU Cache (size = 1GB)

FUNCTION: GetCachedKmerIndex
INPUT: transcriptomeFile (path)
OUTPUT: kmerIndex

BEGIN
    cacheKey ← Hash(transcriptomeFile)

    IF cacheKey IN TranscriptomeCache.kmerIndex THEN
        LogInfo("Using cached k-mer index")
        RETURN TranscriptomeCache.kmerIndex[cacheKey]
    ELSE
        transcriptome ← LoadTranscriptome(transcriptomeFile)
        kmerIndex ← BuildKmerIndex(transcriptome)
        TranscriptomeCache.kmerIndex[cacheKey] ← kmerIndex
        RETURN kmerIndex
    END IF
END
```

---

## Testing and Validation

### 1. Unit Tests

```
TEST: FeatureExtractionConsistency
BEGIN
    gRNA ← "GUUUUAGAGCUAUGCUGUUUUG"
    target ← "GUUUUAGAGCUAUGCUGUUUUA"  // 1 mismatch at end

    candidate ← CreateMockCandidate(target)
    features ← ExtractOffTargetFeatures(gRNA, candidate, mockConfig)

    ASSERT features[0] == 1  // Total mismatches
    ASSERT features[1] == 1  // Seed mismatches
    ASSERT length(features) == 15  // Expected feature count
END

TEST: GradientBoostingPrediction
BEGIN
    model ← CreateMockModel(numTrees = 10)
    features ← [0.5, 1.0, 0.3, 2.1, 0.8]

    prediction ← PredictGradientBoosting(model, features)

    ASSERT prediction >= 0 AND prediction <= 1
END
```

### 2. Integration Tests

```
TEST: EndToEndOffTargetPrediction
BEGIN
    // Use small test dataset
    gRNA ← "GUUUUAGAGCUAUGCUGUUUUG"
    transcriptome ← LoadTestTranscriptome("test_data/small_transcriptome.fa")
    model ← LoadTestModel("test_data/test_model.gbm")

    results ← PredictOffTargets(gRNA, transcriptome, model, testConfig)

    ASSERT results.statistics.totalCandidates > 0
    ASSERT results.statistics.totalPredicted <= results.statistics.totalCandidates
    ASSERT ALL(site.offTargetScore >= 0 AND site.offTargetScore <= 1
               for site in results.offTargetSites)
END
```

---

## References

1. Haeussler, M., et al. (2016). Evaluation of off-target and on-target scoring algorithms. *Genome Biology*.
2. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*.
3. Doench, J. G., et al. (2016). Optimized sgRNA design to maximize activity and minimize off-target effects. *Nature Biotechnology*.

---

**Next Module**: Differential Expression Analysis Module
