# Read Alignment Module - Detailed Pseudocode

## Overview

The Read Alignment Module aligns RNA-seq reads to a reference genome using a hybrid approach combining Burrows-Wheeler Transform (BWT) for fast seeding and Smith-Waterman algorithm for precise local alignment.

---

## Algorithm 1: BWT Index Construction

**Purpose**: Build compressed suffix array index for fast seed lookup.

```
ALGORITHM: BuildBWTIndex
INPUT: referenceGenome (string of length m)
OUTPUT: bwtIndex (BWT structure with auxiliary arrays)

CONSTANTS:
    ALPHABET = ['A', 'C', 'G', 'T', 'N']
    SUFFIX_ARRAY_SAMPLE_RATE = 32

BEGIN
    // Step 1: Construct suffix array
    suffixArray ← ConstructSuffixArray(referenceGenome)

    // Step 2: Build BWT from suffix array
    bwtString ← []
    FOR i ← 0 TO m - 1 DO
        IF suffixArray[i] == 0 THEN
            bwtString.append('$')  // Sentinel character
        ELSE
            bwtString.append(referenceGenome[suffixArray[i] - 1])
        END IF
    END FOR

    // Step 3: Build occurrence tables for fast rank queries
    occurrenceTable ← BuildOccurrenceTable(bwtString, ALPHABET)

    // Step 4: Build count table (C-array)
    countTable ← [0] × (|ALPHABET| + 1)
    FOR base IN ALPHABET DO
        countTable[base] ← CountOccurrences(bwtString, base)
    END FOR

    // Compute cumulative counts
    FOR i ← 1 TO |ALPHABET| DO
        countTable[i] ← countTable[i] + countTable[i-1]
    END FOR

    // Step 5: Sample suffix array positions for position lookup
    sampledSA ← []
    FOR i ← 0 TO m - 1 DO
        IF i MOD SUFFIX_ARRAY_SAMPLE_RATE == 0 THEN
            sampledSA[i] ← suffixArray[i]
        END IF
    END FOR

    RETURN {
        bwtString: bwtString,
        occurrenceTable: occurrenceTable,
        countTable: countTable,
        sampledSA: sampledSA,
        sampleRate: SUFFIX_ARRAY_SAMPLE_RATE
    }
END

SUBROUTINE: ConstructSuffixArray
INPUT: text (string of length m)
OUTPUT: suffixArray (array of integers)

BEGIN
    // Use DC3 (Difference Cover 3) algorithm for O(m) construction
    // This is a well-known linear-time suffix array construction algorithm

    // Step 1: Divide suffixes into three groups based on position mod 3
    group0 ← []  // Positions divisible by 3
    group1 ← []  // Positions = 1 mod 3
    group2 ← []  // Positions = 2 mod 3

    FOR i ← 0 TO m - 1 DO
        IF i MOD 3 == 1 THEN
            group1.append(i)
        ELSE IF i MOD 3 == 2 THEN
            group2.append(i)
        ELSE
            group0.append(i)
        END IF
    END FOR

    // Step 2: Recursively sort groups 1 and 2
    group12 ← group1 + group2
    sortedGroup12 ← RadixSort(group12, text)

    // Step 3: Sort group 0 using sorted group 1
    sortedGroup0 ← SortUsingRanks(group0, sortedGroup12, text)

    // Step 4: Merge sorted groups
    suffixArray ← Merge(sortedGroup0, sortedGroup12)

    RETURN suffixArray
END

SUBROUTINE: BuildOccurrenceTable
INPUT: bwtString (string), alphabet (array of characters)
OUTPUT: occTable (2D array)

BEGIN
    m ← length(bwtString)
    occTable ← CREATE 2D array [m + 1] × |alphabet|

    // Initialize first row to zeros
    FOR base IN alphabet DO
        occTable[0][base] ← 0
    END FOR

    // Fill occurrence table with cumulative counts
    FOR i ← 1 TO m DO
        FOR base IN alphabet DO
            occTable[i][base] ← occTable[i-1][base]
        END FOR

        currentBase ← bwtString[i-1]
        IF currentBase IN alphabet THEN
            occTable[i][currentBase] ← occTable[i][currentBase] + 1
        END IF
    END FOR

    RETURN occTable
END
```

**Complexity Analysis**:
- **Time**: O(m) for DC3 suffix array construction, O(m × |Σ|) for occurrence table (Σ = alphabet)
- **Space**: O(m) for BWT string and auxiliary structures
- **Optimization**: Use wavelet tree for O(log |Σ|) rank queries with O(m log |Σ|) space

---

## Algorithm 2: BWT Exact Match (Backward Search)

**Purpose**: Find all exact matches of a pattern in the reference genome.

```
ALGORITHM: BWTExactMatch
INPUT: pattern (string of length n), bwtIndex (BWT structure)
OUTPUT: matchRanges (array of [start, end] genomic positions)

BEGIN
    // Initialize search range to entire BWT
    top ← 0
    bottom ← length(bwtIndex.bwtString)

    // Search pattern backwards (right to left)
    FOR i ← n - 1 DOWN TO 0 DO
        base ← pattern[i]

        IF NOT base IN bwtIndex.alphabet THEN
            RETURN []  // Invalid character
        END IF

        // Update range using LF-mapping
        top ← bwtIndex.countTable[base] +
              bwtIndex.occurrenceTable[top][base]
        bottom ← bwtIndex.countTable[base] +
                 bwtIndex.occurrenceTable[bottom][base]

        IF top >= bottom THEN
            RETURN []  // No matches found
        END IF
    END FOR

    // Convert SA range to genomic positions
    matchRanges ← []
    FOR saIndex ← top TO bottom - 1 DO
        genomicPos ← GetGenomicPosition(saIndex, bwtIndex)
        matchRanges.append(genomicPos)
    END FOR

    RETURN matchRanges
END

SUBROUTINE: GetGenomicPosition
INPUT: saIndex (integer), bwtIndex (BWT structure)
OUTPUT: genomicPosition (integer)

BEGIN
    steps ← 0
    currentIndex ← saIndex

    // Walk back until we hit a sampled position
    WHILE currentIndex NOT IN bwtIndex.sampledSA DO
        base ← bwtIndex.bwtString[currentIndex]
        currentIndex ← bwtIndex.countTable[base] +
                      bwtIndex.occurrenceTable[currentIndex][base]
        steps ← steps + 1
    END WHILE

    genomicPosition ← bwtIndex.sampledSA[currentIndex] + steps

    RETURN genomicPosition
END
```

**Complexity Analysis**:
- **Time**: O(n) for backward search, O(n × log m) amortized for position lookup
- **Space**: O(1) auxiliary space
- **Query Time**: Typically 10-100 microseconds for 100bp read

---

## Algorithm 3: Seed-and-Extend Alignment

**Purpose**: Use BWT for fast seeding, then extend with Smith-Waterman.

```
ALGORITHM: SeedAndExtendAlignment
INPUT: read (string of length n), referenceGenome, bwtIndex, config
OUTPUT: alignmentResult (AlignmentRecord or null)

CONSTANTS:
    SEED_LENGTH = 20
    MAX_SEED_HITS = 1000
    MIN_SEED_COVERAGE = 0.7
    MAX_EDIT_DISTANCE = 5

BEGIN
    seeds ← []

    // Step 1: Extract seeds from read
    FOR i ← 0 TO n - SEED_LENGTH STEP SEED_LENGTH DO
        seedSequence ← read[i : i + SEED_LENGTH]
        seedMatches ← BWTExactMatch(seedSequence, bwtIndex)

        IF length(seedMatches) > 0 AND length(seedMatches) < MAX_SEED_HITS THEN
            FOR EACH match IN seedMatches DO
                seeds.append({
                    readPos: i,
                    genomePos: match,
                    length: SEED_LENGTH
                })
            END FOR
        END IF
    END FOR

    IF length(seeds) == 0 THEN
        RETURN null  // No seeds found
    END IF

    // Step 2: Cluster seeds by genomic location
    seedClusters ← ClusterSeeds(seeds, windowSize = 1000)

    // Step 3: Evaluate each cluster
    bestAlignment ← null
    bestScore ← -INFINITY

    FOR EACH cluster IN seedClusters DO
        // Check if cluster covers enough of the read
        coverage ← CalculateCoverage(cluster, n)
        IF coverage < MIN_SEED_COVERAGE THEN
            CONTINUE
        END IF

        // Extract reference region for alignment
        clusterStart ← MIN(seed.genomePos for seed in cluster)
        clusterEnd ← MAX(seed.genomePos + seed.length for seed in cluster)

        // Add flanking regions for soft clipping
        refStart ← MAX(0, clusterStart - 100)
        refEnd ← MIN(length(referenceGenome), clusterEnd + 100)
        refRegion ← referenceGenome[refStart : refEnd]

        // Step 4: Perform Smith-Waterman alignment
        alignment ← SmithWaterman(read, refRegion, config.scoringMatrix)

        IF alignment.score > bestScore THEN
            bestScore ← alignment.score
            bestAlignment ← {
                chromosome: bwtIndex.chromosome,
                position: refStart + alignment.refStart,
                cigar: alignment.cigar,
                score: alignment.score,
                mapq: CalculateMAPQ(alignment, seedClusters)
            }
        END IF
    END FOR

    RETURN bestAlignment
END

SUBROUTINE: ClusterSeeds
INPUT: seeds (array), windowSize (integer)
OUTPUT: clusters (array of seed arrays)

BEGIN
    IF length(seeds) == 0 THEN
        RETURN []
    END IF

    // Sort seeds by genomic position
    sortedSeeds ← Sort(seeds, by = genomePos)

    clusters ← []
    currentCluster ← [sortedSeeds[0]]

    FOR i ← 1 TO length(sortedSeeds) - 1 DO
        prevSeed ← sortedSeeds[i-1]
        currentSeed ← sortedSeeds[i]

        distance ← currentSeed.genomePos - prevSeed.genomePos

        IF distance <= windowSize THEN
            currentCluster.append(currentSeed)
        ELSE
            clusters.append(currentCluster)
            currentCluster ← [currentSeed]
        END IF
    END FOR

    clusters.append(currentCluster)

    // Sort clusters by total seed weight
    sortedClusters ← Sort(clusters, by = totalSeedWeight, descending = true)

    RETURN sortedClusters
END

SUBROUTINE: CalculateCoverage
INPUT: cluster (array of seeds), readLength (integer)
OUTPUT: coverage (float in [0, 1])

BEGIN
    coveredPositions ← SET()

    FOR EACH seed IN cluster DO
        FOR pos ← seed.readPos TO seed.readPos + seed.length - 1 DO
            coveredPositions.add(pos)
        END FOR
    END FOR

    coverage ← length(coveredPositions) / readLength

    RETURN coverage
END
```

**Complexity Analysis**:
- **Time**: O(k × n²) where k = number of seed clusters (typically k << 10)
- **Space**: O(n + m) for reference region extraction
- **Typical Performance**: 1-5 ms per read on modern CPU

---

## Algorithm 4: Smith-Waterman Local Alignment

**Purpose**: Compute optimal local alignment with affine gap penalties.

```
ALGORITHM: SmithWaterman
INPUT: query (string of length n), reference (string of length m), scoring
OUTPUT: alignment (Alignment object)

CONSTANTS:
    MATCH_SCORE = 2
    MISMATCH_PENALTY = -3
    GAP_OPEN = -5
    GAP_EXTEND = -2

BEGIN
    // Initialize DP matrices
    // H: main score matrix
    // E: gap in query
    // F: gap in reference
    H ← CREATE 2D array (n+1) × (m+1), initialized to 0
    E ← CREATE 2D array (n+1) × (m+1), initialized to -INFINITY
    F ← CREATE 2D array (n+1) × (m+1), initialized to -INFINITY

    maxScore ← 0
    maxI ← 0
    maxJ ← 0

    // Fill DP matrices
    FOR i ← 1 TO n DO
        FOR j ← 1 TO m DO
            // Calculate gap scores
            E[i][j] ← MAX(
                E[i][j-1] + GAP_EXTEND,
                H[i][j-1] + GAP_OPEN
            )

            F[i][j] ← MAX(
                F[i-1][j] + GAP_EXTEND,
                H[i-1][j] + GAP_OPEN
            )

            // Calculate match/mismatch score
            IF query[i-1] == reference[j-1] THEN
                matchScore ← H[i-1][j-1] + MATCH_SCORE
            ELSE
                matchScore ← H[i-1][j-1] + MISMATCH_PENALTY
            END IF

            // Take maximum (including 0 for local alignment)
            H[i][j] ← MAX(
                0,
                matchScore,
                E[i][j],
                F[i][j]
            )

            // Track maximum score position
            IF H[i][j] > maxScore THEN
                maxScore ← H[i][j]
                maxI ← i
                maxJ ← j
            END IF
        END FOR
    END FOR

    // Traceback to construct alignment
    alignment ← Traceback(H, E, F, query, reference, maxI, maxJ)
    alignment.score ← maxScore

    RETURN alignment
END

SUBROUTINE: Traceback
INPUT: H, E, F (DP matrices), query, reference, startI, startJ
OUTPUT: alignment (Alignment object)

BEGIN
    cigar ← []
    queryAlign ← []
    refAlign ← []

    i ← startI
    j ← startJ

    WHILE i > 0 OR j > 0 DO
        IF H[i][j] == 0 THEN
            BREAK  // Reached end of local alignment
        END IF

        currentScore ← H[i][j]

        // Determine which matrix we came from
        IF i > 0 AND j > 0 AND currentScore == H[i-1][j-1] + Score(query[i-1], reference[j-1]) THEN
            // Match or mismatch
            IF query[i-1] == reference[j-1] THEN
                cigar.prepend('M')
            ELSE
                cigar.prepend('X')
            END IF
            queryAlign.prepend(query[i-1])
            refAlign.prepend(reference[j-1])
            i ← i - 1
            j ← j - 1

        ELSE IF i > 0 AND currentScore == F[i][j] THEN
            // Gap in reference
            cigar.prepend('I')
            queryAlign.prepend(query[i-1])
            refAlign.prepend('-')
            i ← i - 1

        ELSE IF j > 0 AND currentScore == E[i][j] THEN
            // Gap in query
            cigar.prepend('D')
            queryAlign.prepend('-')
            refAlign.prepend(reference[j-1])
            j ← j - 1
        ELSE
            BREAK  // Shouldn't happen with correct implementation
        END IF
    END WHILE

    // Compress CIGAR string (e.g., "MMMM" -> "4M")
    compressedCigar ← CompressCigar(cigar)

    RETURN {
        cigar: compressedCigar,
        queryStart: i,
        refStart: j,
        queryEnd: startI,
        refEnd: startJ,
        queryAlign: queryAlign,
        refAlign: refAlign
    }
END

SUBROUTINE: CompressCigar
INPUT: cigar (array of characters)
OUTPUT: compressed (string)

BEGIN
    IF length(cigar) == 0 THEN
        RETURN "*"
    END IF

    result ← ""
    currentOp ← cigar[0]
    count ← 1

    FOR i ← 1 TO length(cigar) - 1 DO
        IF cigar[i] == currentOp THEN
            count ← count + 1
        ELSE
            result ← result + ToString(count) + currentOp
            currentOp ← cigar[i]
            count ← 1
        END IF
    END FOR

    result ← result + ToString(count) + currentOp

    RETURN result
END
```

**Complexity Analysis**:
- **Time**: O(n × m) for DP matrix fill, O(n + m) for traceback
- **Space**: O(n × m) for three DP matrices
- **Optimization**: Banded alignment reduces to O(n × w) where w = band width (typically 50-100)

---

## Algorithm 5: MAPQ Calculation

**Purpose**: Calculate mapping quality score (0-60) indicating alignment confidence.

```
ALGORITHM: CalculateMAPQ
INPUT: alignment (Alignment object), alternativeAlignments (array)
OUTPUT: mapq (integer in [0, 60])

BEGIN
    // Use Phred-scaled probability of incorrect mapping
    // MAPQ = -10 × log10(P(mapping is wrong))

    bestScore ← alignment.score

    // Find second-best score
    secondBestScore ← -INFINITY
    FOR EACH altAlign IN alternativeAlignments DO
        IF altAlign.score > secondBestScore AND altAlign.score < bestScore THEN
            secondBestScore ← altAlign.score
        END IF
    END FOR

    // Calculate score difference
    IF secondBestScore == -INFINITY THEN
        // Unique alignment
        scoreDiff ← bestScore
    ELSE
        scoreDiff ← bestScore - secondBestScore
    END IF

    // Empirical formula based on score difference
    // Tuned for RNA-seq data with typical error rates
    IF scoreDiff >= 40 THEN
        mapq ← 60
    ELSE IF scoreDiff >= 20 THEN
        mapq ← 40 + (scoreDiff - 20) × 1.0
    ELSE IF scoreDiff >= 10 THEN
        mapq ← 20 + (scoreDiff - 10) × 2.0
    ELSE
        mapq ← MAX(0, scoreDiff × 2.0)
    END IF

    // Penalize for alignment errors
    numMismatches ← CountMismatches(alignment)
    numGaps ← CountGaps(alignment)

    errorPenalty ← (numMismatches × 2) + (numGaps × 3)
    mapq ← MAX(0, mapq - errorPenalty)

    RETURN FLOOR(mapq)
END
```

**Complexity Analysis**:
- **Time**: O(k + n) where k = number of alternative alignments
- **Space**: O(1)

---

## Algorithm 6: Main Alignment Pipeline

**Purpose**: Orchestrate full alignment workflow with parallelization.

```
ALGORITHM: AlignReads
INPUT: fastqFile (path), referenceGenome (path), outputBam (path), config
OUTPUT: alignmentStats (AlignmentStatistics object)

BEGIN
    // Step 1: Load or build BWT index
    IF IndexFileExists(referenceGenome + ".bwt") THEN
        bwtIndex ← LoadBWTIndex(referenceGenome + ".bwt")
    ELSE
        Print("Building BWT index...")
        genome ← LoadGenome(referenceGenome)
        bwtIndex ← BuildBWTIndex(genome)
        SaveBWTIndex(bwtIndex, referenceGenome + ".bwt")
    END IF

    // Step 2: Open input/output files
    fastqReader ← OpenFASTQ(fastqFile)
    bamWriter ← OpenBAM(outputBam, writeMode = true)

    // Step 3: Initialize statistics
    stats ← {
        totalReads: 0,
        alignedReads: 0,
        unmappedReads: 0,
        multiMappedReads: 0,
        totalAlignmentTime: 0
    }

    // Step 4: Process reads in parallel batches
    BATCH_SIZE ← 10000
    NUM_THREADS ← config.threads

    WHILE NOT fastqReader.EOF DO
        readBatch ← fastqReader.ReadBatch(BATCH_SIZE)

        // Parallel alignment
        alignmentResults ← ParallelMap(readBatch, FUNCTION(read)
            startTime ← GetCurrentTime()
            alignment ← SeedAndExtendAlignment(read, genome, bwtIndex, config)
            endTime ← GetCurrentTime()

            RETURN {
                read: read,
                alignment: alignment,
                alignmentTime: endTime - startTime
            }
        END FUNCTION, numThreads = NUM_THREADS)

        // Write results and update statistics
        FOR EACH result IN alignmentResults DO
            stats.totalReads ← stats.totalReads + 1
            stats.totalAlignmentTime ← stats.totalAlignmentTime + result.alignmentTime

            IF result.alignment != null THEN
                stats.alignedReads ← stats.alignedReads + 1

                IF result.alignment.mapq < 10 THEN
                    stats.multiMappedReads ← stats.multiMappedReads + 1
                END IF

                bamWriter.WriteAlignment(result.read, result.alignment)
            ELSE
                stats.unmappedReads ← stats.unmappedReads + 1
                bamWriter.WriteUnmapped(result.read)
            END IF
        END FOR

        // Progress reporting
        IF stats.totalReads MOD 100000 == 0 THEN
            Print("Processed {stats.totalReads} reads...")
        END IF
    END WHILE

    // Step 5: Finalize and generate report
    bamWriter.Close()
    fastqReader.Close()

    stats.alignmentRate ← stats.alignedReads / stats.totalReads
    stats.avgAlignmentTime ← stats.totalAlignmentTime / stats.totalReads

    RETURN stats
END
```

**Complexity Analysis**:
- **Time**: O(N × n × m / p) where N = total reads, p = parallelism factor
- **Space**: O(B × n) where B = batch size
- **Throughput**: 10,000-50,000 reads/second on 32-core system

---

## Error Handling

### 1. Invalid Input Data

```
FUNCTION: ValidateFASTQRead
INPUT: readRecord
OUTPUT: boolean (valid or not)

BEGIN
    // Check read ID
    IF NOT readRecord.id.startsWith('@') THEN
        LogError("Invalid FASTQ format: missing @ in read ID")
        RETURN false
    END IF

    // Check sequence quality
    IF length(readRecord.sequence) != length(readRecord.quality) THEN
        LogError("Sequence-quality length mismatch in read {readRecord.id}")
        RETURN false
    END IF

    // Check for valid bases
    validBases ← {'A', 'C', 'G', 'T', 'N'}
    FOR EACH base IN readRecord.sequence DO
        IF base NOT IN validBases THEN
            LogWarning("Invalid base '{base}' in read {readRecord.id}, replacing with 'N'")
            readRecord.sequence.replace(base, 'N')
        END IF
    END FOR

    RETURN true
END
```

### 2. Memory Management

```
FUNCTION: AlignWithMemoryLimit
INPUT: read, genome, bwtIndex, maxMemoryMB
OUTPUT: alignment or error

BEGIN
    currentMemoryMB ← GetProcessMemoryUsage()

    IF currentMemoryMB > maxMemoryMB × 0.9 THEN
        // Approaching memory limit, use memory-efficient alignment
        RETURN BandedSmithWaterman(read, genome, bandWidth = 50)
    ELSE
        // Sufficient memory, use full Smith-Waterman
        RETURN SmithWaterman(read, genome)
    END IF
END
```

### 3. Graceful Degradation

```
FUNCTION: AlignWithFallback
INPUT: read, genome, bwtIndex
OUTPUT: alignment (possibly suboptimal)

BEGIN
    TRY
        // Attempt optimal alignment
        alignment ← SeedAndExtendAlignment(read, genome, bwtIndex, highQuality = true)
        RETURN alignment
    CATCH OutOfMemoryError AS e DO
        LogWarning("OOM in high-quality mode, falling back to fast mode for read {read.id}")
        alignment ← SeedAndExtendAlignment(read, genome, bwtIndex, highQuality = false)
        alignment.mapq ← MAX(0, alignment.mapq - 10)  // Penalize MAPQ
        RETURN alignment
    CATCH TimeoutError AS e DO
        LogWarning("Alignment timeout for read {read.id}, marking as unmapped")
        RETURN null
    END TRY
END
```

---

## Performance Optimization Strategies

### 1. SIMD Acceleration

```
// Vectorized Smith-Waterman using SSE/AVX instructions
FUNCTION: SmithWatermanSIMD
INPUT: query, reference, scoring
OUTPUT: alignment

BEGIN
    // Process 8 or 16 cells in parallel using SIMD registers
    // Requires specialized implementation in low-level language
    // 4-8× speedup over scalar implementation

    // Pseudocode omitted for brevity
    // See: Farrar, M. (2007) striped Smith-Waterman algorithm
END
```

### 2. GPU Acceleration

```
FUNCTION: AlignBatchGPU
INPUT: readBatch (array of reads), genome, bwtIndex
OUTPUT: alignments (array)

BEGIN
    // Transfer data to GPU
    gpuReads ← CopyToGPU(readBatch)
    gpuGenome ← CopyToGPU(genome)

    // Launch parallel alignment kernels
    gpuAlignments ← LaunchGPUKernel(
        kernel = SmithWatermanKernel,
        numBlocks = length(readBatch),
        threadsPerBlock = 256,
        args = [gpuReads, gpuGenome, scoring]
    )

    // Transfer results back
    alignments ← CopyFromGPU(gpuAlignments)

    RETURN alignments
END
```

---

## Testing Strategy

### 1. Unit Tests

```
TEST: BWTIndexCorrectness
BEGIN
    genome ← "ACGTACGTACGT$"
    index ← BuildBWTIndex(genome)

    ASSERT index.bwtString == "TTTT$GGCCCCAAAA"
    ASSERT BWTExactMatch("ACG", index).length > 0
    ASSERT BWTExactMatch("XYZ", index).length == 0
END

TEST: SmithWatermanBasicAlignment
BEGIN
    query ← "ACGT"
    ref ← "AACGTT"
    alignment ← SmithWaterman(query, ref, defaultScoring)

    ASSERT alignment.score > 0
    ASSERT alignment.cigar == "4M"
    ASSERT alignment.refStart == 1
END
```

### 2. Integration Tests

```
TEST: EndToEndAlignment
BEGIN
    // Create synthetic FASTQ with known alignments
    CreateSyntheticFASTQ("test_reads.fq", numReads = 1000)
    CreateReferenceGenome("test_ref.fa", length = 100000)

    stats ← AlignReads("test_reads.fq", "test_ref.fa", "test_output.bam", defaultConfig)

    ASSERT stats.alignmentRate > 0.95  // Most reads should align
    ASSERT stats.avgAlignmentTime < 0.005  // Less than 5ms per read

    CleanupTestFiles()
END
```

---

## References

1. Ferragina, P., & Manzini, G. (2000). Opportunistic data structures with applications. FOCS.
2. Li, H., & Durbin, R. (2009). Fast and accurate short read alignment with Burrows-Wheeler transform. Bioinformatics.
3. Smith, T. F., & Waterman, M. S. (1981). Identification of common molecular subsequences. JMB.
4. Farrar, M. S. (2007). Striped Smith-Waterman speeds database searches six times over other SIMD implementations. Bioinformatics.

---

**Next Module**: Off-Target Prediction Module
