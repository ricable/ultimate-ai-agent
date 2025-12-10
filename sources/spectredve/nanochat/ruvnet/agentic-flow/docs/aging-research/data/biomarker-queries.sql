-- Comprehensive Aging Biomarker Query Examples
-- Date: 2025-11-08
-- Purpose: Practical SQL queries for clinical and research applications

-- ====================
-- CLINICAL SCREENING QUERIES
-- ====================

-- 1. Find affordable, insurance-covered biomarkers for routine aging assessment
SELECT name, category, cost_per_test, insurance_coverage, clinical_validation_level
FROM biomarkers
WHERE cost_per_test < 100
  AND insurance_coverage IN ('typically-covered', 'sometimes-covered')
  AND clinical_validation_level IN ('FDA-approved', 'clinically-validated')
ORDER BY cost_per_test;

-- 2. High-predictive power biomarkers with strong evidence
SELECT name, category, predictive_power, evidence_grade, cost_per_test
FROM biomarkers
WHERE predictive_power >= 0.75
  AND evidence_grade = 'A'
ORDER BY predictive_power DESC;

-- 3. Identify biomarkers that worsen with age (risk factors)
SELECT name, category, direction_with_age, pathological_threshold, predictive_power
FROM biomarkers
WHERE direction_with_age = 'increases'
  AND pathological_threshold IS NOT NULL
ORDER BY predictive_power DESC;

-- 4. Find comprehensive panels by category
SELECT category, COUNT(*) as biomarker_count,
       AVG(cost_per_test) as avg_cost,
       AVG(predictive_power) as avg_predictive_power
FROM biomarkers
WHERE clinical_validation_level IN ('FDA-approved', 'clinically-validated')
GROUP BY category
ORDER BY biomarker_count DESC;

-- ====================
-- COST OPTIMIZATION QUERIES
-- ====================

-- 5. Build cost-effective comprehensive aging panel (<$500 total)
SELECT name, category, cost_per_test, predictive_power, insurance_coverage
FROM biomarkers
WHERE cost_per_test < 100
  AND predictive_power > 0.70
  AND clinical_validation_level IN ('FDA-approved', 'clinically-validated')
ORDER BY predictive_power DESC
LIMIT 10;

-- 6. Compare cost vs. predictive power across categories
SELECT category,
       MIN(cost_per_test) as min_cost,
       MAX(cost_per_test) as max_cost,
       AVG(cost_per_test) as avg_cost,
       AVG(predictive_power) as avg_predictive_power
FROM biomarkers
WHERE cost_per_test IS NOT NULL
GROUP BY category
ORDER BY avg_predictive_power DESC;

-- 7. Best value biomarkers (high predictive power per dollar)
SELECT name, category, cost_per_test, predictive_power,
       ROUND(predictive_power / cost_per_test * 1000, 2) as value_score
FROM biomarkers
WHERE cost_per_test > 0 AND predictive_power IS NOT NULL
ORDER BY value_score DESC
LIMIT 20;

-- ====================
-- DISEASE ASSOCIATION QUERIES
-- ====================

-- 8. Find biomarkers strongly associated with cardiovascular disease
SELECT b.name, b.category, b.predictive_power, bda.association_strength, bda.odds_ratio
FROM biomarkers b
JOIN biomarker_disease_associations bda ON b.id = bda.biomarker_id
JOIN diseases d ON bda.disease_id = d.id
WHERE d.category = 'cardiovascular'
  AND bda.association_strength IN ('strong', 'moderate')
ORDER BY bda.odds_ratio DESC;

-- 9. Identify biomarkers associated with multiple diseases (pleiotropic)
SELECT b.name, b.category, COUNT(DISTINCT bda.disease_id) as disease_count
FROM biomarkers b
JOIN biomarker_disease_associations bda ON b.id = bda.biomarker_id
GROUP BY b.id, b.name, b.category
HAVING COUNT(DISTINCT bda.disease_id) >= 3
ORDER BY disease_count DESC;

-- ====================
-- INTERVENTION EFFECTIVENESS QUERIES
-- ====================

-- 10. Find interventions that improve multiple biomarkers
SELECT i.name, i.type, COUNT(*) as biomarkers_affected,
       AVG(bi.effect_magnitude) as avg_improvement
FROM interventions i
JOIN biomarker_interventions bi ON i.id = bi.intervention_id
WHERE bi.effect_direction IN ('improves', 'normalizes')
GROUP BY i.id, i.name, i.type
HAVING COUNT(*) >= 3
ORDER BY biomarkers_affected DESC;

-- 11. Rapidly modifiable biomarkers (for intervention tracking)
SELECT b.name, b.category, bi.intervention_id, bi.time_to_effect_days, bi.effect_magnitude
FROM biomarkers b
JOIN biomarker_interventions bi ON b.id = bi.biomarker_id
WHERE bi.time_to_effect_days <= 90
  AND bi.effect_direction IN ('improves', 'normalizes')
ORDER BY bi.time_to_effect_days;

-- 12. Lifestyle interventions with strong evidence
SELECT i.name as intervention, b.name as biomarker,
       bi.effect_direction, bi.effect_magnitude, bi.evidence_level
FROM interventions i
JOIN biomarker_interventions bi ON i.id = bi.intervention_id
JOIN biomarkers b ON bi.biomarker_id = b.id
WHERE i.type = 'lifestyle'
  AND bi.evidence_level IN ('Level 1a', 'Level 1b')
ORDER BY i.name, bi.effect_magnitude DESC;

-- ====================
-- LONGITUDINAL TRACKING QUERIES
-- ====================

-- 13. Track individual patient biomarker trends over time
SELECT pm.patient_id, b.name, pm.measurement_date, pm.value, pm.unit,
       CASE
         WHEN b.direction_with_age = 'increases' AND pm.value > CAST(json_extract(b.pathological_threshold, '$') AS REAL) THEN 'High Risk'
         WHEN b.direction_with_age = 'decreases' AND pm.value < CAST(json_extract(b.pathological_threshold, '$') AS REAL) THEN 'High Risk'
         ELSE 'Normal Range'
       END as risk_status
FROM patient_measurements pm
JOIN biomarkers b ON pm.biomarker_id = b.id
WHERE pm.patient_id = 'PATIENT_001'
ORDER BY b.name, pm.measurement_date;

-- 14. Calculate biomarker change rate for individual patient
SELECT b.name, b.category,
       MIN(pm.measurement_date) as first_date,
       MAX(pm.measurement_date) as last_date,
       COUNT(*) as measurement_count,
       ROUND(AVG(pm.value), 2) as mean_value,
       ROUND(
         (MAX(pm.value) - MIN(pm.value)) /
         (julianday(MAX(pm.measurement_date)) - julianday(MIN(pm.measurement_date))) * 365,
         2
       ) as annual_change_rate
FROM patient_measurements pm
JOIN biomarkers b ON pm.biomarker_id = b.id
WHERE pm.patient_id = 'PATIENT_001'
GROUP BY b.id, b.name, b.category
HAVING COUNT(*) >= 3;

-- 15. Identify patients with accelerated biological aging
SELECT pm.patient_id,
       COUNT(DISTINCT CASE WHEN pm.value > CAST(json_extract(b.pathological_threshold, '$') AS REAL)
                           AND b.direction_with_age = 'increases' THEN b.id END) as elevated_markers,
       COUNT(DISTINCT CASE WHEN pm.value < CAST(json_extract(b.pathological_threshold, '$') AS REAL)
                           AND b.direction_with_age = 'decreases' THEN b.id END) as depleted_markers
FROM patient_measurements pm
JOIN biomarkers b ON pm.biomarker_id = b.id
WHERE b.pathological_threshold IS NOT NULL
GROUP BY pm.patient_id
HAVING elevated_markers + depleted_markers >= 5;

-- ====================
-- CLINICAL PANEL CREATION
-- ====================

-- 16. Create cardiovascular aging panel
INSERT INTO clinical_panels (name, description, clinical_indication, cost_estimate, turnaround_time_days)
VALUES (
  'Cardiovascular Aging Panel',
  'Comprehensive assessment of vascular health, cardiac function, and CVD risk',
  'Adults >40 years, cardiovascular disease screening',
  750,
  3
);

-- Add biomarkers to cardiovascular panel (run after above INSERT)
INSERT INTO panel_biomarkers (panel_id, biomarker_id)
SELECT
  (SELECT id FROM clinical_panels WHERE name = 'Cardiovascular Aging Panel'),
  id
FROM biomarkers
WHERE name IN (
  'Pulse Wave Velocity (PWV)',
  'VO2 Max',
  'Heart Rate Variability (HRV)',
  'Carotid Intima-Media Thickness (cIMT)',
  'NT-proBNP (N-terminal pro-B-type Natriuretic Peptide)',
  'Troponin I (High-Sensitivity)',
  'CRP (C-Reactive Protein)',
  'LDL Cholesterol',
  'ApoB (Apolipoprotein B)',
  'Lipoprotein(a) [Lp(a)]'
);

-- 17. Create inflammaging panel
INSERT INTO clinical_panels (name, description, clinical_indication, cost_estimate, turnaround_time_days)
VALUES (
  'Inflammaging Panel',
  'Assessment of chronic inflammation and immune senescence',
  'Adults >50 years, chronic disease risk assessment',
  550,
  5
);

INSERT INTO panel_biomarkers (panel_id, biomarker_id)
SELECT
  (SELECT id FROM clinical_panels WHERE name = 'Inflammaging Panel'),
  id
FROM biomarkers
WHERE name IN (
  'IL-6 (Interleukin-6)',
  'CRP (C-Reactive Protein)',
  'TNF-α (Tumor Necrosis Factor-alpha)',
  'IL-1β (Interleukin-1 beta)',
  'IL-8 (Interleukin-8)',
  'Neutrophil-to-Lymphocyte Ratio (NLR)',
  'CD4+/CD8+ T-cell Ratio',
  'GDF15 (Growth Differentiation Factor 15)'
);

-- 18. Create epigenetic aging panel
INSERT INTO clinical_panels (name, description, clinical_indication, cost_estimate, turnaround_time_days)
VALUES (
  'Epigenetic Aging Panel',
  'DNA methylation-based biological age assessment',
  'Longevity optimization, biological age determination',
  900,
  14
);

INSERT INTO panel_biomarkers (panel_id, biomarker_id)
SELECT
  (SELECT id FROM clinical_panels WHERE name = 'Epigenetic Aging Panel'),
  id
FROM biomarkers
WHERE name IN (
  'Horvath DNAm Clock',
  'GrimAge Clock',
  'PhenoAge Clock',
  'Hannum DNAm Clock',
  'Telomere Length'
);

-- ====================
-- RESEARCH QUERIES
-- ====================

-- 19. Compare epigenetic vs. chronological age
SELECT b.name, pm.patient_id,
       AVG(pm.value) as biological_age,
       (SELECT AVG(value) FROM patient_measurements
        WHERE patient_id = pm.patient_id AND biomarker_id = b.id) as avg_bio_age
FROM biomarkers b
JOIN patient_measurements pm ON b.id = pm.biomarker_id
WHERE b.subcategory = 'methylation_clock'
GROUP BY b.name, pm.patient_id;

-- 20. Identify biomarker clusters (correlations)
SELECT b1.name as biomarker1, b2.name as biomarker2,
       b1.category as category1, b2.category as category2
FROM biomarkers b1, biomarkers b2
WHERE b1.id < b2.id
  AND b1.direction_with_age = b2.direction_with_age
  AND b1.category != b2.category
  AND ABS(b1.predictive_power - b2.predictive_power) < 0.1
ORDER BY b1.predictive_power DESC;

-- 21. Emerging vs. established biomarkers
SELECT
  CASE
    WHEN clinical_validation_level = 'FDA-approved' THEN 'Established'
    WHEN clinical_validation_level = 'clinically-validated' THEN 'Clinical'
    ELSE 'Emerging'
  END as maturity_level,
  COUNT(*) as biomarker_count,
  AVG(predictive_power) as avg_predictive_power,
  AVG(cost_per_test) as avg_cost
FROM biomarkers
GROUP BY maturity_level
ORDER BY avg_predictive_power DESC;

-- ====================
-- PRACTICAL CLINICAL SCENARIOS
-- ====================

-- 22. Basic aging panel for 40-year-old (preventive screening)
SELECT name, category, cost_per_test,
       json_extract(normal_range_40s, '$.min') as min_value,
       json_extract(normal_range_40s, '$.max') as max_value,
       json_extract(normal_range_40s, '$.unit') as unit
FROM biomarkers
WHERE cost_per_test < 75
  AND insurance_coverage = 'typically-covered'
  AND normal_range_40s IS NOT NULL
ORDER BY category, name;

-- 23. Comprehensive aging panel for 60-year-old
SELECT name, category, cost_per_test, predictive_power,
       json_extract(normal_range_60s, '$.min') as min_value,
       json_extract(normal_range_60s, '$.max') as max_value
FROM biomarkers
WHERE predictive_power > 0.70
  AND clinical_validation_level IN ('FDA-approved', 'clinically-validated')
  AND normal_range_60s IS NOT NULL
ORDER BY predictive_power DESC;

-- 24. Urgent biomarkers for symptomatic elderly patient
SELECT name, category, turnaround_time_days, cost_per_test, pathological_threshold
FROM biomarkers
WHERE turnaround_time_days <= 1
  AND clinical_validation_level = 'FDA-approved'
  AND pathological_threshold IS NOT NULL
ORDER BY turnaround_time_days, predictive_power DESC;

-- 25. Longevity optimization panel (research-grade)
SELECT name, category, cost_per_test, predictive_power, measurement_method
FROM biomarkers
WHERE predictive_power >= 0.80
  OR subcategory IN ('methylation_clock', 'longevity_gene', 'longevity_factor')
ORDER BY predictive_power DESC;

-- ====================
-- SUMMARY STATISTICS
-- ====================

-- 26. Overall database statistics
SELECT
  COUNT(*) as total_biomarkers,
  COUNT(DISTINCT category) as categories,
  AVG(cost_per_test) as avg_cost,
  AVG(predictive_power) as avg_predictive_power,
  SUM(CASE WHEN insurance_coverage = 'typically-covered' THEN 1 ELSE 0 END) as covered_by_insurance,
  SUM(CASE WHEN clinical_validation_level = 'FDA-approved' THEN 1 ELSE 0 END) as fda_approved
FROM biomarkers;

-- 27. Biomarker accessibility matrix
SELECT
  clinical_validation_level,
  insurance_coverage,
  COUNT(*) as count,
  AVG(cost_per_test) as avg_cost,
  AVG(predictive_power) as avg_predictive_power
FROM biomarkers
WHERE cost_per_test IS NOT NULL
GROUP BY clinical_validation_level, insurance_coverage
ORDER BY clinical_validation_level, insurance_coverage;

-- 28. Most actionable biomarkers (modifiable + high predictive power)
SELECT b.name, b.category, b.predictive_power,
       COUNT(DISTINCT bi.intervention_id) as intervention_count
FROM biomarkers b
JOIN biomarker_interventions bi ON b.id = bi.biomarker_id
WHERE b.predictive_power > 0.70
  AND bi.effect_direction IN ('improves', 'normalizes')
GROUP BY b.id, b.name, b.category, b.predictive_power
HAVING intervention_count >= 2
ORDER BY b.predictive_power DESC, intervention_count DESC;
