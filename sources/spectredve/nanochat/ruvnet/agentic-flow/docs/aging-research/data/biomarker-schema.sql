-- Comprehensive Aging Biomarker Database Schema
-- Date: 2025-11-08
-- Purpose: Clinical and research tracking of aging biomarkers

-- Main biomarkers table
CREATE TABLE biomarkers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL, -- 'epigenetic', 'metabolic', 'proteomic', 'cellular', 'inflammatory', 'cardiovascular', 'hormonal', 'genetic'
    subcategory TEXT, -- More specific classification
    measurement_method TEXT NOT NULL,
    specimen_type TEXT NOT NULL, -- 'blood', 'urine', 'saliva', 'tissue', 'imaging'

    -- Age-stratified reference ranges (JSON format for flexibility)
    normal_range_20s TEXT, -- JSON: {"min": x, "max": y, "unit": "unit"}
    normal_range_40s TEXT,
    normal_range_60s TEXT,
    normal_range_80s TEXT,

    -- Clinical validation and utility
    clinical_validation_level TEXT, -- 'FDA-approved', 'clinically-validated', 'research-use', 'experimental'
    evidence_grade TEXT, -- 'A' (strong), 'B' (moderate), 'C' (weak), 'D' (very weak)
    sensitivity REAL, -- For diagnostic tests (0.0-1.0)
    specificity REAL, -- For diagnostic tests (0.0-1.0)

    -- Predictive power
    predictive_power REAL, -- Correlation with mortality/morbidity (0.0-1.0)
    hazard_ratio REAL, -- If applicable from clinical studies
    hr_confidence_interval TEXT, -- e.g., "0.85-0.95"

    -- Clinical utility
    cost_per_test REAL, -- USD
    turnaround_time_days INTEGER,
    insurance_coverage TEXT, -- 'typically-covered', 'sometimes-covered', 'rarely-covered', 'not-covered'

    -- Associations and interpretation
    direction_with_age TEXT, -- 'increases', 'decreases', 'variable', 'stable'
    pathological_threshold TEXT, -- Value indicating disease risk
    optimal_range TEXT, -- Target range for healthy aging

    -- Metadata
    citations TEXT, -- JSON array of PMIDs or DOIs
    first_discovered INTEGER, -- Year
    last_updated DATE DEFAULT CURRENT_DATE,
    notes TEXT
);

-- Age-related diseases table
CREATE TABLE diseases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    category TEXT, -- 'cardiovascular', 'neurodegenerative', 'metabolic', 'cancer', 'musculoskeletal'
    icd10_code TEXT,
    description TEXT
);

-- Biomarker-disease associations
CREATE TABLE biomarker_disease_associations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    biomarker_id INTEGER NOT NULL,
    disease_id INTEGER NOT NULL,
    association_strength TEXT, -- 'strong', 'moderate', 'weak'
    odds_ratio REAL,
    or_confidence_interval TEXT,
    evidence_level TEXT, -- 'Level 1a', 'Level 1b', 'Level 2', 'Level 3'
    study_population_size INTEGER,
    citations TEXT, -- JSON array
    FOREIGN KEY (biomarker_id) REFERENCES biomarkers(id),
    FOREIGN KEY (disease_id) REFERENCES diseases(id)
);

-- Interventions table
CREATE TABLE interventions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    type TEXT, -- 'pharmacological', 'lifestyle', 'supplement', 'procedure'
    description TEXT,
    mechanism TEXT,
    fda_status TEXT, -- 'approved', 'investigational', 'off-label', 'supplement'
    evidence_grade TEXT
);

-- Biomarker-intervention effects
CREATE TABLE biomarker_interventions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    biomarker_id INTEGER NOT NULL,
    intervention_id INTEGER NOT NULL,
    effect_direction TEXT, -- 'improves', 'worsens', 'normalizes', 'no-effect'
    effect_magnitude REAL, -- Percentage change
    effect_magnitude_unit TEXT,
    time_to_effect_days INTEGER,
    evidence_level TEXT,
    study_population_size INTEGER,
    citations TEXT,
    FOREIGN KEY (biomarker_id) REFERENCES biomarkers(id),
    FOREIGN KEY (intervention_id) REFERENCES interventions(id)
);

-- Clinical panels (groups of biomarkers commonly tested together)
CREATE TABLE clinical_panels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    clinical_indication TEXT,
    cost_estimate REAL,
    turnaround_time_days INTEGER
);

CREATE TABLE panel_biomarkers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    panel_id INTEGER NOT NULL,
    biomarker_id INTEGER NOT NULL,
    FOREIGN KEY (panel_id) REFERENCES clinical_panels(id),
    FOREIGN KEY (biomarker_id) REFERENCES biomarkers(id)
);

-- Patient measurements (for longitudinal tracking)
CREATE TABLE patient_measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL, -- Anonymized patient identifier
    biomarker_id INTEGER NOT NULL,
    measurement_date DATE NOT NULL,
    value REAL NOT NULL,
    unit TEXT,
    lab_name TEXT,
    notes TEXT,
    FOREIGN KEY (biomarker_id) REFERENCES biomarkers(id)
);

-- Indexes for performance
CREATE INDEX idx_biomarker_category ON biomarkers(category);
CREATE INDEX idx_biomarker_validation ON biomarkers(clinical_validation_level);
CREATE INDEX idx_biomarker_evidence ON biomarkers(evidence_grade);
CREATE INDEX idx_disease_category ON diseases(category);
CREATE INDEX idx_patient_measurements_date ON patient_measurements(measurement_date);
CREATE INDEX idx_patient_measurements_biomarker ON patient_measurements(biomarker_id, patient_id);

-- Views for common queries
CREATE VIEW biomarkers_fda_approved AS
SELECT * FROM biomarkers
WHERE clinical_validation_level = 'FDA-approved'
ORDER BY category, name;

CREATE VIEW high_predictive_biomarkers AS
SELECT * FROM biomarkers
WHERE predictive_power >= 0.7 AND evidence_grade IN ('A', 'B')
ORDER BY predictive_power DESC;

CREATE VIEW affordable_biomarkers AS
SELECT * FROM biomarkers
WHERE cost_per_test < 200 AND insurance_coverage IN ('typically-covered', 'sometimes-covered')
ORDER BY cost_per_test;
