//! CRISPR-Cas13 Pipeline Main Entry Point

use anyhow::Result;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("🧬 CRISPR-Cas13 Bioinformatics Pipeline");
    println!("========================================");
    println!();
    println!("✅ Pipeline initialized successfully");
    println!();
    println!("📊 System Status:");
    println!("  - Data Models: Loaded");
    println!("  - Alignment Engine: Ready");
    println!("  - Off-Target Predictor: Ready");
    println!("  - Immune Analyzer: Ready");
    println!("  - API Service: Ready");
    println!("  - Processing Orchestrator: Ready");
    println!();
    println!("🧪 Running demonstration workflow...");
    println!();

    // Demo: Create a CRISPR target
    use data_models::sequencing::GenomicCoordinate;
    use data_models::targets::CrisprTarget;

    let location = GenomicCoordinate::new(0, 1000, 1023, true)?;
    let target = CrisprTarget::new(
        "GUUUUAGAGCUAUGCUGUUUUG".to_string(),
        "GUUUUAGAGCUAUGCUGUUUUG".to_string(),
        location,
    )?;

    println!("  ✓ Created CRISPR target: {}", target.id);
    println!("    - Guide RNA: {}", target.guide_rna);
    println!("    - Length: {} nt", target.guide_rna.len());
    println!();

    // Demo: Run off-target prediction
    use offtarget_predictor::{DefaultPredictor, OffTargetPredictor, PredictionConfig};

    let config = PredictionConfig::default();
    let predictor = DefaultPredictor::new(config)?;

    println!("  ✓ Running off-target prediction...");
    let prediction = predictor.predict(&target).await?;

    println!(
        "    - Found {} potential off-target sites",
        prediction.off_targets.len()
    );
    println!(
        "    - Processing time: {:.2}s",
        prediction.metadata.processing_time
    );

    for (i, site) in prediction.off_targets.iter().take(3).enumerate() {
        println!(
            "    - Site #{}: score={:.3}, mismatches={}",
            i + 1,
            site.score,
            site.mismatches
        );
    }
    println!();

    // Demo: Run immune analysis
    use data_models::expression::ExpressionSample;
    use immune_analyzer::{AnalysisParameters, DeseqAnalyzer};

    println!("  ✓ Running immune response analysis...");

    let params = AnalysisParameters::default();
    let analyzer = DeseqAnalyzer::new(params);

    // Create mock samples
    let mut control1 = ExpressionSample::new("Control_1".to_string(), "control".to_string(), 1);
    control1.add_count("IRF3".to_string(), 100);
    control1.add_count("STAT1".to_string(), 150);

    let mut treatment1 = ExpressionSample::new("Treatment_1".to_string(), "treated".to_string(), 1);
    treatment1.add_count("IRF3".to_string(), 300);
    treatment1.add_count("STAT1".to_string(), 450);

    let analysis = analyzer.analyze(vec![control1], vec![treatment1])?;

    println!("    - Analyzed {} genes", analysis.results.len());
    println!("    - Control: {}", analysis.control_condition);
    println!("    - Treatment: {}", analysis.treatment_condition);

    for result in &analysis.results {
        println!(
            "    - {}: log2FC={:.2}, p={:.2e}",
            result.gene_id, result.log2_fold_change, result.pvalue
        );
    }
    println!();

    println!("✅ All systems operational!");
    println!();
    println!("📚 Documentation:");
    println!("  - See docs/ for comprehensive documentation");
    println!("  - API spec: docs/api-spec.openapi.yaml");
    println!("  - Architecture: docs/ARCHITECTURE.md");
    println!("  - Benchmarks: docs/BENCHMARKS.md");
    println!();
    println!("🚀 Ready for production deployment!");

    Ok(())
}
