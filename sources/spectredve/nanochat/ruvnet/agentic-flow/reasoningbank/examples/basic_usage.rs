//! Basic usage example for ReasoningBank

use reasoningbank_core::{Pattern, TaskOutcome, ReasoningEngine};
use reasoningbank_learning::{AdaptiveLearner, LearningConfig};
use reasoningbank_storage::{SqliteStorage, StorageConfig};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    println!("🧠 ReasoningBank - Basic Usage Example\n");

    // 1. Setup storage
    let storage_config = StorageConfig {
        database_path: PathBuf::from("reasoningbank_example.db"),
        ..Default::default()
    };
    let storage = SqliteStorage::new(storage_config)?;

    // 2. Create reasoning engine
    let engine = ReasoningEngine::default();

    // 3. Create adaptive learner
    let mut learner = AdaptiveLearner::new(
        engine,
        storage.clone(),
        LearningConfig::default(),
    );

    // 4. Store some patterns
    println!("📝 Storing reasoning patterns...\n");

    let patterns = vec![
        (
            "Implement JWT authentication for REST API",
            "api-development",
            "token-based-auth",
            0.95,
            10.0,
        ),
        (
            "Build GraphQL API with authentication",
            "api-development",
            "graphql-auth",
            0.88,
            12.0,
        ),
        (
            "Create microservice with OAuth2",
            "api-development",
            "oauth2-flow",
            0.92,
            15.0,
        ),
        (
            "Design database schema for e-commerce",
            "database-design",
            "relational-model",
            0.90,
            8.0,
        ),
    ];

    for (desc, category, strategy, score, duration) in patterns {
        let pattern = Pattern::new(
            desc.to_string(),
            category.to_string(),
            strategy.to_string(),
        ).with_outcome(TaskOutcome::partial(score, duration));

        let insight = learner.learn_from_task(&pattern)?;

        println!("✓ Stored: {}", desc);
        println!("  Similar patterns found: {}", insight.similar_patterns_found);
        println!("  Improvement potential: {:.1}%", insight.improvement_potential * 100.0);
        if !insight.suggested_optimizations.is_empty() {
            println!("  Suggestion: {}", insight.suggested_optimizations[0]);
        }
        println!();
    }

    // 5. Apply learning to a new task
    println!("🔍 Applying learning to new task...\n");

    let applied = learner.apply_learning(
        "Build secure REST API with authentication",
        "api-development",
    )?;

    println!("📊 Recommendation:");
    println!("  Strategy: {:?}", applied.recommended_strategy);
    println!("  Confidence: {:.1}%", applied.confidence * 100.0);
    println!("  Expected success: {:.1}%", applied.expected_success * 100.0);
    println!("  Based on {} similar patterns", applied.similar_patterns_count);
    println!();

    // 6. Get learning statistics
    let stats = learner.get_learning_stats()?;

    println!("📈 Learning Statistics:");
    println!("  Total patterns: {}", stats.total_patterns);
    println!("  Categories: {}", stats.total_categories);
    println!("  Successful tasks: {}", stats.total_successes);
    println!("  Average success rate: {:.1}%", stats.avg_success_score * 100.0);
    println!();

    // 7. Optimize strategies
    use reasoningbank_learning::StrategyOptimizer;

    let optimizer = StrategyOptimizer::new(storage);
    let optimization = optimizer.optimize_for_category("api-development")?;

    println!("🎯 Strategy Optimization for 'api-development':");
    println!("  Total patterns analyzed: {}", optimization.total_patterns);
    println!("  Recommended: {:?}", optimization.recommended_strategy);
    println!("\n  Rankings:");

    for (i, ranking) in optimization.strategy_rankings.iter().enumerate() {
        println!("  {}. {} (score: {:.3})", i + 1, ranking.strategy, ranking.composite_score);
        println!("     Success rate: {:.1}%, Avg duration: {:.1}s",
                 ranking.success_rate * 100.0, ranking.avg_duration_seconds);
    }

    println!("\n✅ Example complete!");
    println!("💾 Database saved to: reasoningbank_example.db");

    Ok(())
}
