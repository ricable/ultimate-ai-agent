//! Real-time Performance Dashboard for Neural Swarm
//! 
//! This application provides a real-time dashboard for monitoring
//! the performance of the neural swarm operations.

use std::io::{self, Write};
use std::time::Duration;
use tokio::time::sleep;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use std::collections::HashMap;

// Import the performance monitoring system
use ran::pfs_core::performance::{
    NeuralSwarmPerformanceMonitor, PerformanceDashboard, AlertSeverity,
    visualization,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting Neural Swarm Performance Dashboard...");
    
    // Initialize the performance monitor
    let monitor = NeuralSwarmPerformanceMonitor::new(5, 10000); // 5 second intervals, 10k history
    
    // Start monitoring in background
    monitor.start_monitoring().await;
    
    // Wait a moment for initial data collection
    sleep(Duration::from_secs(2)).await;
    
    // Check if we should run in interactive mode or continuous mode
    let args: Vec<String> = std::env::args().collect();
    let interactive_mode = args.len() > 1 && args[1] == "--interactive";
    
    if interactive_mode {
        run_interactive_dashboard(monitor).await?;
    } else {
        run_continuous_dashboard(monitor).await?;
    }
    
    Ok(())
}

async fn run_interactive_dashboard(monitor: NeuralSwarmPerformanceMonitor) -> Result<(), Box<dyn std::error::Error>> {
    // Setup terminal for interactive mode
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    
    let mut current_view = DashboardView::Overview;
    let mut refresh_interval = Duration::from_secs(5);
    
    loop {
        // Clear screen and display dashboard
        execute!(stdout, crossterm::terminal::Clear(crossterm::terminal::ClearType::All))?;
        execute!(stdout, crossterm::cursor::MoveTo(0, 0))?;
        
        match current_view {
            DashboardView::Overview => {
                display_overview(&monitor).await?;
            },
            DashboardView::Alerts => {
                display_alerts(&monitor).await?;
            },
            DashboardView::Benchmarks => {
                display_benchmarks(&monitor).await?;
            },
            DashboardView::Analytics => {
                display_analytics(&monitor).await?;
            },
        }
        
        // Display navigation help
        println!("\n📋 Navigation:");
        println!("├── [1] Overview  [2] Alerts  [3] Benchmarks  [4] Analytics");
        println!("├── [r] Refresh  [+/-] Change refresh rate  [q] Quit");
        println!("└── Current refresh: {:?}", refresh_interval);
        
        stdout.flush()?;
        
        // Handle user input with timeout
        if event::poll(refresh_interval)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') => break,
                        KeyCode::Char('1') => current_view = DashboardView::Overview,
                        KeyCode::Char('2') => current_view = DashboardView::Alerts,
                        KeyCode::Char('3') => current_view = DashboardView::Benchmarks,
                        KeyCode::Char('4') => current_view = DashboardView::Analytics,
                        KeyCode::Char('r') => {}, // Refresh (just continue loop)
                        KeyCode::Char('+') => {
                            refresh_interval = Duration::from_secs((refresh_interval.as_secs() + 1).min(30));
                        },
                        KeyCode::Char('-') => {
                            refresh_interval = Duration::from_secs((refresh_interval.as_secs().saturating_sub(1)).max(1));
                        },
                        _ => {},
                    }
                }
            }
        }
    }
    
    // Cleanup terminal
    execute!(stdout, LeaveAlternateScreen)?;
    disable_raw_mode()?;
    
    println!("👋 Performance dashboard closed.");
    Ok(())
}

async fn run_continuous_dashboard(monitor: NeuralSwarmPerformanceMonitor) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Running continuous performance monitoring (Ctrl+C to stop)...\n");
    
    let mut iteration = 0;
    
    loop {
        iteration += 1;
        
        println!("🔄 Dashboard Update #{} - {}", iteration, chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"));
        println!("===============================================");
        
        // Display overview
        let dashboard = monitor.get_dashboard().await;
        let dashboard_output = visualization::format_performance_dashboard(&dashboard);
        println!("{}", dashboard_output);
        
        // Display critical alerts
        let critical_alerts = monitor.get_alerts(Some(AlertSeverity::Critical)).await;
        if !critical_alerts.is_empty() {
            println!("🚨 CRITICAL ALERTS:");
            for alert in critical_alerts {
                println!("   • {}: {}", alert.alert_type, alert.message);
                println!("     Action: {}", alert.recommended_action);
            }
            println!();
        }
        
        // Display system health summary
        display_health_summary(&dashboard).await;
        
        println!("⏰ Next update in 30 seconds...\n");
        
        sleep(Duration::from_secs(30)).await;
    }
}

#[derive(Debug, Clone, Copy)]
enum DashboardView {
    Overview,
    Alerts,
    Benchmarks,
    Analytics,
}

async fn display_overview(monitor: &NeuralSwarmPerformanceMonitor) -> Result<(), Box<dyn std::error::Error>> {
    let dashboard = monitor.get_dashboard().await;
    let dashboard_output = visualization::format_performance_dashboard(&dashboard);
    println!("{}", dashboard_output);
    
    // Additional real-time metrics
    println!("⚡ Real-time Metrics:");
    println!("├── Active Agents: {}", dashboard.system_overview.total_agents);
    println!("├── Neural Networks: {}", dashboard.system_overview.active_neural_networks);
    println!("├── Predictions/sec: {:.1}", dashboard.system_overview.total_predictions_per_second);
    println!("└── System Health: {:.1}%", dashboard.system_overview.average_system_health * 100.0);
    
    Ok(())
}

async fn display_alerts(monitor: &NeuralSwarmPerformanceMonitor) -> Result<(), Box<dyn std::error::Error>> {
    println!("⚠️ Performance Alerts Dashboard");
    println!("==============================\n");
    
    // Display alerts by severity
    for severity in [AlertSeverity::Critical, AlertSeverity::High, AlertSeverity::Medium, AlertSeverity::Low] {
        let alerts = monitor.get_alerts(Some(severity)).await;
        if !alerts.is_empty() {
            let severity_name = match severity {
                AlertSeverity::Critical => "🔴 CRITICAL",
                AlertSeverity::High => "🟠 HIGH",
                AlertSeverity::Medium => "🟡 MEDIUM",
                AlertSeverity::Low => "🟢 LOW",
                AlertSeverity::Info => "🔵 INFO",
            };
            
            println!("{} Alerts ({}):", severity_name, alerts.len());
            for alert in alerts {
                println!("├── {}: {}", alert.alert_type, alert.message);
                println!("│   Agent: {} | Threshold: {:.3} | Actual: {:.3}", 
                        alert.agent_id, alert.threshold_value, alert.actual_value);
                println!("│   Action: {}", alert.recommended_action);
                if alert.auto_remediation_available {
                    println!("│   🤖 Auto-remediation available");
                }
                println!("│");
            }
            println!();
        }
    }
    
    Ok(())
}

async fn display_benchmarks(monitor: &NeuralSwarmPerformanceMonitor) -> Result<(), Box<dyn std::error::Error>> {
    println!("🏁 Performance Benchmarks Dashboard");
    println!("===================================\n");
    
    println!("🔄 Running benchmarks...");
    let benchmark_results = monitor.run_benchmarks().await;
    
    let benchmark_output = visualization::format_benchmark_results(&benchmark_results);
    println!("{}", benchmark_output);
    
    // Performance analysis
    println!("📈 Performance Analysis:");
    for (benchmark, time_ms) in &benchmark_results {
        let performance_rating = match time_ms {
            t if *t < 100.0 => "🟢 Excellent",
            t if *t < 500.0 => "🟡 Good",
            t if *t < 1000.0 => "🟠 Average",
            _ => "🔴 Poor",
        };
        println!("├── {}: {}", benchmark, performance_rating);
    }
    
    // Recommendations
    println!("\n💡 Optimization Recommendations:");
    for (benchmark, time_ms) in &benchmark_results {
        if *time_ms > 500.0 {
            println!("├── {}: Consider optimization - current time {:.2}ms", benchmark, time_ms);
        }
    }
    
    Ok(())
}

async fn display_analytics(monitor: &NeuralSwarmPerformanceMonitor) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Performance Analytics Dashboard");
    println!("==================================\n");
    
    // Get historical metrics
    let metrics_24h = monitor.get_metrics_history(24).await;
    let metrics_1h = monitor.get_metrics_history(1).await;
    
    println!("📈 Historical Analysis:");
    println!("├── Data Points (24h): {}", metrics_24h.len());
    println!("├── Data Points (1h): {}", metrics_1h.len());
    
    if !metrics_24h.is_empty() {
        // Calculate averages and trends
        let avg_accuracy_24h = metrics_24h.iter()
            .map(|m| m.prediction_accuracy)
            .sum::<f64>() / metrics_24h.len() as f64;
        
        let avg_accuracy_1h = if !metrics_1h.is_empty() {
            metrics_1h.iter()
                .map(|m| m.prediction_accuracy)
                .sum::<f64>() / metrics_1h.len() as f64
        } else {
            avg_accuracy_24h
        };
        
        println!("├── Avg Accuracy (24h): {:.2}%", avg_accuracy_24h * 100.0);
        println!("├── Avg Accuracy (1h): {:.2}%", avg_accuracy_1h * 100.0);
        
        let trend = if avg_accuracy_1h > avg_accuracy_24h {
            "🟢 Improving"
        } else if avg_accuracy_1h < avg_accuracy_24h {
            "🔴 Declining"
        } else {
            "🟡 Stable"
        };
        println!("└── Accuracy Trend: {}", trend);
        
        // Resource utilization analysis
        println!("\n💾 Resource Utilization Analysis:");
        let avg_cpu_24h = metrics_24h.iter()
            .map(|m| m.cpu_utilization)
            .sum::<f64>() / metrics_24h.len() as f64;
        
        let avg_memory_24h = metrics_24h.iter()
            .map(|m| m.memory_usage_mb)
            .sum::<f64>() / metrics_24h.len() as f64;
        
        println!("├── Avg CPU Usage (24h): {:.1}%", avg_cpu_24h);
        println!("├── Avg Memory Usage (24h): {:.1}MB", avg_memory_24h);
        
        // Performance correlation analysis
        println!("\n🔗 Performance Correlations:");
        let high_accuracy_metrics: Vec<_> = metrics_24h.iter()
            .filter(|m| m.prediction_accuracy > 0.9)
            .collect();
        
        if !high_accuracy_metrics.is_empty() {
            let avg_inference_time_high_acc = high_accuracy_metrics.iter()
                .map(|m| m.inference_time_ms)
                .sum::<f64>() / high_accuracy_metrics.len() as f64;
            
            println!("├── Avg inference time at >90% accuracy: {:.1}ms", avg_inference_time_high_acc);
        }
        
        // Optimization effectiveness
        let avg_optimization_effect = metrics_24h.iter()
            .map(|m| m.network_optimization_effectiveness)
            .sum::<f64>() / metrics_24h.len() as f64;
        
        println!("└── Network Optimization Effectiveness: {:.1}%", avg_optimization_effect * 100.0);
    }
    
    Ok(())
}

async fn display_health_summary(dashboard: &PerformanceDashboard) {
    println!("🏥 System Health Summary:");
    
    let health_score = dashboard.system_overview.average_system_health;
    let health_status = match health_score {
        h if h >= 0.95 => "🟢 Excellent",
        h if h >= 0.85 => "🟡 Good",
        h if h >= 0.70 => "🟠 Fair",
        _ => "🔴 Poor",
    };
    
    println!("├── Overall Health: {} ({:.1}%)", health_status, health_score * 100.0);
    
    // CPU status
    let cpu_status = match dashboard.system_overview.total_cpu_usage_percentage {
        cpu if cpu < 70.0 => "🟢 Normal",
        cpu if cpu < 85.0 => "🟡 Elevated",
        _ => "🔴 High",
    };
    println!("├── CPU Status: {} ({:.1}%)", cpu_status, dashboard.system_overview.total_cpu_usage_percentage);
    
    // Memory status
    let memory_gb = dashboard.system_overview.total_memory_usage_gb;
    let memory_status = match memory_gb {
        mem if mem < 8.0 => "🟢 Normal",
        mem if mem < 12.0 => "🟡 Elevated",
        _ => "🔴 High",
    };
    println!("├── Memory Status: {} ({:.1}GB)", memory_status, memory_gb);
    
    // Neural network performance
    let nn_accuracy = dashboard.neural_network_stats.ensemble_accuracy;
    let nn_status = match nn_accuracy {
        acc if acc >= 0.9 => "🟢 Excellent",
        acc if acc >= 0.8 => "🟡 Good",
        acc if acc >= 0.7 => "🟠 Fair",
        _ => "🔴 Poor",
    };
    println!("├── Neural Network Status: {} ({:.1}%)", nn_status, nn_accuracy * 100.0);
    
    // Alert summary
    let alert_count = dashboard.recent_alerts.len();
    let critical_alerts = dashboard.recent_alerts.iter()
        .filter(|a| a.severity == AlertSeverity::Critical)
        .count();
    
    if critical_alerts > 0 {
        println!("└── Alerts: 🔴 {} critical, {} total", critical_alerts, alert_count);
    } else if alert_count > 0 {
        println!("└── Alerts: 🟡 {} total (no critical)", alert_count);
    } else {
        println!("└── Alerts: 🟢 No active alerts");
    }
}

/// Export performance data to file
async fn export_performance_data(monitor: &NeuralSwarmPerformanceMonitor, format: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("📤 Exporting performance data in {} format...", format);
    
    let data = monitor.export_performance_data(format).await?;
    let filename = format!("performance_data_{}.{}", 
                          chrono::Utc::now().format("%Y%m%d_%H%M%S"), 
                          format);
    
    std::fs::write(&filename, data)?;
    println!("✅ Data exported to: {}", filename);
    
    Ok(())
}

/// Display help information
fn display_help() {
    println!("🐝 Neural Swarm Performance Dashboard");
    println!("====================================\n");
    println!("Usage:");
    println!("  performance_dashboard                Run in continuous mode");
    println!("  performance_dashboard --interactive  Run in interactive mode");
    println!("  performance_dashboard --help         Show this help\n");
    println!("Interactive Mode Controls:");
    println!("  1-4: Switch between dashboard views");
    println!("  r:   Refresh display");
    println!("  +/-: Adjust refresh rate");
    println!("  q:   Quit\n");
    println!("Dashboard Views:");
    println!("  1. Overview    - System overview and neural network stats");
    println!("  2. Alerts      - Performance alerts and recommendations");
    println!("  3. Benchmarks  - Performance benchmarks and analysis");
    println!("  4. Analytics   - Historical analysis and trends");
}