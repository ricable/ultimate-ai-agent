//! Visualization Utilities
//! 
//! This module provides text-based visualization for results and metrics.

use crate::models::{OptimizationSummary, AgentPerformance};
use std::fmt::Write;

pub struct ResultVisualizer;

impl ResultVisualizer {
    pub fn create_convergence_plot(convergence_history: &[f32], width: usize) -> String {
        if convergence_history.is_empty() {
            return "No convergence data available".to_string();
        }
        
        let mut output = String::new();
        let height = 10;
        
        let min_fitness = convergence_history.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_fitness = convergence_history.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_fitness - min_fitness;
        
        if range == 0.0 {
            return "Convergence data has no variation".to_string();
        }
        
        writeln!(output, "Convergence Plot (Fitness over Time)").unwrap();
        writeln!(output, "Max: {:.4} ┬", max_fitness).unwrap();
        
        for row in 0..height {
            write!(output, "         ").unwrap();
            
            let threshold = max_fitness - (row as f32 / (height - 1) as f32) * range;
            
            for (i, &fitness) in convergence_history.iter().enumerate() {
                if i * width / convergence_history.len() < width {
                    let symbol = if fitness >= threshold { "█" } else { " " };
                    write!(output, "{}", symbol).unwrap();
                }
            }
            
            if row == height - 1 {
                writeln!(output, " ┤ Min: {:.4}", min_fitness).unwrap();
            } else {
                writeln!(output, " │").unwrap();
            }
        }
        
        write!(output, "         ").unwrap();
        for _ in 0..width {
            write!(output, "─").unwrap();
        }
        writeln!(output, " ┴").unwrap();
        writeln!(output, "         0{:>width$}", 
                convergence_history.len(), width = width.saturating_sub(1)).unwrap();
        
        output
    }
    
    pub fn create_agent_performance_chart(performances: &[AgentPerformance]) -> String {
        if performances.is_empty() {
            return "No agent performance data available".to_string();
        }
        
        let mut output = String::new();
        writeln!(output, "Agent Performance Summary").unwrap();
        writeln!(output, "========================").unwrap();
        
        let max_fitness = performances.iter()
            .map(|p| p.personal_best_fitness)
            .fold(f32::NEG_INFINITY, |a, b| a.max(b));
        
        for performance in performances {
            let normalized = if max_fitness > 0.0 {
                (performance.personal_best_fitness / max_fitness * 20.0) as usize
            } else {
                0
            };
            
            write!(output, "{:12} ", performance.agent_id).unwrap();
            write!(output, "│").unwrap();
            
            for _ in 0..normalized {
                write!(output, "█").unwrap();
            }
            
            writeln!(output, " {:.4}", performance.personal_best_fitness).unwrap();
        }
        
        output
    }
    
    pub fn create_summary_report(summary: &OptimizationSummary) -> String {
        let mut output = String::new();
        
        writeln!(output, "🚀 NEURAL SWARM OPTIMIZATION SUMMARY").unwrap();
        writeln!(output, "====================================").unwrap();
        writeln!(output).unwrap();
        
        writeln!(output, "📊 OPTIMIZATION RESULTS").unwrap();
        writeln!(output, "  ├─ Timestamp: {}", summary.timestamp.format("%Y-%m-%d %H:%M:%S UTC")).unwrap();
        writeln!(output, "  ├─ Total Iterations: {}", summary.total_iterations).unwrap();
        writeln!(output, "  ├─ Execution Time: {:.2}s", summary.execution_time_seconds).unwrap();
        writeln!(output, "  └─ Best Fitness: {:.6}", summary.best_fitness).unwrap();
        writeln!(output).unwrap();
        
        writeln!(output, "📡 OPTIMAL RAN CONFIGURATION").unwrap();
        writeln!(output, "  ├─ Cell ID: {}", summary.best_configuration.cell_id).unwrap();
        writeln!(output, "  ├─ Power Level: {:.1} dBm", summary.best_configuration.power_level).unwrap();
        writeln!(output, "  ├─ Antenna Tilt: {:.1}°", summary.best_configuration.antenna_tilt).unwrap();
        writeln!(output, "  ├─ Bandwidth: {:.0} MHz", summary.best_configuration.bandwidth).unwrap();
        writeln!(output, "  ├─ Frequency: {:.0} MHz", summary.best_configuration.frequency_band).unwrap();
        writeln!(output, "  ├─ Modulation: {}", summary.best_configuration.modulation_scheme).unwrap();
        writeln!(output, "  ├─ MIMO: {}", summary.best_configuration.mimo_config).unwrap();
        writeln!(output, "  └─ Beamforming: {}", if summary.best_configuration.beamforming_enabled { "Enabled" } else { "Disabled" }).unwrap();
        writeln!(output).unwrap();
        
        writeln!(output, "📈 PERFORMANCE METRICS").unwrap();
        writeln!(output, "  ├─ Throughput: {:.2} Mbps", summary.best_metrics.throughput).unwrap();
        writeln!(output, "  ├─ Latency: {:.2} ms", summary.best_metrics.latency).unwrap();
        writeln!(output, "  ├─ Energy Efficiency: {:.3}", summary.best_metrics.energy_efficiency).unwrap();
        writeln!(output, "  └─ Interference Level: {:.3}", summary.best_metrics.interference_level).unwrap();
        writeln!(output).unwrap();
        
        if !summary.agent_performances.is_empty() {
            writeln!(output, "🤖 AGENT PERFORMANCE").unwrap();
            let chart = Self::create_agent_performance_chart(&summary.agent_performances);
            writeln!(output, "{}", chart).unwrap();
        }
        
        if !summary.convergence_history.is_empty() {
            writeln!(output, "📊 CONVERGENCE ANALYSIS").unwrap();
            let plot = Self::create_convergence_plot(&summary.convergence_history, 60);
            writeln!(output, "{}", plot).unwrap();
        }
        
        output
    }
    
    pub fn create_progress_bar(current: usize, total: usize, width: usize) -> String {
        let percentage = (current as f64 / total as f64 * 100.0) as usize;
        let filled = (current as f64 / total as f64 * width as f64) as usize;
        
        let mut bar = String::new();
        write!(bar, "[").unwrap();
        
        for i in 0..width {
            if i < filled {
                write!(bar, "█").unwrap();
            } else {
                write!(bar, "░").unwrap();
            }
        }
        
        write!(bar, "] {}% ({}/{})", percentage, current, total).unwrap();
        bar
    }
}