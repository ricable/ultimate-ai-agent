//! DOT format export for network visualization

use crate::io::error::IoResult;
use std::io::Write;

// Import the mock types for now
use crate::mock_types::MockNetwork;

/// DOT format exporter for network visualization
pub struct DotExporter {
    /// Show weights on edges
    pub show_weights: bool,
    /// Show neuron indices
    pub show_indices: bool,
    /// Graph layout direction
    pub layout_direction: LayoutDirection,
}

/// Graph layout direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutDirection {
    /// Left to right (default for neural networks)
    LeftToRight,
    /// Top to bottom
    TopToBottom,
    /// Right to left
    RightToLeft,
    /// Bottom to top
    BottomToTop,
}

impl DotExporter {
    /// Create a new DOT exporter with default settings
    pub fn new() -> Self {
        Self {
            show_weights: false,
            show_indices: true,
            layout_direction: LayoutDirection::LeftToRight,
        }
    }

    /// Create a new DOT exporter with custom settings
    pub fn with_options(
        show_weights: bool,
        show_indices: bool,
        layout_direction: LayoutDirection,
    ) -> Self {
        Self {
            show_weights,
            show_indices,
            layout_direction,
        }
    }

    /// Export a neural network to DOT format
    pub fn export_network<W: Write>(&self, network: &MockNetwork, writer: &mut W) -> IoResult<()> {
        writeln!(writer, "digraph NeuralNetwork {{")?;

        // Set graph attributes
        let rankdir = match self.layout_direction {
            LayoutDirection::LeftToRight => "LR",
            LayoutDirection::TopToBottom => "TB",
            LayoutDirection::RightToLeft => "RL",
            LayoutDirection::BottomToTop => "BT",
        };
        writeln!(writer, "  rankdir={rankdir};")?;
        writeln!(writer, "  node [shape=circle];",)?;
        writeln!(writer, "  edge [dir=forward];",)?;
        writeln!(writer)?;

        // Create nodes for each layer
        let mut node_id = 0;
        let mut layer_nodes = Vec::new();

        for (layer_idx, &layer_size) in network.layer_sizes.iter().enumerate() {
            writeln!(writer, "  // Layer {layer_idx}")?;
            writeln!(writer, "  {{")?;
            writeln!(writer, "    rank=same;")?;

            let mut current_layer_nodes = Vec::new();

            for neuron_idx in 0..layer_size {
                let label = if self.show_indices {
                    format!("L{layer_idx}N{neuron_idx}")
                } else {
                    match layer_idx {
                        0 => format!("I{neuron_idx}"),
                        idx if idx == network.layer_sizes.len() - 1 => format!("O{neuron_idx}"),
                        _ => format!("H{neuron_idx}"),
                    }
                };

                writeln!(writer, "    n{node_id} [label=\"{label}\"];")?;
                current_layer_nodes.push(node_id);
                node_id += 1;
            }

            writeln!(writer, "  }}")?;
            writeln!(writer)?;

            layer_nodes.push(current_layer_nodes);
        }

        // Create edges between layers
        for layer_idx in 0..layer_nodes.len() - 1 {
            let current_layer = &layer_nodes[layer_idx];
            let next_layer = &layer_nodes[layer_idx + 1];

            writeln!(
                writer,
                "  // Connections from layer {} to layer {}",
                layer_idx,
                layer_idx + 1
            )?;

            for &from_node in current_layer {
                for &to_node in next_layer {
                    if self.show_weights && !network.weights.is_empty() {
                        // For demonstration, use a simple weight assignment
                        // In a real implementation, this would use actual network topology
                        let weight_idx = (from_node + to_node) % network.weights.len();
                        let weight = network.weights[weight_idx];

                        let color = if weight > 0.0 { "blue" } else { "red" };
                        let width = (weight.abs() * 5.0).max(0.5).min(5.0);

                        writeln!(
                            writer,
                            "  n{from_node} -> n{to_node} [label=\"{weight:.3}\", color={color}, penwidth={width:.1}];"
                        )?;
                    } else {
                        writeln!(writer, "  n{from_node} -> n{to_node};")?;
                    }
                }
            }
            writeln!(writer)?;
        }

        // Add network information as a comment
        writeln!(writer, "  // Network Information")?;
        writeln!(writer, "  // Layers: {}", network.num_layers)?;
        writeln!(writer, "  // Learning Rate: {:.6}", network.learning_rate)?;
        writeln!(
            writer,
            "  // Connection Rate: {:.6}",
            network.connection_rate
        )?;

        writeln!(writer, "}}")?;

        Ok(())
    }
}

impl Default for DotExporter {
    fn default() -> Self {
        Self::new()
    }
}
