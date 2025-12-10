"""
Demo tab implementation for SAFLA HuggingFace Space.
"""

import gradio as gr
from typing import Optional
import logging

from src.core.safla_manager import SAFLAManager
from src.ui.components.memory_demo import MemoryDemo

logger = logging.getLogger(__name__)


class DemoTab:
    """Interactive demonstration tab for SAFLA capabilities."""
    
    def __init__(self, safla_manager: SAFLAManager):
        """
        Initialize demo tab.
        
        Args:
            safla_manager: SAFLA system manager instance
        """
        self.safla_manager = safla_manager
        self.memory_demo = MemoryDemo(safla_manager)
        # Initialize other demo components as needed
        self.safety_demo = None  # Placeholder
        self.metacognitive_demo = None  # Placeholder
    
    def create_interface(self) -> None:
        """Create the demo tab interface."""
        with gr.Column():
            gr.Markdown(
                "## 🎯 SAFLA Interactive Demonstrations\n\n"
                "Explore the advanced capabilities of the Self-Aware Feedback Loop Algorithm "
                "through interactive demonstrations. Choose a demo below to get started."
            )
            
            # Demo selector
            with gr.Row():
                demo_type = gr.Radio(
                    choices=["Memory System", "Safety Validation", "Meta-Cognitive Engine"],
                    value="Memory System",
                    label="Select Demo",
                    elem_id="demo-selector"
                )
            
            # Demo content area
            with gr.Column(visible=True) as memory_content:
                self.memory_demo.create_interface()
            
            with gr.Column(visible=False) as safety_content:
                gr.Markdown("### 🛡️ Safety Validation Demo")
                gr.Markdown("Safety validation demo coming soon...")
                # self.safety_demo.create_interface()
            
            with gr.Column(visible=False) as meta_content:
                gr.Markdown("### 🤖 Meta-Cognitive Engine Demo")
                gr.Markdown("Meta-cognitive demo coming soon...")
                # self.metacognitive_demo.create_interface()
            
            # Demo switcher logic
            def switch_demo(demo_name):
                return (
                    gr.update(visible=demo_name == "Memory System"),
                    gr.update(visible=demo_name == "Safety Validation"),
                    gr.update(visible=demo_name == "Meta-Cognitive Engine")
                )
            
            demo_type.change(
                fn=switch_demo,
                inputs=[demo_type],
                outputs=[memory_content, safety_content, meta_content]
            )
            
            # Add general information
            with gr.Accordion("ℹ️ About SAFLA Demonstrations", open=False):
                gr.Markdown(
                    """
                    ### What is SAFLA?
                    
                    SAFLA (Self-Aware Feedback Loop Algorithm) is an advanced AI system that combines:
                    
                    - **Hybrid Memory Architecture**: Vector, episodic, semantic, and working memory
                    - **Safety Validation Framework**: Multi-layer safety checks and risk assessment
                    - **Meta-Cognitive Engine**: Self-awareness and adaptive learning capabilities
                    - **Delta Evaluation System**: Performance tracking and optimization
                    
                    ### How to Use These Demos
                    
                    1. Select a demo type from the options above
                    2. Follow the instructions in each demo section
                    3. Experiment with different parameters and inputs
                    4. Observe real-time results and visualizations
                    
                    ### Tips for Best Results
                    
                    - Start with the example inputs provided
                    - Adjust parameters gradually to see their effects
                    - Pay attention to the performance metrics
                    - Try different combinations to explore capabilities
                    """
                )