"""
SAFLA HuggingFace Space - Main Application
Self-Aware Feedback Loop Algorithm Demonstration Platform
"""

import gradio as gr
import asyncio
import logging
import sys
from pathlib import Path
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import after path setup
from src.config.settings import AppSettings, UI_CONFIG
from src.core.safla_manager import SAFLAManager
from src.ui.themes.safla_theme import SaflaTheme, get_custom_css
from src.ui.tabs.demo_tab import DemoTab


class SAFLAApp:
    """Main application class for SAFLA HuggingFace Space."""
    
    def __init__(self):
        """Initialize the SAFLA application."""
        logger.info("Initializing SAFLA HuggingFace Space...")
        
        self.settings = AppSettings()
        self.safla_manager = None
        self.interface = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize application components."""
        try:
            # Initialize SAFLA manager
            self.safla_manager = SAFLAManager(config=self.settings)
            
            # Run async initialization
            asyncio.run(self._async_init())
            
            logger.info("SAFLA components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            # Continue with limited functionality
            self.safla_manager = SAFLAManager(config=self.settings, test_mode=True)
    
    async def _async_init(self):
        """Async initialization tasks."""
        await self.safla_manager.initialize_system()
    
    def create_interface(self) -> gr.Blocks:
        """
        Create the main Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        # Create tab instances
        demo_tab = DemoTab(self.safla_manager)
        
        # Build interface with custom theme
        with gr.Blocks(
            theme=SaflaTheme(),
            title="SAFLA - Self-Aware Feedback Loop Algorithm",
            css=get_custom_css(),
            analytics_enabled=self.settings.enable_analytics
        ) as interface:
            
            # Header
            with gr.Row():
                gr.Markdown(
                    "# 🧠 SAFLA - Self-Aware Feedback Loop Algorithm\n"
                    "### Interactive AI Demonstration Platform"
                )
            
            # System status indicator
            with gr.Row():
                with gr.Column(scale=3):
                    system_status = gr.Markdown("🟢 System Status: Initializing...")
                with gr.Column(scale=1):
                    refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
            
            # Main tabs
            with gr.Tabs():
                with gr.TabItem("🎯 Interactive Demo", elem_id="demo-tab"):
                    demo_tab.create_interface()
                
                with gr.TabItem("⚙️ Settings & Configuration", elem_id="settings-tab"):
                    self._create_settings_tab()
                
                with gr.TabItem("📊 Benchmarking & Analytics", elem_id="benchmark-tab"):
                    self._create_benchmark_tab()
                
                with gr.TabItem("📚 Documentation & Tutorials", elem_id="docs-tab"):
                    self._create_docs_tab()
            
            # Footer
            gr.Markdown(
                "---\n"
                "Made with ❤️ by the SAFLA Team | "
                "[GitHub](https://github.com/ruvnet/safla) | "
                "[Paper](https://arxiv.org/safla) | "
                "Version 1.0.0"
            )
            
            # Event handlers
            interface.load(
                fn=self._get_system_status,
                outputs=[system_status]
            )
            
            refresh_btn.click(
                fn=self._get_system_status,
                outputs=[system_status]
            )
        
        self.interface = interface
        return interface
    
    def _create_settings_tab(self):
        """Create settings and configuration tab."""
        gr.Markdown("## ⚙️ Settings & Configuration")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Memory Configuration")
                
                vector_dim = gr.Dropdown(
                    choices=[512, 768, 1024, 1536],
                    value=self.settings.safla_vector_dim,
                    label="Vector Dimensions",
                    info="Higher dimensions capture more nuanced relationships"
                )
                
                memory_size = gr.Slider(
                    minimum=100,
                    maximum=10000,
                    value=self.settings.safla_memory_size,
                    step=100,
                    label="Memory Capacity",
                    info="Maximum number of items to store"
                )
                
                gr.Markdown("### Performance Settings")
                
                cache_timeout = gr.Slider(
                    minimum=60,
                    maximum=3600,
                    value=self.settings.cache_timeout,
                    step=60,
                    label="Cache Timeout (seconds)",
                    info="How long to cache results"
                )
                
                max_concurrent = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=self.settings.max_concurrent_users,
                    step=1,
                    label="Max Concurrent Users"
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### Current Configuration")
                config_display = gr.JSON(
                    value=self._get_current_config(),
                    label="Active Settings"
                )
                
                with gr.Row():
                    apply_btn = gr.Button("✅ Apply Changes", variant="primary")
                    reset_btn = gr.Button("🔄 Reset to Defaults")
                    export_btn = gr.Button("📤 Export Configuration")
        
        # Placeholder for settings functionality
        gr.Markdown(
            "⚠️ **Note**: Configuration changes will be available in the next update. "
            "Current settings are display-only."
        )
    
    def _create_benchmark_tab(self):
        """Create benchmarking and analytics tab."""
        gr.Markdown("## 📊 Benchmarking & Analytics")
        
        with gr.Row():
            # Real-time metrics
            with gr.Column(scale=1):
                gr.Markdown("### Real-time Metrics")
                
                cpu_usage = gr.Number(
                    label="CPU Usage (%)",
                    value=0,
                    precision=1
                )
                
                memory_usage = gr.Number(
                    label="Memory (MB)",
                    value=0,
                    precision=1
                )
                
                response_time = gr.Number(
                    label="Avg Response (ms)",
                    value=0,
                    precision=0
                )
                
                throughput = gr.Number(
                    label="Throughput (ops/s)",
                    value=0,
                    precision=1
                )
            
            with gr.Column(scale=3):
                gr.Markdown("### Performance Dashboard")
                gr.Markdown("📈 Performance metrics and benchmarking dashboard coming soon...")
                
                # Placeholder for charts
                with gr.Tabs():
                    with gr.TabItem("Performance Trends"):
                        gr.Markdown("Performance trend charts will be displayed here")
                    
                    with gr.TabItem("System Health"):
                        gr.Markdown("System health monitoring will be displayed here")
                    
                    with gr.TabItem("Benchmark Results"):
                        gr.Markdown("Benchmark test results will be displayed here")
    
    def _create_docs_tab(self):
        """Create documentation and tutorials tab."""
        gr.Markdown("## 📚 Documentation & Tutorials")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    "### Quick Links\n\n"
                    "- [Getting Started](#getting-started)\n"
                    "- [SAFLA Overview](#safla-overview)\n"
                    "- [Memory System](#memory-system)\n"
                    "- [Safety Framework](#safety-framework)\n"
                    "- [API Reference](#api-reference)\n"
                    "- [Troubleshooting](#troubleshooting)"
                )
            
            with gr.Column(scale=3):
                gr.Markdown(
                    """
                    ### Getting Started
                    
                    Welcome to the SAFLA demonstration platform! This interactive space allows you to explore 
                    the capabilities of the Self-Aware Feedback Loop Algorithm.
                    
                    #### What is SAFLA?
                    
                    SAFLA is an advanced AI system that combines multiple cutting-edge technologies:
                    
                    1. **Hybrid Memory Architecture**: Integrates vector, episodic, semantic, and working memory
                    2. **Safety Validation Framework**: Multi-layer safety checks with risk assessment
                    3. **Meta-Cognitive Engine**: Self-awareness and adaptive learning capabilities
                    4. **Delta Evaluation System**: Continuous performance tracking and optimization
                    
                    #### How to Use This Platform
                    
                    1. **Interactive Demo Tab**: Explore SAFLA's capabilities through hands-on demonstrations
                    2. **Settings Tab**: Configure system parameters and see their effects
                    3. **Benchmarking Tab**: View real-time performance metrics and run benchmarks
                    4. **Documentation Tab**: Learn more about SAFLA's architecture and features
                    
                    #### Example Use Cases
                    
                    - **Content Understanding**: Use vector memory search to find semantically similar content
                    - **Safety Validation**: Test the multi-layer safety framework with various inputs
                    - **Performance Analysis**: Monitor system performance and optimization strategies
                    - **Learning Observation**: Watch the meta-cognitive engine adapt and improve
                    
                    For more detailed information, visit our [GitHub repository](https://github.com/ruvnet/safla).
                    """
                )
    
    def _get_system_status(self) -> str:
        """Get current system status."""
        try:
            # Now using sync method
            status = self.safla_manager.get_system_status()
            
            if status["status"] == "healthy":
                icon = "🟢"
                text = "Operational"
            elif status["status"] == "degraded":
                icon = "🟡"
                text = "Degraded"
            else:
                icon = "🔴"
                text = "Error"
            
            details = []
            if status["memory_initialized"]:
                details.append("Memory: ✅")
            if status["safety_active"]:
                details.append("Safety: ✅")
            if status["metacognitive_running"]:
                details.append("Meta-Cognitive: ✅")
            
            status_text = f"{icon} System Status: {text}"
            if details:
                status_text += f" | {' | '.join(details)}"
            
            return status_text
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return "🔴 System Status: Error retrieving status"
    
    def _get_current_config(self) -> dict:
        """Get current configuration as dictionary."""
        return {
            "environment": self.settings.environment,
            "safla": self.settings.get_safla_config(),
            "performance": self.settings.get_performance_config(),
            "debug": self.settings.is_debug()
        }
    
    def launch(self, **kwargs):
        """
        Launch the Gradio interface.
        
        Args:
            **kwargs: Additional arguments for gr.launch()
        """
        if self.interface is None:
            self.interface = self.create_interface()
        
        # Default launch configuration
        launch_config = {
            "server_name": "0.0.0.0",
            "server_port": 7860,
            "share": False,
            "debug": self.settings.is_debug()
        }
        
        # Override with provided kwargs
        launch_config.update(kwargs)
        
        logger.info(f"Launching SAFLA HuggingFace Space on port {launch_config['server_port']}...")
        
        # Remove any queue-related parameters and launch
        launch_config.pop("enable_queue", None)
        
        # Set max_threads to 1 to minimize queue issues
        launch_config["max_threads"] = 1
        
        return self.interface.launch(**launch_config)


def main():
    """Main entry point for the application."""
    try:
        # Create and launch application
        app = SAFLAApp()
        app.launch()
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    main()