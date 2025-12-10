#!/usr/bin/env python3
"""
Simplified SAFLA app using Interface API to avoid queue issues.
"""

import gradio as gr
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import AppSettings
from src.core.safla_manager import SAFLAManager

# Initialize SAFLA
settings = AppSettings()
safla_manager = SAFLAManager(config=settings, test_mode=True)

def search_memory(query: str, threshold: float = 0.7, max_results: int = 5):
    """Search vector memory and return formatted results."""
    if not query or not query.strip():
        return "Please enter a search query", None, None
    
    try:
        # Use sync method
        results = safla_manager.search_vector_memory_sync(
            query=query,
            threshold=threshold,
            max_results=max_results
        )
        
        if not results:
            return "No results found", None, None
        
        # Format results
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"{i}. **{result['content'][:100]}...**\n"
                f"   Similarity: {result['similarity']:.3f}\n"
                f"   Source: {result['source']}\n"
            )
        
        output_text = f"Found {len(results)} results for '{query}':\n\n" + "\n".join(formatted_results)
        
        # Create simple visualization
        similarities = [r['similarity'] for r in results]
        contents = [r['content'][:30] + '...' for r in results]
        
        fig = go.Figure(data=[
            go.Bar(
                x=similarities,
                y=contents,
                orientation='h',
                marker=dict(color='blue')
            )
        ])
        fig.update_layout(
            title=f"Similarity Scores for: '{query}'",
            xaxis_title="Similarity Score",
            height=300
        )
        
        # Create analytics table
        analytics_data = {
            "Metric": ["Results Found", "Average Similarity", "Max Similarity"],
            "Value": [len(results), f"{sum(similarities)/len(similarities):.3f}", f"{max(similarities):.3f}"]
        }
        analytics_df = pd.DataFrame(analytics_data)
        
        return output_text, fig, analytics_df
        
    except Exception as e:
        return f"Error: {str(e)}", None, None

def get_system_status():
    """Get system status."""
    try:
        status = safla_manager.get_system_status()
        
        if status["status"] == "healthy":
            return "🟢 System Status: Operational | Memory: ✅ | Safety: ✅ | Meta-Cognitive: ✅"
        else:
            return "🟡 System Status: Initializing..."
    except Exception as e:
        return f"🔴 System Status: Error - {str(e)}"

# Create simple tabbed interface
with gr.Blocks(title="SAFLA Demo", analytics_enabled=False) as demo:
    gr.Markdown("# 🧠 SAFLA - Self-Aware Feedback Loop Algorithm")
    gr.Markdown("### Interactive AI Demonstration Platform")
    
    # System status
    with gr.Row():
        status_display = gr.Markdown("🟡 System Status: Loading...")
        refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
    
    with gr.Tabs():
        with gr.TabItem("🔍 Memory Search"):
            gr.Markdown("## Vector Memory Search")
            gr.Markdown("Search through SAFLA's vector memory for semantically similar content.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    search_input = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter concepts, ideas, or questions...",
                        lines=2
                    )
                    
                    threshold_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.05,
                        label="Similarity Threshold"
                    )
                    
                    max_results_slider = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Maximum Results"
                    )
                    
                    search_btn = gr.Button("🔍 Search Memory", variant="primary")
                    
                    gr.Examples(
                        examples=[
                            "artificial intelligence",
                            "neural networks", 
                            "safety validation",
                            "meta-cognitive processing"
                        ],
                        inputs=search_input
                    )
                
                with gr.Column(scale=2):
                    search_results = gr.Textbox(
                        label="Search Results",
                        lines=10,
                        interactive=False
                    )
                    
                    with gr.Row():
                        with gr.Column():
                            similarity_plot = gr.Plot(label="Similarity Visualization")
                        
                        with gr.Column():
                            analytics_table = gr.Dataframe(
                                label="Search Analytics",
                                headers=["Metric", "Value"]
                            )
        
        with gr.TabItem("⚙️ Settings"):
            gr.Markdown("## System Configuration")
            gr.Markdown("Configure SAFLA system parameters.")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Memory Settings")
                    vector_dim = gr.Dropdown(
                        choices=[512, 768, 1024, 1536],
                        value=768,
                        label="Vector Dimensions"
                    )
                    
                    memory_size = gr.Slider(
                        minimum=100,
                        maximum=10000,
                        value=1000,
                        step=100,
                        label="Memory Capacity"
                    )
                
                with gr.Column():
                    gr.Markdown("### Performance Settings")
                    cache_timeout = gr.Slider(
                        minimum=60,
                        maximum=3600,
                        value=300,
                        step=60,
                        label="Cache Timeout (seconds)"
                    )
                    
                    max_concurrent = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Max Concurrent Users"
                    )
        
        with gr.TabItem("📚 Documentation"):
            gr.Markdown("""
            ## About SAFLA
            
            SAFLA (Self-Aware Feedback Loop Algorithm) is an advanced AI system that combines:
            
            ### Core Components
            
            1. **Hybrid Memory Architecture**
               - Vector memory for semantic similarity
               - Episodic memory for temporal experiences
               - Semantic memory for knowledge graphs
               - Working memory for active processing
            
            2. **Safety Validation Framework**
               - Multi-layer safety checks
               - Risk assessment scoring
               - Real-time validation
            
            3. **Meta-Cognitive Engine**
               - Self-awareness monitoring
               - Goal adaptation
               - Strategy selection
            
            4. **Delta Evaluation System**
               - Performance tracking
               - Continuous optimization
               - Adaptive learning
            
            ### How to Use
            
            1. **Memory Search**: Use the search tab to explore SAFLA's vector memory
            2. **Settings**: Configure system parameters (display only)
            3. **Documentation**: Learn about SAFLA's capabilities
            
            For more information, visit the [SAFLA GitHub repository](https://github.com/ruvnet/safla).
            """)
    
    # Event handlers - simple functions only
    search_btn.click(
        fn=search_memory,
        inputs=[search_input, threshold_slider, max_results_slider],
        outputs=[search_results, similarity_plot, analytics_table]
    )
    
    refresh_btn.click(
        fn=get_system_status,
        outputs=status_display
    )
    
    demo.load(
        fn=get_system_status,
        outputs=status_display
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)