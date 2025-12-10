#!/usr/bin/env python3
"""
Simple SAFLA app to test queue issues.
"""

import gradio as gr
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import AppSettings
from src.core.safla_manager import SAFLAManager

def get_status():
    """Simple status function."""
    return "🟢 System Status: Operational | Memory: ✅ | Safety: ✅"

def search_demo(query):
    """Simple search demo."""
    if not query:
        return "Please enter a search query"
    
    return f"Search results for: '{query}'\n\n1. Sample result - AI concepts (similarity: 0.92)\n2. Related content - ML principles (similarity: 0.87)\n3. Associated data - neural networks (similarity: 0.81)"

# Create simple interface
with gr.Blocks(title="SAFLA Demo", analytics_enabled=False) as demo:
    gr.Markdown("# 🧠 SAFLA Demo")
    
    with gr.Row():
        status_display = gr.Markdown("🟢 System Status: Loading...")
        refresh_btn = gr.Button("🔄 Refresh")
    
    with gr.Row():
        search_input = gr.Textbox(label="Search Query", placeholder="Enter search terms...")
        search_btn = gr.Button("🔍 Search")
    
    search_output = gr.Textbox(label="Search Results", lines=5)
    
    # Event handlers - simple functions only
    refresh_btn.click(fn=get_status, outputs=status_display)
    search_btn.click(fn=search_demo, inputs=search_input, outputs=search_output)
    demo.load(fn=get_status, outputs=status_display)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)