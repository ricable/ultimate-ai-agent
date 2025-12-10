"""
Memory demonstration component for SAFLA HuggingFace Space.
"""

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import logging

from src.core.safla_manager import SAFLAManager

logger = logging.getLogger(__name__)


class MemoryDemo:
    """Interactive memory system demonstration."""
    
    def __init__(self, safla_manager: SAFLAManager):
        """
        Initialize memory demo component.
        
        Args:
            safla_manager: SAFLA system manager instance
        """
        self.safla_manager = safla_manager
        self.last_search_results = []
    
    def create_interface(self) -> None:
        """Create the memory demo interface."""
        gr.Markdown("## 🧠 Hybrid Memory System Demonstration")
        gr.Markdown(
            "Explore SAFLA's advanced memory architecture including vector, "
            "episodic, semantic, and working memory systems."
        )
        
        with gr.Tabs():
            with gr.TabItem("🔍 Vector Memory Search"):
                self._create_vector_memory_interface()
            
            with gr.TabItem("📚 Episodic Memory"):
                self._create_episodic_memory_interface()
            
            with gr.TabItem("🕸️ Semantic Memory"):
                self._create_semantic_memory_interface()
            
            with gr.TabItem("💭 Working Memory"):
                self._create_working_memory_interface()
    
    def _create_vector_memory_interface(self) -> None:
        """Create vector memory search interface."""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Vector Memory Search")
                gr.Markdown(
                    "Search through high-dimensional vector space to find "
                    "semantically similar content."
                )
                
                search_input = gr.Textbox(
                    label="Search Query",
                    placeholder="Enter concepts, ideas, or questions...",
                    lines=2,
                    elem_id="vector-search-input"
                )
                
                with gr.Row():
                    vector_dim = gr.Dropdown(
                        choices=[512, 768, 1024, 1536],
                        value=768,
                        label="Vector Dimensions",
                        info="Higher dimensions capture more nuanced relationships"
                    )
                    
                    similarity_threshold = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.05,
                        label="Similarity Threshold",
                        info="Minimum similarity score for results"
                    )
                
                max_results = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Maximum Results"
                )
                
                search_btn = gr.Button(
                    "🔍 Search Vector Memory",
                    variant="primary",
                    elem_id="vector-search-btn"
                )
                
                gr.Examples(
                    examples=[
                        "artificial intelligence and machine learning",
                        "neural network architectures",
                        "safety validation techniques",
                        "meta-cognitive processing"
                    ],
                    inputs=search_input
                )
            
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("📊 Results"):
                        results_display = gr.JSON(
                            label="Search Results",
                            elem_id="vector-results"
                        )
                    
                    with gr.TabItem("🎨 Visualization"):
                        similarity_plot = gr.Plot(
                            label="Similarity Visualization",
                            elem_id="similarity-plot"
                        )
                    
                    with gr.TabItem("📈 Analytics"):
                        search_analytics = gr.Dataframe(
                            headers=["Metric", "Value", "Status"],
                            label="Search Performance Analytics"
                        )
        
        # Event handlers
        search_btn.click(
            fn=self.perform_vector_search,
            inputs=[search_input, similarity_threshold, max_results],
            outputs=[results_display, similarity_plot, search_analytics]
        )
    
    async def perform_vector_search(
        self,
        query: str,
        threshold: float,
        max_results: int
    ) -> Tuple[List[Dict], go.Figure, pd.DataFrame]:
        """
        Perform vector similarity search.
        
        Args:
            query: Search query
            threshold: Similarity threshold
            max_results: Maximum number of results
            
        Returns:
            Tuple of (results, visualization, analytics)
        """
        try:
            # Validate input
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
            
            # Perform search
            results = await self.safla_manager.search_vector_memory(
                query=query,
                threshold=threshold,
                max_results=max_results
            )
            
            # Store results for visualization
            self.last_search_results = results
            
            # Create visualization
            fig = self._create_similarity_visualization(results, query)
            
            # Generate analytics
            analytics = self._generate_search_analytics(results, query)
            
            return results, fig, analytics
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return [], go.Figure(), pd.DataFrame()
    
    def _create_similarity_visualization(
        self,
        results: List[Dict],
        query: str
    ) -> go.Figure:
        """Create similarity score visualization."""
        if not results:
            return go.Figure().add_annotation(
                text="No results found",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )
        
        # Extract data
        contents = [r.get('content', '')[:50] + '...' for r in results]
        similarities = [r.get('similarity', 0) for r in results]
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=similarities,
                y=contents,
                orientation='h',
                marker=dict(
                    color=similarities,
                    colorscale='Blues',
                    showscale=True,
                    colorbar=dict(title="Similarity")
                ),
                text=[f"{s:.3f}" for s in similarities],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title=f"Similarity Scores for: '{query[:50]}...'",
            xaxis_title="Similarity Score",
            yaxis_title="Content",
            height=400,
            margin=dict(l=200),
            xaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def _generate_search_analytics(
        self,
        results: List[Dict],
        query: str
    ) -> pd.DataFrame:
        """Generate search performance analytics."""
        metrics = []
        
        # Basic metrics
        metrics.append({
            "Metric": "Results Found",
            "Value": len(results),
            "Status": "✅" if len(results) > 0 else "❌"
        })
        
        if results:
            similarities = [r.get('similarity', 0) for r in results]
            metrics.extend([
                {
                    "Metric": "Average Similarity",
                    "Value": f"{np.mean(similarities):.3f}",
                    "Status": "✅" if np.mean(similarities) > 0.7 else "⚠️"
                },
                {
                    "Metric": "Max Similarity",
                    "Value": f"{np.max(similarities):.3f}",
                    "Status": "✅"
                },
                {
                    "Metric": "Min Similarity",
                    "Value": f"{np.min(similarities):.3f}",
                    "Status": "✅" if np.min(similarities) > 0.5 else "⚠️"
                }
            ])
        
        metrics.append({
            "Metric": "Query Length",
            "Value": len(query.split()),
            "Status": "✅" if 2 <= len(query.split()) <= 20 else "⚠️"
        })
        
        return pd.DataFrame(metrics)
    
    def _create_episodic_memory_interface(self) -> None:
        """Create episodic memory interface."""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Episodic Memory Explorer")
                gr.Markdown(
                    "Navigate through temporal experiences and memories "
                    "stored in the episodic memory system."
                )
                
                # Time range selector
                time_range = gr.Radio(
                    choices=["Last Hour", "Last Day", "Last Week", "Last Month", "All Time"],
                    value="Last Day",
                    label="Time Range"
                )
                
                experience_type = gr.Dropdown(
                    choices=["All", "Learning", "Safety Events", "Performance", "User Interactions"],
                    value="All",
                    label="Experience Type"
                )
                
                emotion_filter = gr.Dropdown(
                    choices=["All", "Positive", "Negative", "Neutral", "Learning Moments"],
                    value="All",
                    label="Emotional Context"
                )
                
                recall_btn = gr.Button(
                    "🧠 Recall Experiences",
                    variant="primary"
                )
            
            with gr.Column(scale=2):
                experience_timeline = gr.Plot(
                    label="Experience Timeline"
                )
                
                experience_details = gr.JSON(
                    label="Experience Details"
                )
        
        # Placeholder for episodic memory functionality
        recall_btn.click(
            fn=self._recall_episodic_memories,
            inputs=[time_range, experience_type, emotion_filter],
            outputs=[experience_timeline, experience_details]
        )
    
    def _recall_episodic_memories(
        self,
        time_range: str,
        experience_type: str,
        emotion_filter: str
    ) -> Tuple[go.Figure, Dict]:
        """Recall episodic memories based on filters."""
        # Create sample timeline visualization
        fig = go.Figure()
        
        # Generate sample data
        now = datetime.now()
        time_ranges = {
            "Last Hour": timedelta(hours=1),
            "Last Day": timedelta(days=1),
            "Last Week": timedelta(weeks=1),
            "Last Month": timedelta(days=30),
            "All Time": timedelta(days=365)
        }
        
        delta = time_ranges.get(time_range, timedelta(days=1))
        
        # Sample experiences
        experiences = []
        for i in range(10):
            time_offset = delta * (i / 10)
            experiences.append({
                "timestamp": (now - time_offset).isoformat(),
                "type": experience_type if experience_type != "All" else "Learning",
                "content": f"Experience {i}: Sample memory content",
                "emotion": emotion_filter if emotion_filter != "All" else "Positive",
                "importance": np.random.random()
            })
        
        # Create timeline
        fig.add_trace(go.Scatter(
            x=[exp["timestamp"] for exp in experiences],
            y=[exp["importance"] for exp in experiences],
            mode='markers+lines',
            marker=dict(
                size=10,
                color=[exp["importance"] for exp in experiences],
                colorscale='Viridis',
                showscale=True
            ),
            text=[exp["content"] for exp in experiences],
            hovertemplate='%{text}<br>Importance: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Episodic Memory Timeline",
            xaxis_title="Time",
            yaxis_title="Importance Score",
            height=400
        )
        
        return fig, {"experiences": experiences[:5]}  # Return top 5 for details
    
    def _create_semantic_memory_interface(self) -> None:
        """Create semantic memory interface."""
        gr.Markdown("### Semantic Knowledge Graph")
        gr.Markdown(
            "Explore the interconnected knowledge representation "
            "in the semantic memory system."
        )
        
        with gr.Row():
            concept_input = gr.Textbox(
                label="Concept to Explore",
                placeholder="Enter a concept to see its connections..."
            )
            
            depth_slider = gr.Slider(
                minimum=1,
                maximum=5,
                value=2,
                step=1,
                label="Exploration Depth"
            )
            
            explore_btn = gr.Button("🕸️ Explore Connections")
        
        knowledge_graph = gr.Plot(label="Knowledge Graph Visualization")
        
        # Placeholder functionality
        explore_btn.click(
            fn=self._explore_semantic_connections,
            inputs=[concept_input, depth_slider],
            outputs=knowledge_graph
        )
    
    def _explore_semantic_connections(
        self,
        concept: str,
        depth: int
    ) -> go.Figure:
        """Explore semantic connections for a concept."""
        # Create sample knowledge graph
        fig = go.Figure()
        
        # Sample nodes and edges
        nodes = [concept]
        edges = []
        
        # Generate sample connections
        for d in range(1, min(depth + 1, 3)):
            for i in range(3):
                related = f"{concept}_related_{d}_{i}"
                nodes.append(related)
                edges.append((concept if d == 1 else f"{concept}_related_{d-1}_0", related))
        
        # Create network visualization
        # This is a simplified visualization
        fig.add_annotation(
            text=f"Knowledge graph for '{concept}' (depth={depth})<br>Nodes: {len(nodes)}, Edges: {len(edges)}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False
        )
        
        return fig
    
    def _create_working_memory_interface(self) -> None:
        """Create working memory interface."""
        gr.Markdown("### Working Memory Monitor")
        gr.Markdown(
            "View and interact with the active working memory, "
            "including current context and attention focus."
        )
        
        with gr.Row():
            with gr.Column():
                working_memory_status = gr.JSON(
                    label="Current Working Memory State"
                )
                
                refresh_btn = gr.Button("🔄 Refresh Status")
            
            with gr.Column():
                attention_plot = gr.Plot(
                    label="Attention Distribution"
                )
        
        # Placeholder functionality
        refresh_btn.click(
            fn=self._get_working_memory_status,
            outputs=[working_memory_status, attention_plot]
        )
    
    def _get_working_memory_status(self) -> Tuple[Dict, go.Figure]:
        """Get current working memory status."""
        # Sample working memory state
        status = {
            "capacity_used": "45%",
            "active_items": 7,
            "focus": "vector_search",
            "context_stack": [
                "user_query",
                "search_operation",
                "result_processing"
            ],
            "attention_weights": {
                "user_input": 0.3,
                "search_results": 0.4,
                "system_state": 0.2,
                "background": 0.1
            }
        }
        
        # Create attention visualization
        fig = go.Figure(data=[
            go.Pie(
                labels=list(status["attention_weights"].keys()),
                values=list(status["attention_weights"].values()),
                hole=0.3
            )
        ])
        
        fig.update_layout(
            title="Attention Distribution",
            height=300
        )
        
        return status, fig
    
    def validate_search_input(self, query: str, threshold: float) -> None:
        """
        Validate search input parameters.
        
        Args:
            query: Search query
            threshold: Similarity threshold
            
        Raises:
            ValueError: If input is invalid
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")