"""
Custom Gradio theme for SAFLA HuggingFace Space.
"""

import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes


class SaflaTheme(Base):
    """
    Custom theme for SAFLA application with blue-to-purple gradient.
    """
    
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.blue,
        secondary_hue: colors.Color | str = colors.purple,
        neutral_hue: colors.Color | str = colors.gray,
        text_size: sizes.Size | str = sizes.text_md,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        font: list | str = ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono: list | str = ["JetBrains Mono", "ui-monospace", "Consolas", "monospace"],
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            spacing_size=spacing_size,
            radius_size=radius_size,
            font=font,
            font_mono=font_mono,
        )
        
        # Gradio themes have specific styling methods
        # We'll use the built-in color system instead of custom set() calls
        pass


def get_custom_css() -> str:
    """
    Get custom CSS for additional styling.
    """
    return """
    /* Custom gradient backgrounds */
    .gr-button-primary {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    
    .gr-button-primary:hover {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1) !important;
    }
    
    /* Tab styling */
    .gr-tab-item {
        font-weight: 500;
        font-size: 16px;
        padding: 12px 24px;
        border-bottom: 2px solid transparent;
        transition: all 0.2s ease;
    }
    
    .gr-tab-item.selected {
        color: #3b82f6;
        border-bottom-color: #3b82f6;
    }
    
    /* Card effects */
    .gr-box {
        transition: all 0.2s ease;
    }
    
    .gr-box:hover {
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    
    /* Performance metrics styling */
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #3b82f6;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Status indicators */
    .status-healthy {
        color: #10b981;
        font-weight: 600;
    }
    
    .status-degraded {
        color: #f59e0b;
        font-weight: 600;
    }
    
    .status-error {
        color: #ef4444;
        font-weight: 600;
    }
    
    /* Loading animations */
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
    }
    
    .loading {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .gr-tab-item {
            font-size: 14px;
            padding: 8px 16px;
        }
        
        .metric-value {
            font-size: 1.5rem;
        }
    }
    """