#!/usr/bin/env python3
"""
Minimal Gradio test to identify queue issue.
"""

import gradio as gr

def hello(name):
    return f"Hello {name}!"

# Most basic interface possible
iface = gr.Interface(
    fn=hello,
    inputs="text",
    outputs="text",
    title="Minimal Test"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7862, share=False)