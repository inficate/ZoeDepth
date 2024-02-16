import gradio as gr
from depth import get_depth

inficate = gr.Interface(
    fn=get_depth,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[
        gr.Image(type="pil", label="Original Depth Image"),
        gr.Image(type="pil", label="Refined Depth Image")
    ]
)

inficate.launch(share=True, height=800)
