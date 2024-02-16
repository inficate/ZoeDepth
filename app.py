import os
import sys
from IPython.display import clear_output

setup_done_file = '/content/setup_done.flag'

if not os.path.exists(setup_done_file):
    ! pip install --upgrade timm==0.6.7 torch==2.0.1 torchaudio==2.0.2 torchdata==0.6.1 torchtext==0.15.2 torchvision==0.15.2 numpy==1.25.2 pillow==9.4.0 gradio==4.18.0 opencv-python==4.6.0.66 kaleido==0.2.1
    clear_output(wait=True)
    ! git lfs install
    ! if [ -d "/content/ZoeDepth" ]; then rm -rf "/content/ZoeDepth"; fi
    ! git clone https://github.com/inficate/ZoeDepth.git /content/ZoeDepth
    clear_output(wait=True)

    with open(setup_done_file, 'w') as f:
        f.write('Setup Done')

sys.path.append('/content/ZoeDepth')

import gradio as gr
from depth import get_depth
from IPython.display import Javascript

inficate = gr.Interface(
    fn=get_depth,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[
        gr.Image(type="pil", label="Original Depth Image"),
        gr.Image(type="pil", label="Refined Depth Image"),
        gr.File(label="Download the Original Depth Image"),
        gr.File(label="Download the Refined Depth Image")
    ],
    allow_flagging="never"
)

display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 2000})'''))

inficate.launch(share=True, height=800)
