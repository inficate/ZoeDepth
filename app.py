import torch
from PIL import Image
from zoedepth.utils.misc import colorize
import gradio as gr

repo = "isl-org/ZoeDepth"
# Zoe_N
model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)


def get_depth(input_image):
    # Convert the input image to RGB
    image = input_image.convert("RGB")

    # Run inference
    depth = zoe.infer_pil(image)

    # Colorize output
    colored = colorize(depth)

    # Convert array to PIL Image
    output_image = Image.fromarray(colored)

    # Return the colorized output image
    return output_image


# Set up the Gradio interface
inficate = gr.Interface(
    fn=get_depth,  # Use the getDepth function
    inputs=gr.inputs.Image(type="pil", label="Upload Image"),  # Specify input type
    outputs=gr.outputs.Image(type="pil", label="Depth Image"),  # Specify output type
)

# Launch the Gradio app
inficate.launch()
