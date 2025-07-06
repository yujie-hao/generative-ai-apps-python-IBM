import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Role: Preprocesses raw inputs (images/text) into a format the model understands.
# - Preprocesses raw image/text into tensors
# - Converts PIL/Numpy images → PyTorch tensors.
# - Adds model-specific preprocessing (e.g., resizing to 224x224 for BLIP).
# - Generates attention_mask and other required inputs automatically.
# Note: AutoProcessor here detects & selects the correct processor (BlipProcessor)
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")

# Role: Generates text (captions) from the inputs
# - Architecture
#   - Vision Encoder: processes the image (CNN-based)
#   - Text Decoder: generates captions autoregressively
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base") # smaller / faster inference
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large")  # more accurate / slower inference


def caption_image(input_image: np.ndarray):
    # Convert numpy array to PIL Image and convert to RGB
    raw_image = Image.fromarray(input_image).convert('RGB')

    # Process the PIL.image (numpy image) -> returns a dictionary containing:
    # - pixel_values (tensor-encoded image)
    # - input_ids (if text is used)
    inputs = processor(images=raw_image, return_tensors="pt")

    # Generate a caption for the image
    # `**` is used for dictionary unpacking (also called "keyword argument unpacking").
    # It takes a dictionary (inputs) and passes its key-value pairs as named arguments to the function.
    # inputs = {"a": 1, "b": 2}
    # foo(**inputs)  # Equivalent to foo(a=1, b=2)
    # pixel_values=inputs["pixel_values"],
    # attention_mask=inputs["attention_mask"],
    outputs = model.generate(**inputs, max_length=50)
    # Decode the generated tokens to text and store it into `caption`
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption


iface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(),
    outputs="text",
    title="Image Captioning",
    description="This is a simple web app for generating captions for images using a trained model"
)

# iface.launch(server_name="0.0.0.0", server_port=7860)
iface.launch(share=True)