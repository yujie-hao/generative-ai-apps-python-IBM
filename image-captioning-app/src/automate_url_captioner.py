import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from transformers import AutoProcessor, BlipForConditionalGeneration
import gradio as gr
import time

# Load the pretrained processor and model

IS_BASE_MODEL = True
model_name = "Salesforce/blip-image-captioning-base" if IS_BASE_MODEL else "Salesforce/blip-image-captioning-large"

# Using "Salesforce/blip-image-captioning-base" for faster loading during demonstration
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base") if IS_BASE_MODEL \
    else AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
# processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base") if IS_BASE_MODEL \
    else BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")


# Captioning function
def caption_from_url(web_url: str, progress=gr.Progress()) -> list[str]:
    """
    Fetches images from a given URL, processes them, and generates captions.
    Progress updates are sent to the Gradio UI.
    """
    captions = []

    # Initialize progress bar to 0% with a starting message.
    # Adding a small sleep here to ensure this initial message has time to render
    # before potentially fast initial operations complete.
    progress(0, desc="Starting image extraction and processing...")
    time.sleep(0.1) # Small delay to ensure initial progress message is displayed

    try:
        response = requests.get(web_url, timeout=15) # Increased timeout for robustness
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        progress(1, desc=f"Error fetching URL: {e}")
        return [f"Error: Could not access the provided URL. {e}"]

    soup = BeautifulSoup(response.text, 'html.parser')
    img_elements = soup.find_all('img')

    if not img_elements:
        progress(1, desc="No image elements found on the page.")
        return ["No valid images found on the provided URL."]

    # Use progress.tqdm to automatically update the progress bar as the loop runs.
    # It takes an iterable and automatically calculates progress based on its length.
    # The 'desc' parameter provides a descriptive message for the progress bar.
    for i, img_element in progress.tqdm(
        enumerate(img_elements),
        total=len(img_elements),
        desc="Processing images"
    ):
        img_url = img_element.get('src')

        # Skip SVG and 1x1 pixel images as they are often decorative or tracking pixels
        if img_url and ('svg' in img_url or '1x1' in img_url):
            continue

        # Construct full URL if it's relative
        if img_url and img_url.startswith('//'):
            img_url = 'https:' + img_url
        elif not (img_url and (img_url.startswith('http://') or img_url.startswith('https://'))):
            # Skip if not a valid absolute URL
            continue

        try:
            # Fetch image content
            img_response = requests.get(img_url, timeout=10)
            img_response.raise_for_status() # Check for bad status codes on image fetch
            raw_image = Image.open(BytesIO(img_response.content))

            # Skip very small images that are unlikely to be meaningful content
            if raw_image.size[0] * raw_image.size[1] < 400:
                continue

            raw_image = raw_image.convert('RGB')

            # Process image and generate caption
            inputs = processor(raw_image, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(out[0], skip_special_tokens=True)
            captions.append(f"Image: {img_url}\n\nCaption: {caption}")

            # Simulate work to make progress more visible for demonstration
            time.sleep(0.1)

        except Exception as e:
            # Print error for debugging, but continue processing other images
            print(f"Error processing image {img_url}: {e}")
            continue

    # Mark progress as complete
    progress(1, desc="Finished generating captions.")
    return captions

def display_captions(web_url: str, progress=gr.Progress()):
    """
    Wrapper function to call caption_from_url and format its output.
    The 'progress' object is automatically passed by Gradio and then
    forwarded to the 'caption_from_url' function.
    """
    captions = caption_from_url(web_url, progress=progress)
    if not captions:
        return "No valid images found or captions generated."
    formatted_output = '## Generated Captions\n\n' + '\n\n------\n\n'.join(captions)
    return formatted_output

# Gradio interface definition
with gr.Blocks() as app:
    gr.Markdown(
        f"""
        **BLIP Image Captioning from URL**
        - This is a simple web app for generating captions for images using BLIP framework
        - Model: {model_name}
        """
    )

    url_input = gr.Textbox(label="Enter the Web URL", placeholder="https://example.com")
    submit_btn = gr.Button("Generate Captions")
    output = gr.Markdown()

    # The click event listener calls display_captions.
    # Gradio automatically injects the 'progress' object into display_captions
    # because its signature includes 'progress=gr.Progress()'.
    submit_btn.click(
        fn=display_captions,
        inputs=url_input,
        outputs=output
    )

app.launch(share=True)
