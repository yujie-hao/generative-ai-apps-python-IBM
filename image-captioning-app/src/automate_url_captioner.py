import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from transformers import AutoProcessor, BlipForConditionalGeneration
import gradio as gr

# Load the pretrained processor and model
# https://huggingface.co/Salesforce/blip-image-captioning-large
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Captioning function
def caption_from_url(web_url: str) -> list[str]:
    # URL of the page to scrape
    # web_url = "https://en.wikipedia.org/wiki/IBM"
    # web_url = "https://en.wikipedia.org/wiki/Thomas_J._Watson"
    # Returns an array of captions for all images on a webpage.
    captions = []

    # Download the page
    response = requests.get(web_url)
    # Parse the page with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all img elements
    img_elements = soup.find_all('img')

    # iterate over each img element
    for img_element in img_elements:
        img_url = img_element.get('src')

        # skip if the image is an SVG or too small (eg. icon)
        if 'svg' in img_url or '1x1' in img_url:
            continue
        # correct the url if it's malformed
        if img_url.startswith('//'):
            img_url = 'https:' + img_url
        elif not img_url.startswith('http://') and not img_url.startswith('https://'):
            continue

        try:
            # Download the image
            response = requests.get(img_url)
            # Convert the image data to a PIL Image
            raw_image = Image.open(BytesIO(response.content))
            if raw_image.size[0] * raw_image.size[1] < 400:  # skip small img
                continue
            raw_image = raw_image.convert('RGB')

            # Process the image
            inputs = processor(raw_image, return_tensors="pt")
            # Generate a caption for the image
            out = model.generate(**inputs, max_new_tokens=50)
            # Decode the generated tokens to text
            caption = processor.decode(out[0], skip_special_tokens=True)
            captions.append(f"Image: {img_url}\n\nCaption: {caption}")
        except Exception as e:
            print(f"Error processing image {img_url}: {e}")
            continue
    return captions

def caption_from_url_to_file(web_url: str):
    # URL of the page to scrape
    # web_url = "https://en.wikipedia.org/wiki/IBM"
    # web_url = "https://en.wikipedia.org/wiki/Thomas_J._Watson"
    # Returns an array of captions for all images on a webpage.
    captions = []

    # Download the page
    response = requests.get(web_url)
    # Parse the page with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all img elements
    img_elements = soup.find_all('img')

    # Open a file to write the captions
    with open("captions.txt", "w") as caption_file:
        # iterate over each img element
        for img_element in img_elements:
            img_url = img_element.get('src')

            # skip if the image is an SVG or too small (eg. icon)
            if 'svg' in img_url or '1x1' in img_url:
                continue
            # correct the url if it's malformed
            if img_url.startswith('//'):
                img_url = 'https:' + img_url
            elif not img_url.startswith('http://') and not img_url.startswith('https://'):
                continue

            try:
                # Download the image
                response = requests.get(img_url)
                # Convert the image data to a PIL Image
                raw_image = Image.open(BytesIO(response.content))
                if raw_image.size[0] * raw_image.size[1] < 400: # skip small img
                    continue
                raw_image = raw_image.convert('RGB')

                # Process the image
                inputs = processor(raw_image, return_tensors="pt")
                # Generate a caption for the image
                out = model.generate(**inputs, max_new_tokens=50)
                # Decode the generated tokens to text
                caption = processor.decode(out[0], skip_special_tokens=True)

                # Write the caption to the file, prepended by the image URL
                caption_file.write(f"{img_url}: {caption}\n")
            except Exception as e:
                print(f"Error processing image {img_url}: {e}")
                continue


def display_captions(web_url: str):
    captions = caption_from_url(web_url)
    if not captions:
        return "No valid images found or captions generated."

    # Format as Markdown for better readability
    formatted_output = '## Generated Captions\n\n' + '\n\n------\n\n'.join(captions)
    return formatted_output


# Gradio interface
with gr.Blocks() as app:
    gr.Markdown("## BLIP Image Captioning from URL")
    url_input = gr.Textbox(label="Enter the Web URL", placeholder="https://example.com")
    submit_btn = gr.Button("Generate Captions")
    output = gr.Markdown()
    submit_btn.click(
        fn=display_captions,
        inputs=url_input,
        outputs=output
    )

app.launch(share=True)