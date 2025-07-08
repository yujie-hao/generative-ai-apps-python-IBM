# Generative-ai-apps-python
Generative AI applications in Python

## 1. image-captioning-app
### [Description]
- This is a simple image captioning application that uses a pre-trained model to generate captions for images.

### [Language]
- Python<br>

### [Framework]
- Gradio.app

### [Model]
- BLIP (Bootstrapping Language-Image Pre-training)<br>
https://huggingface.co/Salesforce/blip-image-captioning-large
<br>![img.png](image-captioning-app/docs/blip.png)

### [Apps]
1. image captioning app: upload an image and get a caption for it. (https://huggingface.co/spaces/CogitativePanda/img_captioning_app_blip_gradio)
2. automate web page image captioning: enter a URL, the app will fetch the images from it to generate captions. (https://huggingface.co/spaces/CogitativePanda/automate_url_captioner)

### [Demo]
- https://youtu.be/kH2kkC45CVU

### [Libs]
 - beautifulsoup: https://pypi.org/project/beautifulsoup4/
   - Beautiful Soup is a library that makes it easy to scrape information from web pages. It sits atop an HTML or XML parser, providing Pythonic idioms for iterating, searching, and modifying the parse tree.


### [Process]
1. Preprocessing
    ```
    inputs = processor(image, return_tensors="pt")
    ```
    - Resizes, normalizes, and converts the image into tensor format 
    - Adds attention masks and padding for the decoder


2. Encoding (Vision Transformer)
    ```
    vision_embeddings = model.vision_encoder(inputs["pixel_values"])
    ```
   - The image is split into patches → embedded → passed through ViT (Vision Transformer, e.g. ViT-B/16)
   - Outputs are rich visual feature vectors


3. Decoding (Text Generator)
    ```
    generated_ids = model.generate(**inputs)
    ```
    - The decoder receives the visual embeddings and begins generating tokens 
    - Uses autoregressive decoding: each predicted token is fed back into the model to predict the next


4. Postprocessing
    ```
    caption = processor.decode(generated_ids[0], skip_special_tokens=True)
    ```
    - Converts token IDs into a readable sentence

## 2. chatbot
### [Description]
- Simple Chatbot with Open Source LLM (facebook/blenderbot-400M-distill) using Python and Hugging Face.

### [Language]
- Python<br>

### [Framework]
- NA

### [Model]
- blenderbot-400M-distill: https://huggingface.co/facebook/blenderbot-400M-distill

### [Architecture]
![img.png](image-captioning-app/docs/chatbot_arch.png)

### [Process]
1. Input processing
   
   When user send a message to the chatbot, the transformer helps process user's input. It breaks down user's message into smaller parts and represents them in a way that the chatbot can understand. Each part is called a token.
2. Understanding context

   The transformer passes these tokens to the LLM, which is a language model trained on lots of text data. The LLM has learned patterns and meanings from this data, so it tries to understand the context of user's message based on what it has learned.
3. Generating response

   Once the LLM understands user's message, it generates a response based on its understanding. The transformer then takes this response and converts it into a format that can be easily sent back to user.
4. Iterative conversation

   As the conversation continues, this process repeats. The transformer and LLM work together to process each new input message, understand the context, and generate a relevant response.