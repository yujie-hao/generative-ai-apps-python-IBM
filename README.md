# generative-ai-apps-python-IBM
Generative AI applications in Python

## 1. image-captioning-app
**[Description]**
- This is a simple image captioning application that uses a pre-trained model to generate captions for images.

**[Language]**
- Python<br>

**[Framework]**
- Gradio.app

**[Model]**
- BLIP (Bootstrapping Language-Image Pre-training)<br>
https://huggingface.co/Salesforce/blip-image-captioning-large
<br>![img.png](image-captioning-app/docs/blip.png)

**[Apps]**
1. image captioning app: upload an image and get a caption for it.
2. automate web page image captioning: enter a URL, the app will fetch the images from it to generate captions.

 **[Libs]**
 - beautifulsoup: https://pypi.org/project/beautifulsoup4/
   - Beautiful Soup is a library that makes it easy to scrape information from web pages. It sits atop an HTML or XML parser, providing Pythonic idioms for iterating, searching, and modifying the parse tree. 