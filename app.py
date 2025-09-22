import gradio as gr
import torch
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageDraw, ImageFont
import requests
from bs4 import BeautifulSoup
import io
import numpy as np
import os
import json
import time
import warnings
from typing import List, Tuple, Optional
from ddgs import DDGS

# Suppress Starlette deprecation warnings from Gradio
warnings.filterwarnings("ignore", category=DeprecationWarning, module="starlette")

class CLIPImageClassifier:
    def __init__(self):
        """Initialize the CLIP model and processor"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load CLIP model
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Organized default categories by type
        self.category_types = {
            "animals": [
                "cat", "dog", "bird", "horse", "cow", "sheep", "elephant", "bear", 
                "zebra", "giraffe", "lion", "tiger", "fish", "frog", "snake", 
                "spider", "butterfly", "rabbit", "mouse", "hamster", "guinea pig",
                "turtle", "lizard", "monkey", "panda", "koala", "kangaroo", "deer"
            ],
            "vehicles": [
                "car", "truck", "bicycle", "motorcycle", "airplane", "boat", 
                "bus", "train", "helicopter", "scooter", "skateboard", "roller skates",
                "sailboat", "yacht", "submarine", "rocket", "spaceship", "tank"
            ],
            "food": [
                "pizza", "hamburger", "cake", "apple", "banana", "orange", 
                "bread", "sandwich", "salad", "soup", "pasta", "rice", "noodles",
                "chicken", "beef", "fish", "vegetables", "fruits", "dessert",
                "ice cream", "chocolate", "candy", "cookie", "donut", "muffin"
            ],
            "objects": [
                "person", "tree", "house", "flower", "book", "laptop", "phone",
                "computer", "television", "camera", "watch", "glasses", "hat",
                "shirt", "dress", "shoes", "bag", "chair", "table", "bed",
                "lamp", "clock", "mirror", "window", "door", "key", "pen"
            ],
            "nature": [
                "mountain", "ocean", "lake", "river", "forest", "desert", "beach",
                "sky", "cloud", "sun", "moon", "star", "rainbow", "snow",
                "rain", "storm", "sunset", "sunrise", "landscape", "garden"
            ]
        }
        
        # Get all categories as a flat list
        self.all_categories = []
        for category_list in self.category_types.values():
            self.all_categories.extend(category_list)
    
    def search_images_web(self, query: str, num_images: int = 5) -> List[Image.Image]:
        """Search for images using DuckDuckGo with rate limiting handling"""
        try:
            print(f"Searching DuckDuckGo for '{query}' - requesting {num_images} images")
            
            # Add delay to avoid rate limiting
            time.sleep(1)
            
            # Use DuckDuckGo to search for images with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with DDGS() as ddgs:
                        # Search for images
                        image_results = list(ddgs.images(
                            query,
                            max_results=num_images,
                            safesearch="moderate"
                        ))
                    break  # Success, exit retry loop
                except Exception as e:
                    if "403" in str(e) or "Ratelimit" in str(e):
                        wait_time = (attempt + 1) * 5  # Exponential backoff
                        print(f"Rate limited. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                        time.sleep(wait_time)
                        if attempt == max_retries - 1:
                            print("Max retries reached. DuckDuckGo is rate limiting requests.")
                            return []
                    else:
                        print(f"Error with DuckDuckGo search: {e}")
                        return []
            
            if not image_results:
                print(f"Search term '{query}' returned 0 images")
                return []
            
            images = []
            for i, result in enumerate(image_results):
                try:
                    # Download image from the URL
                    img_url = result['image']
                    print(f"Downloading image {i+1}: {img_url}")
                    
                    # Set headers to avoid blocking
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    
                    img_response = requests.get(img_url, headers=headers, timeout=10)
                    if img_response.status_code == 200:
                        img = Image.open(io.BytesIO(img_response.content)).convert('RGB')
                        # Resize to standard size
                        img = img.resize((300, 200), Image.Resampling.LANCZOS)
                        images.append(img)
                        print(f"Successfully downloaded image {i+1}")
                    else:
                        print(f"Failed to download image {i+1}: HTTP {img_response.status_code}")
                        
                except Exception as e:
                    print(f"Error downloading image {i+1}: {e}")
                    continue
            
            print(f"Successfully downloaded {len(images)} images from DuckDuckGo for '{query}'")
            return images
                
        except Exception as e:
            print(f"Error with DuckDuckGo search: {e}")
            return []
    
    def classify_image(self, image: Image.Image, categories: List[str]) -> List[Tuple[str, float]]:
        """Classify an image against given categories"""
        try:
            # Prepare inputs
            inputs = self.processor(
                text=categories, 
                images=image, 
                return_tensors="pt", 
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Get probabilities and sort by confidence
            probs = probs.cpu().numpy()[0]
            results = list(zip(categories, probs))
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results
            
        except Exception as e:
            print(f"Error classifying image: {e}")
            return [("Error", 0.0)]
    
    def classify_multiple_images(self, images: List[Image.Image], categories: List[str]) -> List[List[Tuple[str, float]]]:
        """Classify multiple images"""
        results = []
        for image in images:
            if image is not None:
                result = self.classify_image(image, categories)
                results.append(result)
            else:
                results.append([("No image", 0.0)])
        return results

# Initialize the classifier
classifier = CLIPImageClassifier()

def get_selected_categories(selected_types, custom_categories):
    """Get categories based on selected types and custom input"""
    categories = []
    
    # Add categories from selected types
    for category_type in selected_types:
        if category_type in classifier.category_types:
            categories.extend(classifier.category_types[category_type])
    
    # Add custom categories
    if custom_categories and custom_categories.strip():
        custom_list = [cat.strip() for cat in custom_categories.split(',') if cat.strip()]
        categories.extend(custom_list)
    
    # Remove duplicates while preserving order
    categories = list(dict.fromkeys(categories))
    return categories

def classify_uploaded_images(images, custom_categories, selected_types):
    """Process uploaded images and return classification results"""
    if images is None or len(images) == 0:
        return "Please upload at least one image.", []
    
    # Get categories based on selection
    categories = get_selected_categories(selected_types, custom_categories)
    
    if not categories:
        return "Please select at least one category type or add custom categories.", []
    
    # Load images from file paths
    pil_images = []
    valid_images = []
    print(f"Processing {len(images)} uploaded images")
    for i, image_path in enumerate(images):
        try:
            pil_image = Image.open(image_path).convert('RGB')
            pil_images.append(pil_image)
            valid_images.append(pil_image)  # Only add valid images to gallery
            print(f"Successfully loaded image {i+1}: {image_path} (size: {pil_image.size})")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            pil_images.append(None)
    
    print(f"Total valid images for gallery: {len(valid_images)}")
    
    # Classify images
    results = classifier.classify_multiple_images(pil_images, categories)
    
    # Format results
    result_text = "Classification Results:\n\n"
    for i, image_results in enumerate(results):
        result_text += f"Image {i+1}:\n"
        for category, confidence in image_results[:5]:  # Show top 5 results
            result_text += f"  {category}: {confidence:.3f} ({confidence*100:.1f}%)\n"
        result_text += "\n"
    
    return result_text, valid_images

def search_and_classify_images(search_query, num_images, custom_categories, selected_types):
    """Search for images and classify them"""
    if not search_query or not search_query.strip():
        return "Please enter a search query.", []
    
    # Get categories based on selection
    categories = get_selected_categories(selected_types, custom_categories)
    
    if not categories:
        return "Please select at least one category type or add custom categories.", []
    
    # Parse comma-separated search queries
    search_terms = [term.strip() for term in search_query.split(',') if term.strip()]
    
    if not search_terms:
        return "Please enter valid search terms separated by commas.", []
    
    all_searched_images = []
    search_results_info = []
    
    # Search for each term separately
    images_per_term = max(1, num_images // len(search_terms))  # Distribute images across terms
    
    for i, term in enumerate(search_terms):
        # Search for images for this specific term using real API
        searched_images = classifier.search_images_web(term, images_per_term)
        print(f"Search term '{term}' returned {len(searched_images)} images")
        
        if searched_images:
            all_searched_images.extend(searched_images)
            search_results_info.append(f"Found {len(searched_images)} images for '{term}'")
        else:
            search_results_info.append(f"No images found for '{term}'")
    
    if not all_searched_images:
        return "No images found for any of the search terms. Please try different search terms.", []
    
    # Classify all searched images
    results = classifier.classify_multiple_images(all_searched_images, categories)
    
    # Format results
    result_text = f"Search Results for '{search_query}':\n\n"
    
    # Show search summary
    result_text += "Search Summary:\n"
    for info in search_results_info:
        result_text += f"  • {info}\n"
    result_text += f"\nTotal images found: {len(all_searched_images)}\n\n"
    
    # Show classification results
    result_text += "Classification Results:\n"
    for i, image_results in enumerate(results):
        result_text += f"\nImage {i+1}:\n"
        for category, confidence in image_results[:5]:  # Show top 5 results
            result_text += f"  {category}: {confidence:.3f} ({confidence*100:.1f}%)\n"
    
    return result_text, all_searched_images

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="CLIP Image Classifier", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🖼️ CLIP Image Classifier")
        gr.Markdown("Upload images or search the web and classify them using OpenAI's CLIP model. Choose from organized category types or add your own.")
        
        with gr.Tabs():
            # Upload Images Tab
            with gr.Tab("📁 Upload Images"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("## Configuration")
                        
                        # Category type selection
                        category_types = gr.CheckboxGroup(
                            label="Category Types",
                            choices=list(classifier.category_types.keys()),
                            value=["animals", "vehicles"],  # Default selection
                            info="Select which types of objects to identify"
                        )
                        
                        custom_categories = gr.Textbox(
                            label="Custom Categories (comma-separated)",
                            placeholder="e.g., sports car, mountain, ocean, building",
                            info="Add your own categories separated by commas"
                        )
                        
                        # Show selected categories
                        selected_categories_display = gr.Textbox(
                            label="Selected Categories",
                            lines=8,
                            interactive=False,
                            value="Select category types above to see available categories"
                        )
                        
                        # Update display when types change
                        def update_categories_display(selected_types, custom_cats):
                            categories = get_selected_categories(selected_types, custom_cats)
                            if categories:
                                return "\n".join(categories[:50]) + ("..." if len(categories) > 50 else "")
                            return "No categories selected"
                        
                        category_types.change(
                            fn=update_categories_display,
                            inputs=[category_types, custom_categories],
                            outputs=selected_categories_display
                        )
                        
                        custom_categories.change(
                            fn=update_categories_display,
                            inputs=[category_types, custom_categories],
                            outputs=selected_categories_display
                        )
                        
                    with gr.Column(scale=2):
                        gr.Markdown("## Upload Images")
                        
                        # Image upload
                        uploaded_images = gr.File(
                            label="Upload Images",
                            file_count="multiple",
                            file_types=["image"]
                        )
                        
                        # Classify button
                        classify_btn = gr.Button("🔍 Classify Images", variant="primary")
                        
                        # Results
                        results = gr.Textbox(
                            label="Classification Results",
                            lines=15,
                            max_lines=20,
                            interactive=False
                        )
                        
                        # Image gallery for uploaded images
                        image_gallery = gr.Gallery(
                            label="Classified Images",
                            show_label=True,
                            elem_id="gallery",
                            columns=3,
                            rows=2,
                            height="auto"
                        )
                
                # Event handlers
                classify_btn.click(
                    fn=classify_uploaded_images,
                    inputs=[uploaded_images, custom_categories, category_types],
                    outputs=[results, image_gallery]
                )
            
            # Search Images Tab
            with gr.Tab("🔍 Search Images"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("## Search Configuration")
                        
                        # Search query
                        search_query = gr.Textbox(
                            label="Search Query (comma-separated)",
                            placeholder="e.g., cute cats, vintage cars, delicious food",
                            info="Enter multiple search terms separated by commas. Each term will be searched separately."
                        )
                        
                        # Number of images
                        num_images = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="Total Number of Images",
                            info="Total images to search for (distributed across all search terms)"
                        )
                        
                        # Category type selection (same as upload tab)
                        search_category_types = gr.CheckboxGroup(
                            label="Category Types",
                            choices=list(classifier.category_types.keys()),
                            value=["animals", "vehicles"],
                            info="Select which types of objects to identify"
                        )
                        
                        search_custom_categories = gr.Textbox(
                            label="Custom Categories (comma-separated)",
                            placeholder="e.g., sports car, mountain, ocean, building",
                            info="Add your own categories separated by commas"
                        )
                        
                    with gr.Column(scale=2):
                        gr.Markdown("## Search and Classify")
                        
                        # Search button
                        search_btn = gr.Button("🔍 Search & Classify", variant="primary")
                        
                        # Search results
                        search_results = gr.Textbox(
                            label="Search & Classification Results",
                            lines=15,
                            max_lines=20,
                            interactive=False
                        )
                        
                        # Image gallery for searched images
                        search_image_gallery = gr.Gallery(
                            label="Searched & Classified Images",
                            show_label=True,
                            elem_id="search_gallery",
                            columns=3,
                            rows=2,
                            height="auto"
                        )
                
                # Event handlers
                search_btn.click(
                    fn=search_and_classify_images,
                    inputs=[search_query, num_images, search_custom_categories, search_category_types],
                    outputs=[search_results, search_image_gallery]
                )
    
    return app

if __name__ == "__main__":
    # Create and launch the app
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True
    )
