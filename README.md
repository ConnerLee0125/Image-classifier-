# Image Classifier

A Gradio-based web application for image classification using OpenAI's CLIP model. This app allows users to upload images or search the web and classify them against organized category types or custom object categories.

## Features

**Image Upload**: Upload multiple images for classification
**Web Image Search**: Search for multiple objects online using duckduckgo
**Organized Categories**: Choose from 5 category types (Animals, Vehicles, Food, Objects, Nature)
**Custom Categories**: Define your own object categories to identify
**Confidence Scores**: View classification results with confidence percentages
**Image Display**: See the actual images being classified alongside results
**Easy to Use**: Tabbed interface with Gradio
**Flexible**: Mix category types and custom categories


## Installation

1. **Clone or download this repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Open your browser** and go to `http://localhost:7861`

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Gradio 4.44+
- Transformers 4.30+
