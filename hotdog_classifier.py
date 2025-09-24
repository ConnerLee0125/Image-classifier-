import os
import requests
import json
import time
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm import tqdm
import urllib.parse

# Enable interactive mode for matplotlib
plt.ion()

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def search_images_ddg(query, max_images=50):
    """Search for images using DuckDuckGo"""
    import re
    import json
    
    print(f"Searching for: {query}")
    
    try:
        # Initialize session
        session = requests.Session()
        
        # Headers for initial request
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:66.0) Gecko/20100101 Firefox/66.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://duckduckgo.com/",
            "DNT": "1",
        }
        
        # Parameters for initial search
        params = {
            "q": query,
            "t": "ffsb",
            "iar": "images",
            "iaf": "",  # No filters for now
        }
        
        # Get the search page to extract vqd token
        req = session.get("https://duckduckgo.com", params=params, headers=headers)
        
        # Extract vqd token
        vqd_match = re.search(r'vqd=([\d-]+)&', req.text, re.M | re.I)
        if not vqd_match:
            print("vqd token not found. Status code: %d" % req.status_code)
            return []
        
        vqd = vqd_match.group(1)
        print(f"Found vqd token: {vqd}")
        
        # Update headers for API request
        session.headers.update({
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "en-US,en;q=0.5",
            "DNT": "1",
            "Referer": "https://duckduckgo.com/",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:66.0) Gecko/20100101 Firefox/66.0",
            "X-Requested-With": "XMLHttpRequest",
        })
        
        # Search parameters
        search_params = {
            "f": "",
            "l": "wt-wt",
            "o": "json",
            "p": -1,
            "q": query,
            "s": 0,
            "iaf": "",
            "vqd": vqd,
            "ex": -2,  # Safe search off
        }
        
        image_urls = []
        n = 0
        
        while n < max_images:
            search_params["s"] = n
            req = session.get("https://duckduckgo.com/i.js", params=search_params)
            
            if req.status_code != 200:
                print(f"HTTP status {req.status_code}")
                break
            
            try:
                data = req.json()
                results = data.get("results", [])
                
                if not results:
                    break
                
                for obj in results:
                    if len(image_urls) >= max_images:
                        break
                    
                    image_url = obj.get('image')
                    if image_url:
                        image_urls.append(image_url)
                
                if len(image_urls) >= max_images:
                    break
                    
            except json.JSONDecodeError:
                print("Failed to parse JSON response")
                break
            
            n += 50
        
        print(f"Found {len(image_urls)} images for '{query}'")
        return image_urls
        
    except Exception as e:
        print(f"Error in DuckDuckGo search: {e}")
        return []


def download_image(url, filepath, add_extension=True):
    """Download an image from URL with proper format detection"""
    import tempfile
    import shutil
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    
    try:
        print(f"Downloading ... {url}")
        
        # First, try to get the image with a HEAD request to check if it's accessible
        try:
            head_req = requests.head(url, headers=headers, timeout=5, allow_redirects=True)
            if head_req.status_code not in [200, 301, 302]:
                print(f"    HTTP status: {head_req.status_code}")
                return False
        except:
            pass  # Continue with GET request even if HEAD fails
        
        # Now try to download the image
        req = requests.get(url, stream=True, headers=headers, timeout=15, allow_redirects=True)
        if req.status_code != 200:
            print(f"    HTTP status: {req.status_code}")
            return False
        
        # Check content type
        content_type = req.headers.get('content-type', '').lower()
        if not any(img_type in content_type for img_type in ['image/', 'application/octet-stream']):
            print(f"    Not an image (content-type: {content_type})")
            return False
        
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            try:
                # Download the content
                for chunk in req.iter_content(chunk_size=8192):
                    if chunk:
                        temp.write(chunk)
                temp.flush()
                
                # Try to open and verify the image
                try:
                    with Image.open(temp.name) as img:
                        # Convert to RGB if necessary
                        if img.mode in ('RGBA', 'LA', 'P'):
                            img = img.convert('RGB')
                        
                        ext = img.format.lower() if img.format else 'jpg'
                        print(f"    image type: {ext}, size: {img.size}")
                        
                        # Save with proper extension
                        if add_extension and ext:
                            final_path = f"{filepath}.{ext}"
                        else:
                            final_path = filepath
                        
                        # Save the processed image
                        img.save(final_path, format=ext.upper() if ext else 'JPEG', quality=95)
                        return True
                        
                except Exception as img_error:
                    print(f"    cannot process image: {img_error}")
                    return False
                    
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp.name)
                except:
                    pass
                
    except requests.exceptions.ConnectionError:
        print(f"    Connection error: {url}")
        return False
    except requests.exceptions.Timeout:
        print(f"    Timeout: {url}")
        return False
    except Exception as e:
        print(f"    Error downloading {url}: {e}")
        return False

def download_images(dest_path, urls):
    """Download multiple images to destination path"""
    dest_path = Path(dest_path)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    downloaded = 0
    for i, url in enumerate(urls):
        filename = f"image_{i}"
        filepath = dest_path / filename
        
        if download_image(url, filepath, add_extension=True):
            downloaded += 1
            time.sleep(0.1)  # Be respectful to the server
    
    print(f"Downloaded {downloaded} images to {dest_path}")
    return downloaded

def get_image_files(path):
    """Get all image files from directory"""
    path = Path(path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(path.glob(f'**/*{ext}'))
        image_files.extend(path.glob(f'**/*{ext.upper()}'))
    
    return image_files

def verify_images(image_paths):
    """Verify that images can be opened"""
    failed = []
    for path in image_paths:
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            failed.append(path)
    return failed

def resize_images(input_path, max_size=400):
    """Resize images to maximum size while maintaining aspect ratio"""
    input_path = Path(input_path)
    
    for img_path in input_path.glob('**/*'):
        if img_path.is_file() and img_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            try:
                with Image.open(img_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize while maintaining aspect ratio
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    img.save(img_path, 'JPEG', quality=95)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

def create_data_loaders(data_path, batch_size=32, train_split=0.8):
    """Create training and validation data loaders"""
    data_path = Path(data_path)
    
    # Get all image files and their labels
    image_files = get_image_files(data_path)
    labels = []
    
    for img_path in image_files:
        # Label is determined by parent directory
        parent_dir = img_path.parent.name
        if parent_dir == 'hotdog':
            labels.append(0)
        else:  # not_hotdog
            labels.append(1)
    
    # Split into train and validation
    n_total = len(image_files)
    n_train = int(n_total * train_split)
    
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_files = [image_files[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_files = [image_files[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ImageDataset(train_files, train_labels, train_transform)
    val_dataset = ImageDataset(val_files, val_labels, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, num_epochs=4, learning_rate=0.001):
    """Train the model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Val Accuracy: {val_accuracy:.2f}%')
        
        scheduler.step()
    
    return train_losses, val_losses, val_accuracies


def plot_confusion_matrix(model, val_loader, class_names=['hotdog', 'not_hotdog']):
    """Plot confusion matrix"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.draw()  # Force the plot to draw
    
    return cm

def plot_top_losses(model, val_loader, top_k=5):
    """Plot images with highest loss"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    losses = []
    images = []
    labels = []
    predictions = []
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for batch_images, batch_labels in val_loader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            outputs = model(batch_images)
            batch_losses = criterion(outputs, batch_labels)
            
            losses.extend(batch_losses.cpu().numpy())
            images.extend(batch_images.cpu())
            labels.extend(batch_labels.cpu().numpy())
            predictions.extend(torch.max(outputs, 1)[1].cpu().numpy())
    
    # Get top k losses
    top_indices = np.argsort(losses)[-top_k:]
    
    fig, axes = plt.subplots(1, top_k, figsize=(15, 3))
    if top_k == 1:
        axes = [axes]
    
    for i, idx in enumerate(top_indices):
        img = images[idx]
        # Denormalize image for display
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img = torch.clamp(img, 0, 1)
        
        axes[i].imshow(img.permute(1, 2, 0))
        true_label = 'hotdog' if labels[idx] == 0 else 'not_hotdog'
        pred_label = 'hotdog' if predictions[idx] == 0 else 'not_hotdog'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nLoss: {losses[idx]:.3f}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('top_losses.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.draw()  # Force the plot to draw

def export_model(model, filepath='hotdog_classifier.pth'):
    """Export the trained model"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': 'resnet18',
        'num_classes': 2,
        'class_names': ['hotdog', 'not_hotdog']
    }, filepath)
    print(f"Model exported to {filepath}")

def create_sample_images(path, num_images=20):
    """Create sample images for testing if downloads fail"""
    from PIL import Image, ImageDraw
    import random
    
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_images} sample images in {path}")
    
    for i in range(num_images):
        # Create a random colored image
        width, height = 400, 300
        img = Image.new('RGB', (width, height), color=(
            random.randint(50, 255),
            random.randint(50, 255), 
            random.randint(50, 255)
        ))
        
        # Add some random shapes
        draw = ImageDraw.Draw(img)
        for _ in range(random.randint(3, 8)):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = random.randint(0, width)
            y2 = random.randint(0, height)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.ellipse([x1, y1, x2, y2], fill=color)
        
        # Save the image
        img.save(path / f"sample_{i}.jpg", "JPEG", quality=95)
    
    print(f"Created {num_images} sample images")

def main():
    # Configuration
    categories = ['hotdog', 'not_hotdog']
    path = Path('testingpath')
    searches = ['cat', 'forest', 'bathroom', 'computer']
    
    # Create directories
    if not path.exists():
        path.mkdir()
    
    for category in categories:
        dest = path / category
        dest.mkdir(exist_ok=True)
        
        if category == 'hotdog':
            # Search for hotdog images
            results = search_images_ddg(category, max_images=40)  # Reduced to match not_hotdog
            downloaded = download_images(dest, results)
            if downloaded == 0:
                print(f"No images downloaded for {category}, creating sample images...")
                create_sample_images(dest, num_images=20)
        else:
            # Search for not_hotdog images
            total_downloaded = 0
            for search_term in searches:
                results = search_images_ddg(search_term, max_images=10)
                downloaded = download_images(dest, results)
                total_downloaded += downloaded
            
            if total_downloaded == 0:
                print(f"No images downloaded for {category}, creating sample images...")
                create_sample_images(dest, num_images=20)
    
    # Verify and clean up images
    print("Verifying images...")
    fns = get_image_files(path)
    failed = verify_images(fns)
    print(f"Found {len(failed)} failed images")
    
    for failed_path in failed:
        failed_path.unlink()
    
    # Check if we have any images left
    remaining_images = get_image_files(path)
    if len(remaining_images) == 0:
        print("No images found! Creating sample images for both categories...")
        create_sample_images(path / 'hotdog', num_images=20)
        create_sample_images(path / 'not_hotdog', num_images=20)
    
    # Resize images
    print("Resizing images...")
    for category in categories:
        resize_images(path / category, max_size=400)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(path)
    
    # Load pre-trained ResNet18
    print("Loading pre-trained ResNet18...")
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # 2 classes: hotdog, not_hotdog
    
    # Train the model
    print("Training model...")
    train_losses, val_losses, val_accuracies = train_model(model, train_loader, val_loader)
    
    # Plot results
    print("Plotting confusion matrix...")
    plot_confusion_matrix(model, val_loader)
    
    print("Plotting top losses...")
    plot_top_losses(model, val_loader, top_k=5)
    
    # Export model
    print("Exporting model...")
    export_model(model)
    
    print("Training complete!")
    print("Plots are displayed in separate windows. Press Enter to close the program...")
    
    # Keep the program running until user presses Enter
    input()

if __name__ == "__main__":
    main()
