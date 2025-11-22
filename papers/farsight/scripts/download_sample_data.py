#!/usr/bin/env python3
"""
Download sample images for testing FarSight MLLM.
Downloads images from picsum.photos (Lorem Picsum) - a free image service.
"""
import os
import urllib.request
import ssl
from pathlib import Path

# Disable SSL certificate verification globally
ssl._create_default_https_context = ssl._create_unverified_context

def download_sample_images(output_dir, num_images=10):
    """Download sample images from picsum.photos"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {num_images} sample images to {output_dir}...")
    
    # Different image sizes and IDs for variety
    image_specs = [
        (100, 640, 480),   # ID, width, height
        (200, 640, 480),
        (300, 640, 480),
        (400, 640, 480),
        (500, 640, 480),
        (600, 640, 480),
        (700, 640, 480),
        (800, 640, 480),
        (900, 640, 480),
        (1000, 640, 480),
    ]
    
    success_count = 0
    for i, (img_id, width, height) in enumerate(image_specs[:num_images], 1):
        url = f"https://picsum.photos/id/{img_id}/{width}/{height}"
        output_file = output_path / f"sample_image_{i:03d}.jpg"
        
        try:
            print(f"  [{i}/{num_images}] Downloading {output_file.name}...")
            urllib.request.urlretrieve(url, output_file)
            print(f"  ✓ Saved to {output_file}")
            success_count += 1
        except Exception as e:
            print(f"  ✗ Failed to download image {i}: {e}")
    
    print(f"\n✓ Successfully downloaded {success_count}/{num_images} images to {output_dir}")
    return list(output_path.glob("*.jpg"))

if __name__ == "__main__":
    # Set the data directory relative to this script
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    images_dir = project_root / "data" / "images"
    
    downloaded_files = download_sample_images(images_dir, num_images=10)
    print(f"\nTotal images in {images_dir}: {len(downloaded_files)}")
