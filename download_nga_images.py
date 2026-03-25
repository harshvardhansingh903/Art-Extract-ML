"""
Download National Gallery of Art images from published_images.csv
Saves images to task2_image_similarity/data/nga_images/
"""

import os
import csv
import requests
from pathlib import Path
from tqdm import tqdm
from urllib.parse import urlparse
import time

# Config
CSV_PATH = r"C:\Users\harsh\Downloads\published_images.csv"
OUTPUT_DIR = Path("task2_image_similarity/data/nga_images")
MAX_IMAGES = 2000
TIMEOUT = 10
RETRY_ATTEMPTS = 2

def setup_output_dir():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory ready: {OUTPUT_DIR.absolute()}")

def load_image_urls(csv_path, max_count=2000):
    """Load image URLs from CSV file."""
    urls = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if idx >= max_count:
                    break
                # NGA uses 'iiifthumburl' for thumbnail images
                url = row.get('iiifthumburl', '')
                if url and url.strip():
                    urls.append(url.strip())
        print(f"✓ Loaded {len(urls)} image URLs from CSV")
        return urls
    except FileNotFoundError:
        print(f"✗ CSV file not found: {csv_path}")
        return []
    except Exception as e:
        print(f"✗ Error reading CSV: {e}")
        return []

def generate_filename(url, index):
    """Generate unique filename from URL."""
    try:
        # Extract filename from URL
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        
        # If no filename, use index
        if not filename or '.' not in filename:
            filename = f"nga_image_{index:05d}.jpg"
        else:
            # Add index prefix to ensure uniqueness
            name, ext = os.path.splitext(filename)
            filename = f"nga_{index:05d}_{name}{ext}"
        
        return filename
    except:
        return f"nga_image_{index:05d}.jpg"

def download_image(url, filepath, attempt=1):
    """Download single image with retry logic."""
    try:
        response = requests.get(url, timeout=TIMEOUT, stream=True)
        response.raise_for_status()
        
        # Check if content-type is image
        content_type = response.headers.get('content-type', '')
        if 'image' not in content_type.lower():
            return False, "Not an image"
        
        # Save image
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True, "Success"
    
    except requests.Timeout:
        if attempt < RETRY_ATTEMPTS:
            time.sleep(1)
            return download_image(url, filepath, attempt + 1)
        return False, "Timeout"
    except requests.HTTPError as e:
        return False, f"HTTP {e.response.status_code}"
    except Exception as e:
        return False, str(e)[:50]

def download_all_images(urls):
    """Download all images with progress bar."""
    successful = 0
    failed = 0
    skipped = 0
    
    print(f"\nDownloading images to {OUTPUT_DIR}...")
    
    for idx, url in enumerate(tqdm(urls, desc="Downloading", unit="img"), 1):
        if not url:
            skipped += 1
            continue
        
        try:
            filename = generate_filename(url, idx)
            filepath = OUTPUT_DIR / filename
            
            # Skip if already exists
            if filepath.exists():
                skipped += 1
                continue
            
            # Download image
            success, message = download_image(url, filepath)
            
            if success:
                successful += 1
            else:
                failed += 1
                # Remove partial file if exists
                if filepath.exists():
                    filepath.unlink()
        
        except Exception as e:
            failed += 1
    
    return successful, failed, skipped

def print_summary(total, successful, failed, skipped):
    """Print download summary."""
    print("\n" + "="*50)
    print("DOWNLOAD SUMMARY")
    print("="*50)
    print(f"Total URLs:      {total}")
    print(f"✓ Successful:    {successful}")
    print(f"✗ Failed:        {failed}")
    print(f"⊘ Skipped:       {skipped}")
    print(f"Success Rate:    {successful/total*100:.1f}%" if total > 0 else "N/A")
    
    # Show downloaded files
    image_files = list(OUTPUT_DIR.glob("*"))
    print(f"\nImages saved:    {len(image_files)}")
    print("="*50)

def main():
    """Main execution."""
    print("National Gallery of Art Image Downloader")
    print("="*50)
    
    # Setup
    setup_output_dir()
    
    # Load URLs
    urls = load_image_urls(CSV_PATH, MAX_IMAGES)
    if not urls:
        print("✗ No URLs found. Exiting.")
        return
    
    # Download
    successful, failed, skipped = download_all_images(urls)
    
    # Summary
    print_summary(len(urls), successful, failed, skipped)

if __name__ == "__main__":
    main()
