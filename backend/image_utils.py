"""
Image Utilities Module
Handles downloading images from Twilio CDN and validating image files.
"""

import os
import requests
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image


# Maximum image size in bytes (5MB - Twilio's limit)
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB

# Allowed image formats
ALLOWED_FORMATS = ['JPEG', 'PNG', 'JPG']


def download_image_from_twilio(
    media_url: str,
    local_path: str,
    auth: Tuple[str, str],
    max_retries: int = 2
) -> Tuple[bool, Optional[str]]:
    """
    Download image from Twilio CDN URL with retry logic.
    
    Args:
        media_url: Twilio media URL (from MediaUrl0 field)
        local_path: Local path to save the downloaded image
        auth: Tuple of (account_sid, auth_token) for Twilio authentication
        max_retries: Maximum number of retry attempts (default: 2)
        
    Returns:
        Tuple[bool, Optional[str]]: (success, error_message)
            - (True, None) if successful
            - (False, error_message) if failed
    """
    # Ensure parent directory exists
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(max_retries + 1):
        try:
            print(f"Downloading image from Twilio (attempt {attempt + 1}/{max_retries + 1})...")
            print(f"  URL: {media_url}")
            print(f"  Local path: {local_path}")
            
            # Download with authentication
            response = requests.get(
                media_url,
                auth=auth,
                stream=True,
                timeout=30
            )
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                return False, f"Invalid content type: {content_type}. Expected image."
            
            # Download and save
            total_size = 0
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        total_size += len(chunk)
                        # Check size limit
                        if total_size > MAX_IMAGE_SIZE:
                            os.remove(local_path)  # Clean up
                            return False, f"Image exceeds {MAX_IMAGE_SIZE / 1024 / 1024}MB limit"
                        f.write(chunk)
            
            print(f"✓ Downloaded {total_size / 1024:.2f} KB")
            
            # Validate the downloaded file
            is_valid, error = validate_image_file(local_path)
            if not is_valid:
                os.remove(local_path)  # Clean up invalid file
                return False, error
            
            return True, None
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Download failed: {str(e)}"
            print(f"⚠️ Attempt {attempt + 1} failed: {error_msg}")
            
            # If we have more retries, continue
            if attempt < max_retries:
                print(f"  Retrying...")
                continue
            
            # All retries exhausted
            return False, error_msg
        
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"❌ Error during download: {error_msg}")
            return False, error_msg
    
    return False, "Download failed after all retries"


def validate_image_file(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that a file is a valid image with allowed format.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
            - (True, None) if valid
            - (False, error_message) if invalid
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return False, "File is empty"
        
        if file_size > MAX_IMAGE_SIZE:
            return False, f"File exceeds {MAX_IMAGE_SIZE / 1024 / 1024}MB limit"
        
        # Try to open and validate with PIL
        with Image.open(file_path) as img:
            # Check format
            if img.format not in ALLOWED_FORMATS:
                return False, f"Invalid format: {img.format}. Allowed: {', '.join(ALLOWED_FORMATS)}"
            
            # Verify image is not corrupted by loading it
            img.verify()
        
        # Re-open to get dimensions (verify() closes the file)
        with Image.open(file_path) as img:
            width, height = img.size
            print(f"✓ Valid image: {img.format}, {width}x{height}, {file_size / 1024:.2f} KB")
        
        return True, None
        
    except Exception as e:
        return False, f"Image validation failed: {str(e)}"


def get_twilio_credentials() -> Tuple[str, str]:
    """
    Get Twilio credentials from environment variables.
    
    Returns:
        Tuple[str, str]: (account_sid, auth_token)
        
    Raises:
        ValueError: If credentials are not found in environment
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    
    if not account_sid or not auth_token:
        raise ValueError(
            "Twilio credentials not found. "
            "Please set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in .env file"
        )
    
    return account_sid, auth_token


# Test function
def test_image_utils():
    """Test image utilities with a sample download."""
    print("=" * 80)
    print("Testing Image Utils")
    print("=" * 80)
    
    # Test validation with a real image
    test_image_path = "Data/test_image.jpg"
    
    if os.path.exists(test_image_path):
        is_valid, error = validate_image_file(test_image_path)
        if is_valid:
            print(f"✓ Test image is valid")
        else:
            print(f"❌ Test image validation failed: {error}")
    else:
        print(f"⚠️ No test image found at {test_image_path}")
    
    print("\nImage utils module ready!")


if __name__ == "__main__":
    test_image_utils()
