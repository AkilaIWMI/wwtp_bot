"""
Google Cloud Storage (GCS) Utilities Module
Handles uploading and downloading images to/from GCP bucket with timestamp-based naming.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from google.cloud import storage
from google.api_core import retry
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


def get_gcs_credentials_path() -> Optional[str]:
    """
    Get the path to GCS service account credentials from environment.
    
    Returns:
        str: Path to service account JSON key file, or None if not set
    """
    return os.getenv("GCP_SERVICE_ACCOUNT_KEY_PATH")


def initialize_gcs_client() -> storage.Client:
    """
    Initialize Google Cloud Storage client with credentials.
    
    Returns:
        storage.Client: Initialized GCS client
        
    Raises:
        ValueError: If credentials are not found or invalid
    """
    credentials_path = get_gcs_credentials_path()
    
    if not credentials_path:
        raise ValueError(
            "GCP_SERVICE_ACCOUNT_KEY_PATH not found in environment variables. "
            "Please add it to your .env file."
        )
    
    if not os.path.exists(credentials_path):
        raise ValueError(
            f"Service account key file not found at: {credentials_path}. "
            "Please ensure the path is correct and the file exists."
        )
    
    try:
        # Initialize client with explicit credentials
        client = storage.Client.from_service_account_json(credentials_path)
        print(f"✓ GCS client initialized successfully")
        return client
    except Exception as e:
        raise ValueError(f"Failed to initialize GCS client: {str(e)}")


def generate_timestamp() -> str:
    """
    Generate timestamp string in format YYYYMMDD_HHMMSS.
    
    Returns:
        str: Timestamp string (e.g., "20260122_113045")
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def upload_image_to_bucket(
    local_path: str,
    bucket_name: str,
    blob_path: str,
    max_retries: int = 3
) -> str:
    """
    Upload an image file to Google Cloud Storage bucket.
    
    Args:
        local_path: Path to local image file
        bucket_name: Name of GCS bucket (e.g., "bot-dump")
        blob_path: Destination path in bucket (e.g., "meta_data/image.jpg")
        max_retries: Maximum number of retry attempts
        
    Returns:
        str: GCS URI in format gs://bucket-name/blob-path
        
    Raises:
        FileNotFoundError: If local file doesn't exist
        Exception: If upload fails after retries
    """
    # Verify local file exists
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local file not found: {local_path}")
    
    try:
        # Initialize client
        client = initialize_gcs_client()
        
        # Get bucket
        bucket = client.bucket(bucket_name)
        
        # Create blob (file object in GCS)
        blob = bucket.blob(blob_path)
        
        # Upload with retry logic
        print(f"Uploading {local_path} to gs://{bucket_name}/{blob_path}...")
        
        # Use Google's built-in retry decorator
        @retry.Retry(predicate=retry.if_exception_type(Exception))
        def _upload():
            blob.upload_from_filename(local_path)
        
        _upload()
        
        # Construct GCS URI
        gcs_uri = f"gs://{bucket_name}/{blob_path}"
        
        print(f"✓ Upload successful: {gcs_uri}")
        return gcs_uri
        
    except Exception as e:
        print(f"✗ Upload failed: {str(e)}")
        raise


def download_image_from_bucket(
    bucket_name: str,
    blob_path: str,
    local_path: str,
    max_retries: int = 3
) -> str:
    """
    Download an image file from Google Cloud Storage bucket.
    
    Args:
        bucket_name: Name of GCS bucket (e.g., "bot-dump")
        blob_path: Source path in bucket (e.g., "meta_data/image.jpg")
        local_path: Destination path for downloaded file
        max_retries: Maximum number of retry attempts
        
    Returns:
        str: Path to downloaded local file
        
    Raises:
        Exception: If download fails after retries
    """
    try:
        # Initialize client
        client = initialize_gcs_client()
        
        # Get bucket
        bucket = client.bucket(bucket_name)
        
        # Get blob
        blob = bucket.blob(blob_path)
        
        # Check if blob exists
        if not blob.exists():
            raise FileNotFoundError(f"File not found in bucket: gs://{bucket_name}/{blob_path}")
        
        # Create local directory if needed
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download with retry logic
        print(f"Downloading gs://{bucket_name}/{blob_path} to {local_path}...")
        
        @retry.Retry(predicate=retry.if_exception_type(Exception))
        def _download():
            blob.download_to_filename(local_path)
        
        _download()
        
        print(f"✓ Download successful: {local_path}")
        return local_path
        
    except Exception as e:
        print(f"✗ Download failed: {str(e)}")
        raise


def get_bucket_image_url(bucket_name: str, blob_path: str) -> str:
    """
    Get GCS URI for an image in the bucket.
    
    Args:
        bucket_name: Name of GCS bucket
        blob_path: Path to blob in bucket
        
    Returns:
        str: GCS URI in format gs://bucket-name/blob-path
    """
    return f"gs://{bucket_name}/{blob_path}"


def list_bucket_images(
    bucket_name: str,
    prefix: str = "",
    max_results: int = 100
) -> list:
    """
    List images in a GCS bucket with optional prefix filter.
    
    Args:
        bucket_name: Name of GCS bucket
        prefix: Optional prefix to filter results (e.g., "meta_data/")
        max_results: Maximum number of results to return
        
    Returns:
        list: List of blob names (file paths in bucket)
    """
    try:
        client = initialize_gcs_client()
        bucket = client.bucket(bucket_name)
        
        # List blobs with prefix
        blobs = bucket.list_blobs(prefix=prefix, max_results=max_results)
        
        # Extract names
        blob_names = [blob.name for blob in blobs]
        
        print(f"✓ Found {len(blob_names)} files in gs://{bucket_name}/{prefix}")
        return blob_names
        
    except Exception as e:
        print(f"✗ Failed to list bucket contents: {str(e)}")
        raise


def parse_gcs_uri(gcs_uri: str) -> Tuple[str, str]:
    """
    Parse a GCS URI into bucket name and blob path.
    
    Args:
        gcs_uri: GCS URI in format gs://bucket-name/blob-path
        
    Returns:
        tuple: (bucket_name, blob_path)
        
    Raises:
        ValueError: If URI format is invalid
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI format: {gcs_uri}. Must start with 'gs://'")
    
    # Remove gs:// prefix
    path = gcs_uri[5:]
    
    # Split into bucket and blob path
    parts = path.split("/", 1)
    
    if len(parts) != 2:
        raise ValueError(f"Invalid GCS URI format: {gcs_uri}. Must be gs://bucket-name/blob-path")
    
    bucket_name, blob_path = parts
    return bucket_name, blob_path


def is_gcs_uri(path: str) -> bool:
    """
    Check if a path is a GCS URI.
    
    Args:
        path: Path to check
        
    Returns:
        bool: True if path starts with gs://, False otherwise
    """
    return path.startswith("gs://")


# Test function for standalone execution
def test_gcp_utils():
    """
    Test GCP utilities with a sample image upload/download.
    """
    print("=" * 80)
    print("GCP UTILITIES TEST")
    print("=" * 80)
    
    # Check environment variables
    bucket_name = os.getenv("GCP_BUCKET_NAME", "bot-dump")
    bucket_path = os.getenv("GCP_BUCKET_PATH", "meta_data")
    
    print(f"\nConfiguration:")
    print(f"  Bucket: {bucket_name}")
    print(f"  Path: {bucket_path}")
    print(f"  Credentials: {get_gcs_credentials_path()}")
    
    # Test timestamp generation
    print(f"\nGenerated timestamp: {generate_timestamp()}")
    
    # Test GCS client initialization
    try:
        client = initialize_gcs_client()
        print(f"✓ GCS client initialized")
    except Exception as e:
        print(f"✗ Failed to initialize GCS client: {str(e)}")
        print("\nPlease ensure:")
        print("1. GCP_SERVICE_ACCOUNT_KEY_PATH is set in .env")
        print("2. The service account key file exists")
        print("3. The service account has Storage Object Admin permissions")
        return
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_gcp_utils()
