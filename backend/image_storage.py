"""
Image Storage Module
Handles uploading user-submitted images to GCP and maintaining image metadata in Excel.
"""

import os
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

# Import GCP utilities
try:
    from . import gcp_utils
except ImportError:
    import gcp_utils


# GCP bucket configuration
BUCKET_NAME = "bot-dump"
IMAGE_FOLDER_PREFIX = "loc_wwtp_img"
EXCEL_FOLDER_PREFIX = "results_wwtp_loc"
MASTER_EXCEL_NAME = "location_images_master.xlsx"


def create_image_folder_name(submission_id: str) -> str:
    """
    Create folder name for storing user images.
    
    Format: Submission_{submission_id}
    Example: Submission_SUB20260123_071530_1234
    
    Args:
        submission_id: Unique submission ID (from excel_utils)
        
    Returns:
        str: Folder name
    """
    return f"Submission_{submission_id}"


def upload_user_image_to_gcp(
    local_path: str,
    folder_name: str,
    image_index: int,
    bucket_name: str = BUCKET_NAME,
    max_retries: int = 2
) -> Dict[str, any]:
    """
    Upload user image to GCP bucket with retry logic.
    
    Args:
        local_path: Path to local image file
        folder_name: Folder name in bucket (e.g., "Submission_SUB20260123_071530_1234")
        image_index: Image number (1, 2, 3)
        bucket_name: GCP bucket name
        max_retries: Maximum retry attempts (default: 2)
        
    Returns:
        dict: {
            'success': bool,
            'gcs_uri': str or None,
            'blob_path': str or None,
            'error': str or None
        }
    """
    # Construct blob path
    image_name = f"image_{image_index}.jpg"
    blob_path = f"{IMAGE_FOLDER_PREFIX}/{folder_name}/{image_name}"
    
    print(f"\n{'=' * 80}")
    print(f"Uploading User Image to GCP")
    print(f"{'=' * 80}")
    print(f"Local path: {local_path}")
    print(f"Bucket: {bucket_name}")
    print(f"Blob path: {blob_path}")
    
    for attempt in range(max_retries + 1):
        try:
            print(f"\nAttempt {attempt + 1}/{max_retries + 1}...")
            
            # Upload to GCP
            gcs_uri = gcp_utils.upload_image_to_bucket(
                local_path=local_path,
                bucket_name=bucket_name,
                blob_path=blob_path,
                max_retries=1  # GCP utils has its own retry, we just use 1 here
            )
            
            if gcs_uri:
                print(f"✓ Upload successful: {gcs_uri}")
                return {
                    'success': True,
                    'gcs_uri': gcs_uri,
                    'blob_path': blob_path,
                    'error': None
                }
        
        except Exception as e:
            error_msg = f"Upload failed: {str(e)}"
            print(f"⚠️ Attempt {attempt + 1} failed: {error_msg}")
            
            # If we have more retries, continue
            if attempt < max_retries:
                print(f"  Retrying...")
                continue
            
            # All retries exhausted
            print(f"❌ Upload failed after {max_retries + 1} attempts")
            return {
                'success': False,
                'gcs_uri': None,
                'blob_path': blob_path,
                'error': error_msg
            }
    
    return {
        'success': False,
        'gcs_uri': None,
        'blob_path': blob_path,
        'error': "Upload failed after all retries"
    }


def save_location_images_metadata(
    submission_id: str,
    phone_number: str,
    image_folder_path: Optional[str],
    output_dir: str = "Data"
) -> Dict[str, any]:
    """
    Save or append image metadata to master Excel file.
    
    This creates/updates a single master Excel file with all submissions.
    
    Args:
        submission_id: Unique submission ID
        phone_number: User's WhatsApp phone number
        image_folder_path: GCS path to image folder, or None if upload failed
        output_dir: Local directory to save Excel
        
    Returns:
        dict: {
            'success': bool,
            'local_path': str,
            'excel_name': str,
            'error': str or None
        }
    """
    try:
        print(f"\n{'=' * 80}")
        print(f"Saving Location Images Metadata to Excel")
        print(f"{'=' * 80}")
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Excel file path
        excel_path = output_path / MASTER_EXCEL_NAME
        
        # Create new row data
        new_row = pd.DataFrame([{
            'Submission_ID': submission_id,
            'Phone_Number': phone_number,
            'Image_Folder_Path': image_folder_path if image_folder_path else ""
        }])
        
        # Check if Excel exists
        if excel_path.exists():
            print(f"✓ Appending to existing Excel: {excel_path}")
            # Read existing data
            existing_df = pd.read_excel(excel_path)
            # Append new row
            updated_df = pd.concat([existing_df, new_row], ignore_index=True)
        else:
            print(f"✓ Creating new Excel: {excel_path}")
            updated_df = new_row
        
        # Save to Excel
        updated_df.to_excel(excel_path, index=False)
        
        print(f"✓ Excel saved with {len(updated_df)} total rows")
        print(f"  New entry: {submission_id} | {phone_number}")
        print(f"  Image folder: {image_folder_path if image_folder_path else 'UPLOAD FAILED'}")
        
        return {
            'success': True,
            'local_path': str(excel_path),
            'excel_name': MASTER_EXCEL_NAME,
            'error': None
        }
    
    except Exception as e:
        error_msg = f"Failed to save Excel: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            'success': False,
            'local_path': None,
            'excel_name': MASTER_EXCEL_NAME,
            'error': error_msg
        }


def upload_location_metadata_to_gcp(
    local_excel_path: str,
    bucket_name: str = BUCKET_NAME
) -> Optional[str]:
    """
    Upload location images metadata Excel to GCP bucket.
    
    Args:
        local_excel_path: Path to local Excel file
        bucket_name: GCP bucket name
        
    Returns:
        str: GCS URI if successful, None if failed
    """
    try:
        print(f"\n{'=' * 80}")
        print(f"Uploading Metadata Excel to GCP")
        print(f"{'=' * 80}")
        
        # Blob path in bucket
        blob_path = f"{EXCEL_FOLDER_PREFIX}/{MASTER_EXCEL_NAME}"
        
        print(f"Local path: {local_excel_path}")
        print(f"Bucket: {bucket_name}")
        print(f"Blob path: {blob_path}")
        
        # Upload to GCP
        gcs_uri = gcp_utils.upload_image_to_bucket(
            local_path=local_excel_path,
            bucket_name=bucket_name,
            blob_path=blob_path,
            max_retries=3
        )
        
        if gcs_uri:
            print(f"✓ Excel uploaded to GCP: {gcs_uri}")
            return gcs_uri
        else:
            print(f"⚠️ Excel upload to GCP failed")
            return None
    
    except Exception as e:
        print(f"❌ Error uploading Excel to GCP: {str(e)}")
        return None


def process_user_image_upload(
    local_image_path: str,
    submission_id: str,
    image_index: int,
    max_retries: int = 2
) -> Dict[str, any]:
    """
    Complete workflow for processing a single user image upload.
    
    This is a convenience function that:
    1. Creates the folder name
    2. Uploads the image to GCP with retries
    
    Args:
        local_image_path: Path to downloaded image
        submission_id: Unique submission ID
        image_index: Image number (1, 2, 3)
        max_retries: Maximum retry attempts for GCP upload
        
    Returns:
        dict: {
            'success': bool,
            'gcs_uri': str or None,
            'folder_name': str,
            'blob_path': str,
            'error': str or None
        }
    """
    # Create folder name
    folder_name = create_image_folder_name(submission_id)
    
    # Upload to GCP
    upload_result = upload_user_image_to_gcp(
        local_path=local_image_path,
        folder_name=folder_name,
        image_index=image_index,
        max_retries=max_retries
    )
    
    # Add folder name to result
    upload_result['folder_name'] = folder_name
    
    return upload_result


# Test function
def test_image_storage():
    """Test image storage utilities."""
    print("=" * 80)
    print("Testing Image Storage Module")
    print("=" * 80)
    
    # Test folder name creation
    test_submission_id = "SUB20260123_071530_1234"
    folder_name = create_image_folder_name(test_submission_id)
    print(f"\n✓ Folder name: {folder_name}")
    
    # Test Excel creation
    result = save_location_images_metadata(
        submission_id=test_submission_id,
        phone_number="whatsapp:+1234567890",
        image_folder_path=f"gs://bot-dump/loc_wwtp_img/{folder_name}",
        output_dir="Data"
    )
    
    if result['success']:
        print(f"\n✓ Excel created successfully: {result['local_path']}")
    else:
        print(f"\n❌ Excel creation failed: {result['error']}")
    
    print("\nImage storage module ready!")


if __name__ == "__main__":
    test_image_storage()
