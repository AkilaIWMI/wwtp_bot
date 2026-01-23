"""
Excel Utilities Module
Handles saving tank measurement data to Excel files and uploading to GCP bucket.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd
from osgeo import gdal

# Import GCP utilities
try:
    from . import gcp_utils
except ImportError:
    import gcp_utils


def generate_submission_id(phone_number: str, timestamp: Optional[datetime] = None) -> str:
    """
    Generate unique submission ID.
    
    Format: SUB{YYYYMMDD}_{HHMMSS}_{phone_last4}
    Example: SUB20260122_203015_7890
    
    Args:
        phone_number: User's phone number (e.g., "+1234567890")
        timestamp: Optional timestamp (defaults to current time)
        
    Returns:
        str: Unique submission ID
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    # Extract last 4 digits of phone number
    digits = re.sub(r'\D', '', phone_number)  # Remove non-digits
    last4 = digits[-4:] if len(digits) >= 4 else digits.zfill(4)
    
    # Format: SUB{YYYYMMDD}_{HHMMSS}_{last4}
    date_part = timestamp.strftime("%Y%m%d")
    time_part = timestamp.strftime("%H%M%S")
    
    submission_id = f"SUB{date_part}_{time_part}_{last4}"
    return submission_id


def sanitize_phone_number(phone_number: str) -> str:
    """
    Sanitize phone number for use in filename.
    
    Replaces special characters with underscores.
    
    Args:
        phone_number: Raw phone number (e.g., "+1234567890")
        
    Returns:
        str: Sanitized phone number (e.g., "_1234567890")
    """
    # Replace non-alphanumeric characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9]', '_', phone_number)
    return sanitized


def get_tank_center_coords(satellite_image_path: str, tank_pixel_coords: tuple) -> tuple:
    """
    Get tank center geographic coordinates from pixel coordinates.
    
    Uses the same CRS transformation as Cal_area_Geodesic.get_geo_coords()
    
    Args:
        satellite_image_path: Path to satellite image (TIFF with geo-referencing)
        tank_pixel_coords: Tuple of (center_x, center_y) in pixels
        
    Returns:
        tuple: (longitude, latitude) or (None, None) if transformation fails
    """
    try:
        # Open GDAL dataset
        ds = gdal.Open(satellite_image_path)
        if ds is None:
            print(f"Error: Could not open satellite image: {satellite_image_path}")
            return None, None
        
        # Get geotransform
        geotransform = ds.GetGeoTransform()
        
        # Extract pixel coordinates
        pixel_x, pixel_y = tank_pixel_coords
        
        # Convert pixel coordinates to source CRS coordinates
        x = geotransform[0] + pixel_x * geotransform[1] + pixel_y * geotransform[2]
        y = geotransform[3] + pixel_x * geotransform[4] + pixel_y * geotransform[5]
        
        # Get source CRS
        from osgeo import osr
        source_crs = osr.SpatialReference()
        source_crs.ImportFromWkt(ds.GetProjection())
        
        # Create target CRS (WGS84 - lat/lon)
        target_crs = osr.SpatialReference()
        target_crs.ImportFromEPSG(4326)  # WGS84
        
        # Create transformation
        transform = osr.CoordinateTransformation(source_crs, target_crs)
        
        # Transform coordinates
        lon, lat, z = transform.TransformPoint(x, y)
        
        # Close dataset
        ds = None
        
        return lon, lat
        
    except Exception as e:
        print(f"Error extracting tank center coordinates: {e}")
        return None, None


def save_tank_data_to_excel(
    phone_number: str,
    wwtp_location: tuple,
    circular_tanks: list,
    tank_heights: dict,
    volume_data: dict,
    satellite_image_path: str,
    output_dir: str = "Data",
    timestamp: Optional[datetime] = None
) -> Optional[str]:
    """
    Save tank measurement data to Excel file (per-user file).
    
    Creates or appends to: tank_measurements_{phone_number}.xlsx
    
    Args:
        phone_number: User's phone number (e.g., "whatsapp:+1234567890")
        wwtp_location: Tuple of (lat, lon) for original bbox center
        circular_tanks: List of circular tank data from YOLO detection
        tank_heights: Dictionary mapping tank_id to height in meters
        volume_data: Dictionary with volume calculations
        satellite_image_path: Path to satellite image for coordinate extraction
        output_dir: Directory to save Excel file (default: "Data")
        timestamp: Optional timestamp (defaults to current time)
        
    Returns:
        str: Path to Excel file, or None if save failed
    """
    try:
        if timestamp is None:
            timestamp = datetime.now()
        
        # Generate submission ID
        submission_id = generate_submission_id(phone_number, timestamp)
        
        # Sanitize phone number for filename
        sanitized_phone = sanitize_phone_number(phone_number)
        excel_filename = f"tank_measurements{sanitized_phone}.xlsx"
        excel_path = os.path.join(output_dir, excel_filename)
        
        # Prepare data rows
        rows = []
        
        for tank_info in volume_data['tank_data']:
            tank_id = tank_info['tank_id']
            
            # Find corresponding tank in circular_tanks to get pixel coordinates
            tank = next((t for t in circular_tanks if t['tank_id'] == tank_id), None)
            
            if tank is None:
                print(f"Warning: Could not find tank {tank_id} in circular_tanks")
                continue
            
            # Get tank center pixel coordinates from measurements
            center_x = tank['measurements']['center_x']
            center_y = tank['measurements']['center_y']
            
            # Extract geographic coordinates for tank center
            tank_lon, tank_lat = get_tank_center_coords(
                satellite_image_path,
                (center_x, center_y)
            )
            
            if tank_lon is None or tank_lat is None:
                print(f"Warning: Could not extract coordinates for tank {tank_id}")
                continue
            
            # Create row data
            row = {
                'Submission_ID': submission_id,
                'Phone_Number': phone_number,
                'Timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'WWTP_Location_Lat': wwtp_location[0],
                'WWTP_Location_Lon': wwtp_location[1],
                'Tank_ID': tank_id,
                'Tank_Center_Lon': round(tank_lon, 6),
                'Tank_Center_Lat': round(tank_lat, 6),
                'Height_m': tank_info['height'],
                'Radius_m': tank_info['radius'],
                'Surface_Area_sqm': tank_info['surface_area'],
                'Volume_m3': tank_info['volume']
            }
            
            rows.append(row)
        
        if not rows:
            print("Error: No valid tank data to save")
            return None
        
        # Create DataFrame
        new_df = pd.DataFrame(rows)
        
        # Check if file already exists
        if os.path.exists(excel_path):
            # Append to existing file
            print(f"Appending to existing Excel file: {excel_path}")
            existing_df = pd.read_excel(excel_path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_excel(excel_path, index=False)
        else:
            # Create new file
            print(f"Creating new Excel file: {excel_path}")
            os.makedirs(output_dir, exist_ok=True)
            new_df.to_excel(excel_path, index=False)
        
        print(f"✓ Tank data saved to Excel: {excel_path}")
        print(f"  Submission ID: {submission_id}")
        print(f"  Tanks saved: {len(rows)}")
        
        return excel_path
        
    except Exception as e:
        print(f"✗ Failed to save Excel file: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def upload_excel_to_gcp(
    excel_path: str,
    bucket_name: str = "bot-dump",
    bucket_folder: str = "results"
) -> Optional[str]:
    """
    Upload Excel file to GCP bucket.
    
    Args:
        excel_path: Path to local Excel file
        bucket_name: GCP bucket name (default: "bot-dump")
        bucket_folder: Folder in bucket (default: "results")
        
    Returns:
        str: GCS URI (gs://bucket/path), or None if upload failed
    """
    try:
        if not os.path.exists(excel_path):
            print(f"Error: Excel file not found: {excel_path}")
            return None
        
        # Extract filename
        filename = os.path.basename(excel_path)
        
        # Construct blob path
        blob_path = f"{bucket_folder}/{filename}"
        
        # Upload using gcp_utils
        print(f"Uploading Excel to GCP bucket...")
        gcs_uri = gcp_utils.upload_image_to_bucket(
            local_path=excel_path,
            bucket_name=bucket_name,
            blob_path=blob_path
        )
        
        print(f"✓ Excel uploaded to: {gcs_uri}")
        return gcs_uri
        
    except Exception as e:
        print(f"✗ Failed to upload Excel to GCP: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def save_and_upload_tank_data(
    phone_number: str,
    wwtp_location: tuple,
    circular_tanks: list,
    tank_heights: dict,
    volume_data: dict,
    satellite_image_path: str,
    output_dir: str = "Data",
    upload_to_gcp: bool = True
) -> Dict[str, Optional[str]]:
    """
    Save tank data to Excel and optionally upload to GCP bucket.
    
    This is the main function to call from whatsapp_bot.py.
    
    Args:
        phone_number: User's phone number
        wwtp_location: Tuple of (lat, lon) for original bbox center
        circular_tanks: List of circular tank data
        tank_heights: Dictionary of tank heights
        volume_data: Dictionary with volume calculations
        satellite_image_path: Path to satellite image
        output_dir: Directory to save Excel (default: "Data")
        upload_to_gcp: Whether to upload to GCP (default: True)
        
    Returns:
        dict: {
            'local_path': Path to local Excel file or None,
            'gcs_uri': GCS URI or None,
            'success': True if at least local save succeeded,
            'submission_id': Unique submission ID (e.g., "SUB20260123_071530_1234"),
            'timestamp_str': Timestamp string (e.g., "20260123_071530")
        }
    """
    # Generate timestamp and submission ID
    timestamp = datetime.now()
    submission_id = generate_submission_id(phone_number, timestamp)
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    
    # Save locally (pass the timestamp to ensure consistency)
    local_path = save_tank_data_to_excel(
        phone_number=phone_number,
        wwtp_location=wwtp_location,
        circular_tanks=circular_tanks,
        tank_heights=tank_heights,
        volume_data=volume_data,
        satellite_image_path=satellite_image_path,
        output_dir=output_dir,
        timestamp=timestamp  # Pass timestamp for consistency
    )
    
    gcs_uri = None
    
    # Upload to GCP if requested and local save succeeded
    if upload_to_gcp and local_path:
        gcs_uri = upload_excel_to_gcp(local_path)
    
    return {
        'local_path': local_path,
        'gcs_uri': gcs_uri,
        'success': local_path is not None,
        'submission_id': submission_id,
        'timestamp_str': timestamp_str
    }


# Test function
def test_excel_utils():
    """Test Excel utilities with sample data"""
    print("=" * 80)
    print("EXCEL UTILITIES TEST")
    print("=" * 80)
    
    # Test submission ID generation
    phone = "whatsapp:+1234567890"
    submission_id = generate_submission_id(phone)
    print(f"\nSubmission ID: {submission_id}")
    
    # Test phone sanitization
    sanitized = sanitize_phone_number(phone)
    print(f"Sanitized phone: {sanitized}")
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_excel_utils()
