"""
WWTP Analysis Pipeline
This script downloads satellite imagery, runs YOLO object detection,
and calculates real-world measurements for circular wastewater treatment tanks.
"""

import os
import math
import time
import traceback
from pathlib import Path
import cv2
import numpy as np
import leafmap
from ultralytics import YOLO
from PIL import Image

# Import GCP utilities for bucket operations
try:
    from . import gcp_utils  # When used as module
except ImportError:
    import gcp_utils  # When run as standalone script


# ============================================
# HARDCODED PARAMETERS - EDIT THESE VALUES
# ============================================

# Center coordinates (Beirut, Lebanon example)
CENTER_LAT = 32.566846
CENTER_LON = 35.933951

# CENTER_LAT = 38.5654
# CENTER_LON = 39.9378

# Bounding box size in meters (e.g., 250m x 250m coverage area)
BBOX_SIZE = 250

# Zoom level (higher = more detail, typically 19-20 for satellite images)
ZOOM_LEVEL = 20

# Output directory for data
OUTPUT_DIR = "Data"

# YOLO model configuration
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "model" / "yolo8_segmentation .pt"
CONF_THRESHOLD = 0.25  # Confidence threshold for detections

# Class names mapping
CLASS_NAMES = {
    0: 'Circular-Tank',
    1: 'Rectangular-Tank',
    2: 'WWTP'
}

# Color mapping for each class (BGR format for OpenCV)
CLASS_COLORS = {
    0: (255, 0, 0),      # Blue for Circular-Tank
    1: (0, 255, 0),      # Green for Rectangular-Tank
    2: (0, 0, 255)       # Red for WWTP
}

# ============================================


def calculate_bbox(lat, lon, size_meters=100):
    """
    Calculate a bounding box around a point (lat, lon) of a fixed size.

    Parameters:
    - lat (float): Latitude of the center point.
    - lon (float): Longitude of the center point.
    - size_meters (int): The size of the bounding box in meters (e.g., 100m x 100m).

    Returns:
    - list: [min_lon, min_lat, max_lon, max_lat]
    """
    # Approximate conversion factor from meters to degrees for latitude
    meters_per_degree_lat = 111320  # Roughly, meters per degree of latitude

    # Adjust meters per degree of longitude based on latitude
    meters_per_degree_lon = meters_per_degree_lat * math.cos(math.radians(lat))

    # Calculate the degree difference for the given size in meters
    delta_lat = size_meters / meters_per_degree_lat
    delta_lon = size_meters / meters_per_degree_lon
    
    min_lat = lat - delta_lat / 2
    max_lat = lat + delta_lat / 2
    min_lon = lon - delta_lon / 2
    max_lon = lon + delta_lon / 2
    
    return [min_lon, min_lat, max_lon, max_lat]


def download_satellite_image(lat, lon, output_path, bbox_size=100, zoom=19, source="Satellite", max_retries=10):
    """
    Download a high-resolution satellite image using leafmap.

    Parameters:
    - lat (float): Latitude of the center point.
    - lon (float): Longitude of the center point.
    - output_path (str): File path to save the downloaded image (must end with .tif).
    - bbox_size (int): Size of the bounding box in meters (e.g., 100 for 100m x 100m).
    - zoom (int): Zoom level for the image. Higher zoom levels give more detail (default: 19).
    - source (str): The source of satellite imagery (default: "Satellite").
    - max_retries (int): Maximum number of retry attempts in case of failure.
    """
    # Calculate bounding box
    bbox = calculate_bbox(lat, lon, bbox_size)
    
    # Debug: Print bbox details
    print(f"  DEBUG: Calculated bbox for lat={lat}, lon={lon}, size={bbox_size}m")
    print(f"  DEBUG: bbox = {bbox}")
    
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Download the image with the specified bounding box and zoom level
            print(f"  Attempting to download satellite image with leafmap...")
            print(f"  Output path: {output_path}")
            print(f"  Bbox: {bbox}")
            print(f"  Zoom: {zoom}, Source: {source}")
            
            leafmap.tms_to_geotiff(
                output=output_path, 
                bbox=bbox, 
                zoom=zoom, 
                source=source, 
                overwrite=True, 
                quiet=False  # Changed to False to see any error messages
            )
            print(f"✓ Image successfully downloaded to: {output_path}")
            print(f"  Center: ({lat}, {lon})")
            print(f"  Bounding box: {bbox}")
            print(f"  Size: {bbox_size}m x {bbox_size}m")
            return True
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e) if str(e) else "(no error message)"
            print(f"✗ Attempt {retry_count + 1} failed:")
            print(f"  Error Type: {error_type}")
            print(f"  Error Message: {error_msg}")
            print(f"  Traceback:")
            traceback.print_exc()
            retry_count += 1
            
            if retry_count < max_retries:
                sleep_time = 2 ** retry_count  # Exponential backoff strategy
                print(f"  Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"✗ Failed after {max_retries} attempts.")
                print(f"  Final error: {error_type}: {error_msg}")
                raise
    
    return False


def load_image(image_path):
    """
    Load image from file path.
    Supports both regular images and TIFF files.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        numpy.ndarray: Loaded image in BGR format
    """
    # Try loading with OpenCV first
    img = cv2.imread(str(image_path))
    
    if img is None:
        # If OpenCV fails, try with PIL (better TIFF support)
        print(f"OpenCV couldn't load {image_path}, trying PIL...")
        pil_img = Image.open(image_path)
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    return img


def get_real_world_radius(image_path, detection_box, bbox_coverage_meters=250):
    """
    Calculate the real-world radius of a circular object from its bounding box.
    
    Uses pixel-to-meter ratio methodology:
    1. Calculate meters_per_pixel from known bbox coverage and image width
    2. Measure object diameter in pixels from bounding box
    3. Convert to real-world diameter and divide by 2 for radius
    
    Args:
        image_path (str): Path to the satellite image
        detection_box (list/tuple): Bounding box coordinates [xmin, ymin, xmax, ymax]
        bbox_coverage_meters (int): Real-world size of the image in meters (default: 250)
        
    Returns:
        dict: Dictionary containing:
            - radius_meters (float): Calculated radius in meters
            - diameter_meters (float): Calculated diameter in meters
            - diameter_pixels (float): Measured diameter in pixels
            - meters_per_pixel (float): Conversion ratio
            - center_x (float): Center X coordinate in pixels
            - center_y (float): Center Y coordinate in pixels
    """
    # Load image to get dimensions
    img = load_image(image_path)
    image_height, image_width = img.shape[:2]
    
    # Calculate meters per pixel
    # Assumes square bbox coverage (same for width and height)
    meters_per_pixel = bbox_coverage_meters / image_width
    
    # Extract bounding box coordinates
    xmin, ymin, xmax, ymax = detection_box
    
    # Calculate bounding box dimensions in pixels
    bbox_width_pixels = xmax - xmin
    bbox_height_pixels = ymax - ymin
    
    # Calculate diameter as average of width and height
    # (accounts for slight distortions in circular objects)
    diameter_pixels = (bbox_width_pixels + bbox_height_pixels) / 2.0
    
    # Convert to real-world measurements
    diameter_meters = diameter_pixels * meters_per_pixel
    radius_meters = diameter_meters / 2.0
    
    # Calculate center coordinates
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    
    # Calculate surface area (π × r²)
    import math
    surface_area = math.pi * (radius_meters ** 2)
    
    return {
        'radius_meters': round(radius_meters, 2),
        'diameter_meters': round(diameter_meters, 2),
        'surface_area_sqm': round(surface_area, 2),
        'diameter_pixels': round(diameter_pixels, 2),
        'meters_per_pixel': round(meters_per_pixel, 4),
        'center_x': round(center_x, 2),
        'center_y': round(center_y, 2)
    }


def run_yolo_inference(model_path, image_path, conf_threshold=0.25):
    """
    Run YOLO inference on the image.
    
    Args:
        model_path: Path to the YOLO model file
        image_path: Path to the input image
        conf_threshold: Confidence threshold for detections
        
    Returns:
        tuple: (original_image, results)
    """
    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(str(model_path))
    
    print(f"Loading image from: {image_path}")
    image = load_image(image_path)
    
    print(f"Running inference with confidence threshold: {conf_threshold}")
    results = model(image, conf=conf_threshold)
    
    return image, results


def draw_predictions(image, results, class_names, class_colors, image_path, bbox_coverage_meters):
    """
    Draw bounding boxes and labels on the image.
    For circular tanks, also calculate and display real-world radius.
    
    Args:
        image: Original image (numpy array)
        results: YOLO results object
        class_names: Dictionary mapping class IDs to names
        class_colors: Dictionary mapping class IDs to BGR colors
        image_path: Path to the image (for radius calculation)
        bbox_coverage_meters: Real-world coverage in meters
        
    Returns:
        tuple: (annotated_image, circular_tank_data, all_detections)
    """
    annotated_image = image.copy()
    circular_tank_data = []
    all_detections = []
    
    # Get the first result (single image inference)
    result = results[0]
    
    # Extract boxes, confidences, and class IDs
    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes in xyxy format
    confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
    
    print(f"\nDetected {len(boxes)} objects:")
    print("=" * 80)
    
    # Track circular tank IDs separately for unique numbering
    circular_tank_counter = 0
    
    # Draw each detection
    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
        x1, y1, x2, y2 = box.astype(int)
        
        # Get class name and color
        class_name = class_names.get(cls_id, f"Class-{cls_id}")
        color = class_colors.get(cls_id, (255, 255, 255))
        
        print(f"\nObject {i+1}: {class_name}")
        print(f"  Confidence: {conf:.2f}")
        print(f"  Bounding Box: ({x1}, {y1}, {x2}, {y2})")
        
        # Store detection info for all objects
        detection_info = {
            'object_id': i + 1,
            'class_name': class_name,
            'class_id': int(cls_id),
            'confidence': float(conf),
            'bounding_box': [int(x1), int(y1), int(x2), int(y2)]
        }
        
        # Calculate real-world radius for circular tanks
        if cls_id == 0:  # Circular-Tank
            circular_tank_counter += 1
            tank_id = circular_tank_counter
            
            measurements = get_real_world_radius(
                image_path, 
                [x1, y1, x2, y2], 
                bbox_coverage_meters
            )
            
            detection_info['measurements'] = measurements
            detection_info['tank_id'] = tank_id  # Add unique tank ID
            circular_tank_data.append(detection_info)
            
            print(f"  Tank ID: {tank_id}")
            print(f"  Real-World Measurements:")
            print(f"    - Radius: {measurements['radius_meters']} meters")
            print(f"    - Diameter: {measurements['diameter_meters']} meters")
            print(f"    - Center: ({measurements['center_x']}, {measurements['center_y']}) pixels")
            print(f"    - Scale: {measurements['meters_per_pixel']} meters/pixel")
            
            # Draw center point
            center_x = int(measurements['center_x'])
            center_y = int(measurements['center_y'])
            cv2.circle(annotated_image, (center_x, center_y), 5, color, -1)
            cv2.circle(annotated_image, (center_x, center_y), 8, (255, 255, 255), 2)
            
            # Draw large unique ID number on the tank
            id_text = f"#{tank_id}"
            id_font_scale = 1.5
            id_thickness = 3
            
            # Get ID text size for positioning
            (id_width, id_height), id_baseline = cv2.getTextSize(
                id_text, cv2.FONT_HERSHEY_DUPLEX, id_font_scale, id_thickness
            )
            
            # Position ID at top-left corner of bounding box, slightly inside
            id_x = x1 + 10
            id_y = y1 + id_height + 15
            
            # Draw black outline for better visibility
            cv2.putText(
                annotated_image,
                id_text,
                (id_x, id_y),
                cv2.FONT_HERSHEY_DUPLEX,
                id_font_scale,
                (0, 0, 0),  # Black outline
                id_thickness + 2
            )
            
            # Draw white ID text
            cv2.putText(
                annotated_image,
                id_text,
                (id_x, id_y),
                cv2.FONT_HERSHEY_DUPLEX,
                id_font_scale,
                (255, 255, 255),  # White text
                id_thickness
            )
            
            # Create enhanced label with radius
            label = f"{class_name}: {conf:.2f} | R={measurements['radius_meters']}m"
        else:
            label = f"{class_name}: {conf:.2f}"
        
        # Add to all detections list
        all_detections.append(detection_info)
        
        # Draw bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        
        # Get label size for background rectangle
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        # Draw label background
        cv2.rectangle(
            annotated_image,
            (x1, y1 - label_height - baseline - 5),
            (x1 + label_width, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            annotated_image,
            label,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),  # White text
            2
        )
    
    return annotated_image, circular_tank_data, all_detections



def save_image(image, output_path):
    """
    Save the annotated image to disk.
    
    Args:
        image: Image to save (numpy array)
        output_path: Path where to save the image
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the image
    cv2.imwrite(str(output_path), image)
    print(f"\nAnnotated image saved to: {output_path}")


def print_summary(circular_tank_data):
    """
    Print a summary of all circular tanks detected.
    
    Args:
        circular_tank_data: List of dictionaries containing tank measurements
    """
    if not circular_tank_data:
        print("\nNo circular tanks detected in the image.")
        return
    
    print("\n" + "=" * 80)
    print("CIRCULAR TANK SUMMARY")
    print("=" * 80)
    
    for tank in circular_tank_data:
        print(f"\nCircular Tank #{tank['object_id']}:")
        print(f"  Confidence: {tank['confidence']:.2f}")
        print(f"  Radius: {tank['measurements']['radius_meters']} meters")
        print(f"  Diameter: {tank['measurements']['diameter_meters']} meters")
        print(f"  Center Coordinates: ({tank['measurements']['center_x']}, {tank['measurements']['center_y']}) pixels")
        print(f"  Bounding Box: {tank['bounding_box']}")
    
    print("\n" + "=" * 80)


def analyze_wwtp(lat, lon, output_dir="Data"):
    """
    Analyze WWTP from satellite imagery at given coordinates.
    
    This function orchestrates the complete pipeline:
    1. Download satellite image
    2. Run YOLO inference
    3. Calculate measurements for circular tanks
    4. Save annotated image
    
    Args:
        lat (float): Latitude coordinate
        lon (float): Longitude coordinate
        output_dir (str): Directory to save images (default: "Data")
        
    Returns:
        dict: Results dictionary containing:
            - success (bool): Whether analysis completed successfully
            - wwtp_detected (bool): Whether any WWTP-related objects were detected
            - circular_tanks (list): List of circular tank data with measurements
            - all_detections (list): All detected objects
            - detection_counts (dict): Count of each class detected
            - annotated_image_path (str): Path to annotated image
            - satellite_image_path (str): Path to original satellite image
            - error (str): Error message if failed
    """
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        
        # Step 1: Download satellite image
        print(f"\n{'='*80}")
        print("STEP 1: Downloading Satellite Image")
        print(f"{'='*80}")
        
        # Generate timestamp for unique filename
        timestamp = gcp_utils.generate_timestamp()
        satellite_image_path = os.path.join(output_dir, f"satellite_{lat}_{lon}_{timestamp}.tif")
        
        download_satellite_image(
            lat=lat,
            lon=lon,
            output_path=satellite_image_path,
            bbox_size=BBOX_SIZE,
            zoom=ZOOM_LEVEL
        )
        
        # Upload satellite image to GCP bucket
        try:
            bucket_name = os.getenv("GCP_BUCKET_NAME", "bot-dump")
            bucket_path = os.getenv("GCP_BUCKET_PATH", "meta_data")
            
            # Construct blob path: meta_data/satellite_LAT_LON_TIMESTAMP.tif
            blob_path = f"{bucket_path}/satellite_{lat}_{lon}_{timestamp}.tif"
            
            print(f"\nUploading satellite image to GCP bucket...")
            satellite_gcs_uri = gcp_utils.upload_image_to_bucket(
                local_path=satellite_image_path,
                bucket_name=bucket_name,
                blob_path=blob_path
            )
            print(f"✓ Satellite image uploaded: {satellite_gcs_uri}")
        except Exception as e:
            print(f"✗ Failed to upload satellite image to bucket: {str(e)}")
            satellite_gcs_uri = None
        
        # Step 2: Check if model exists
        print(f"\n{'='*80}")
        print("STEP 2: Loading YOLO Model")
        print(f"{'='*80}")
        
        if not MODEL_PATH.exists():
            return {
                'success': False,
                'wwtp_detected': False,
                'circular_tanks': [],
                'all_detections': [],
                'detection_counts': {},
                'annotated_image_path': None,
                'satellite_image_path': satellite_image_path,
                'satellite_gcs_uri': satellite_gcs_uri,
                'annotated_gcs_uri': None,
                'annotated_public_url': None,
                'error': f"Model file not found at {MODEL_PATH}"
            }
        
        # Step 3: Run YOLO inference
        print(f"\n{'='*80}")
        print("STEP 3: Running YOLO Inference")
        print(f"{'='*80}")
        
        image, results = run_yolo_inference(MODEL_PATH, satellite_image_path, conf_threshold=CONF_THRESHOLD)
        
        # Step 4: Draw predictions and calculate measurements
        print(f"\n{'='*80}")
        print("STEP 4: Calculating Measurements")
        print(f"{'='*80}")
        
        annotated_image, circular_tank_data, all_detections = draw_predictions(
            image, results, CLASS_NAMES, CLASS_COLORS, satellite_image_path, BBOX_SIZE
        )
        
        # Step 5: Save annotated image
        print(f"\n{'='*80}")
        print("STEP 5: Saving Results")
        print(f"{'='*80}")
        
        output_path = os.path.join(output_dir, f"annotated_{lat}_{lon}_{timestamp}.jpg")
        save_image(annotated_image, output_path)
        
        # Upload annotated image to GCP bucket
        annotated_public_url = None
        try:
            # Construct blob path: meta_data/annotated_LAT_LON_TIMESTAMP.jpg
            annotated_blob_path = f"{bucket_path}/annotated_{lat}_{lon}_{timestamp}.jpg"
            
            print(f"\nUploading annotated image to GCP bucket...")
            annotated_gcs_uri = gcp_utils.upload_image_to_bucket(
                local_path=output_path,
                bucket_name=bucket_name,
                blob_path=annotated_blob_path
            )
            print(f"✓ Annotated image uploaded: {annotated_gcs_uri}")
            
            # Generate public URL for Twilio WhatsApp media
            print(f"\nGenerating public URL for annotated image...")
            annotated_public_url = gcp_utils.get_public_url_for_blob(
                bucket_name=bucket_name,
                blob_path=annotated_blob_path,
                expiration_minutes=60
            )
            print(f"✓ Public URL generated (expires in 60 minutes)")
        except Exception as e:
            print(f"✗ Failed to upload annotated image to bucket: {str(e)}")
            annotated_gcs_uri = None
        
        # Calculate detection counts
        detection_counts = {}
        for detection in all_detections:
            class_name = detection['class_name']
            detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
        
        # Check if WWTP was detected
        wwtp_detected = any(d['class_name'] == 'WWTP' for d in all_detections)
        
        # Print summary
        print_summary(circular_tank_data)
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        
        return {
            'success': True,
            'wwtp_detected': wwtp_detected,
            'circular_tanks': circular_tank_data,
            'all_detections': all_detections,
            'detection_counts': detection_counts,
            'annotated_image_path': output_path,
            'satellite_image_path': satellite_image_path,
            'satellite_gcs_uri': satellite_gcs_uri,  # GCS URI for satellite image
            'annotated_gcs_uri': annotated_gcs_uri,  # GCS URI for annotated image
            'annotated_public_url': annotated_public_url,  # Public URL for Twilio WhatsApp
            'error': None
        }
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: Analysis failed - {str(e)}")
        print(f"{'='*80}")
        
        return {
            'success': False,
            'wwtp_detected': False,
            'circular_tanks': [],
            'all_detections': [],
            'detection_counts': {},
            'annotated_image_path': None,
            'satellite_image_path': None,
            'satellite_gcs_uri': None,
            'annotated_gcs_uri': None,
            'annotated_public_url': None,
            'error': str(e)
        }


def main():
    """
    Main function to run complete WWTP analysis pipeline with GCP upload.
    Uses the analyze_wwtp() function which includes GCP bucket integration.
    """
    print("=" * 80)
    print("WWTP SATELLITE ANALYSIS PIPELINE")
    print("=" * 80)
    
    # Use analyze_wwtp() function which includes GCP upload
    result = analyze_wwtp(
        lat=CENTER_LAT,
        lon=CENTER_LON,
        output_dir=OUTPUT_DIR
    )
    
    # Print results
    if result['success']:
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nLocal files:")
        print(f"  - Satellite image: {result['satellite_image_path']}")
        print(f"  - Annotated image: {result['annotated_image_path']}")
        
        if result['satellite_gcs_uri']:
            print(f"\nGCS Bucket uploads:")
            print(f"  - Satellite image: {result['satellite_gcs_uri']}")
            print(f"  - Annotated image: {result['annotated_gcs_uri']}")
        else:
            print(f"\n⚠️ GCS upload failed (images saved locally only)")
        
        print(f"\nDetection summary:")
        print(f"  - WWTP detected: {result['wwtp_detected']}")
        print(f"  - Circular tanks: {len(result['circular_tanks'])}")
        print(f"  - Total detections: {len(result['all_detections'])}")
        
        if result['detection_counts']:
            print(f"\nDetection counts:")
            for class_name, count in result['detection_counts'].items():
                print(f"  - {class_name}: {count}")
    else:
        print("\n" + "=" * 80)
        print("ANALYSIS FAILED!")
        print("=" * 80)
        print(f"Error: {result['error']}")
    
    return result


if __name__ == "__main__":
    main()
    