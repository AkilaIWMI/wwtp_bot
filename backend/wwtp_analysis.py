"""
WWTP Analysis Pipeline
This script downloads satellite imagery, runs YOLO object detection,
and calculates real-world measurements for circular wastewater treatment tanks.
"""

import os
import math
import time
from pathlib import Path
import cv2
import numpy as np
import leafmap
from ultralytics import YOLO
from PIL import Image


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
    
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Download the image with the specified bounding box and zoom level
            leafmap.tms_to_geotiff(
                output=output_path, 
                bbox=bbox, 
                zoom=zoom, 
                source=source, 
                overwrite=True, 
                quiet=True
            )
            print(f"✓ Image successfully downloaded to: {output_path}")
            print(f"  Center: ({lat}, {lon})")
            print(f"  Bounding box: {bbox}")
            print(f"  Size: {bbox_size}m x {bbox_size}m")
            return True
            
        except Exception as e:
            print(f"✗ Attempt {retry_count + 1} failed: {str(e)}")
            retry_count += 1
            
            if retry_count < max_retries:
                sleep_time = 2 ** retry_count  # Exponential backoff strategy
                print(f"  Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"✗ Failed after {max_retries} attempts.")
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
    
    return {
        'radius_meters': round(radius_meters, 2),
        'diameter_meters': round(diameter_meters, 2),
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
        tuple: (annotated_image, circular_tank_data)
    """
    annotated_image = image.copy()
    circular_tank_data = []
    
    # Get the first result (single image inference)
    result = results[0]
    
    # Extract boxes, confidences, and class IDs
    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes in xyxy format
    confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
    
    print(f"\nDetected {len(boxes)} objects:")
    print("=" * 80)
    
    # Draw each detection
    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
        x1, y1, x2, y2 = box.astype(int)
        
        # Get class name and color
        class_name = class_names.get(cls_id, f"Class-{cls_id}")
        color = class_colors.get(cls_id, (255, 255, 255))
        
        print(f"\nObject {i+1}: {class_name}")
        print(f"  Confidence: {conf:.2f}")
        print(f"  Bounding Box: ({x1}, {y1}, {x2}, {y2})")
        
        # Calculate real-world radius for circular tanks
        if cls_id == 0:  # Circular-Tank
            measurements = get_real_world_radius(
                image_path, 
                [x1, y1, x2, y2], 
                bbox_coverage_meters
            )
            
            circular_tank_data.append({
                'object_id': i + 1,
                'class_name': class_name,
                'confidence': conf,
                'bounding_box': [x1, y1, x2, y2],
                'measurements': measurements
            })
            
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
            
            # Create enhanced label with radius
            label = f"{class_name}: {conf:.2f} | R={measurements['radius_meters']}m"
        else:
            label = f"{class_name}: {conf:.2f}"
        
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
    
    return annotated_image, circular_tank_data


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


def main():
    """
    Main function to run complete WWTP analysis pipeline:
    1. Download satellite image from hardcoded coordinates
    2. Run YOLO inference
    3. Calculate radius for circular tanks
    4. Save annotated image
    """
    print("=" * 80)
    print("WWTP SATELLITE ANALYSIS PIPELINE")
    print("=" * 80)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")
    
    # Step 1: Download satellite image
    print("\n" + "=" * 80)
    print("STEP 1: Downloading Satellite Image")
    print("=" * 80)
    
    satellite_image_path = os.path.join(OUTPUT_DIR, "satellite_image.tif")
    
    download_satellite_image(
        lat=CENTER_LAT,
        lon=CENTER_LON,
        output_path=satellite_image_path,
        bbox_size=BBOX_SIZE,
        zoom=ZOOM_LEVEL
    )
    
    # Step 2: Check if model exists
    print("\n" + "=" * 80)
    print("STEP 2: Loading YOLO Model")
    print("=" * 80)
    
    if not MODEL_PATH.exists():
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
    
    # Step 3: Run YOLO inference
    print("\n" + "=" * 80)
    print("STEP 3: Running YOLO Inference")
    print("=" * 80)
    
    image, results = run_yolo_inference(MODEL_PATH, satellite_image_path, conf_threshold=CONF_THRESHOLD)
    
    # Step 4: Draw predictions and calculate measurements
    print("\n" + "=" * 80)
    print("STEP 4: Calculating Measurements")
    print("=" * 80)
    
    annotated_image, circular_tank_data = draw_predictions(
        image, results, CLASS_NAMES, CLASS_COLORS, satellite_image_path, BBOX_SIZE
    )
    
    # Step 5: Save annotated image
    print("\n" + "=" * 80)
    print("STEP 5: Saving Results")
    print("=" * 80)
    
    output_path = os.path.join(OUTPUT_DIR, "satellite_image_annotated.jpg")
    save_image(annotated_image, output_path)
    
    # Print summary of circular tanks
    print_summary(circular_tank_data)
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"  - Original image: {satellite_image_path}")
    print(f"  - Annotated image: {output_path}")
    
    return circular_tank_data


if __name__ == "__main__":
    main()
