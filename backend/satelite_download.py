import os
import math
import time
import leafmap


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
            break  # Exit loop if operation is successful
            
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


if __name__ == "__main__":
    # ============================================
    # HARDCODED PARAMETERS - EDIT THESE VALUES
    # ============================================
    
    # Center coordinates (example: Beirut, Lebanon)
    CENTER_LAT = 32.566846
    CENTER_LON = 35.933951
    
    # Bounding box size in meters (100m x 100m)
    BBOX_SIZE = 250
    
    # Zoom level (19 is very high resolution)
    ZOOM_LEVEL = 20
    
    # Output directory and filename
    OUTPUT_DIR = "Data"
    OUTPUT_FILENAME = "satellite_image.tif"
    
    # ============================================
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")
    
    # Full output path
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    
    # Download the satellite image
    print(f"\nDownloading satellite image...")
    print(f"=" * 50)
    
    download_satellite_image(
        lat=CENTER_LAT,
        lon=CENTER_LON,
        output_path=output_path,
        bbox_size=BBOX_SIZE,
        zoom=ZOOM_LEVEL
    )
    
    print(f"=" * 50)
    print("Done!")