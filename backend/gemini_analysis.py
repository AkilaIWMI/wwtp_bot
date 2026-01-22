"""
Gemini AI Analysis Module
Integrates Google Gemini AI for advanced WWTP satellite image analysis.
"""

import os
import json
import base64
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Import GCP utilities for bucket operations
try:
    from . import gcp_utils  # When used as module
except ImportError:
    import gcp_utils  # When run as standalone script


# Load environment variables from .env file
load_dotenv()


def ensure_local_image(image_path: str, cache_dir: str = "Data/cache") -> str:
    """
    Ensure image is available locally, downloading from GCS bucket if needed.
    
    Args:
        image_path: Path to image (local path or GCS URI gs://bucket/path)
        cache_dir: Directory to cache downloaded images
        
    Returns:
        str: Local path to the image
        
    Raises:
        FileNotFoundError: If image doesn't exist locally or in bucket
    """
    # If it's a GCS URI, download it
    if gcp_utils.is_gcs_uri(image_path):
        print(f"Image is in GCS bucket, downloading for analysis...")
        
        # Parse GCS URI
        bucket_name, blob_path = gcp_utils.parse_gcs_uri(image_path)
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Extract filename from blob path
        filename = os.path.basename(blob_path)
        local_cache_path = os.path.join(cache_dir, filename)
        
        # Download if not already cached
        if not os.path.exists(local_cache_path):
            gcp_utils.download_image_from_bucket(
                bucket_name=bucket_name,
                blob_path=blob_path,
                local_path=local_cache_path
            )
        else:
            print(f"Using cached image: {local_cache_path}")
        
        return local_cache_path
    
    # Otherwise, it's a local path - verify it exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    return image_path


def load_gemini_api_key():
    """
    Load Google Gemini API key from environment variables.
    
    Returns:
        str: API key or None if not found
        
    Raises:
        ValueError: If API key is not found in environment
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "Google API key not found. Please add GOOGLE_API_KEY to your .env file. "
            "Get your key from: https://makersuite.google.com/app/apikey"
        )
    
    return api_key


def analyze_image_with_gemini(image_path):
    """
    Analyze a satellite image using Google Gemini AI to identify WWTP features.
    
    This function sends the image to Gemini with a specialized prompt for
    WWTP identification, circular tank counting, and description generation.
    
    Args:
        image_path (str): Path to the satellite image file
        
    Returns:
        dict: Analysis results containing:
            - is_wwtp (bool): Whether the image contains a WWTP
            - circular_tank_count (int): Number of circular tanks detected
            - description (str): Descriptive analysis of the image
            - error (str): Error message if analysis failed (None if successful)
            
    Raises:
        FileNotFoundError: If image file doesn't exist
    """
    try:
        # Ensure image is available locally (download from GCS if needed)
        local_image_path = ensure_local_image(image_path)
        
        # Load API key and configure Gemini
        api_key = load_gemini_api_key()
        
        # Initialize the Gemini client with new API
        client = genai.Client(api_key=api_key)
        
        # Load the image
        print(f"Loading image for Gemini analysis: {local_image_path}")
        
        # Read image file and convert to base64 (required for new API)
        import base64
        with open(local_image_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Determine mime type based on file extension
        file_ext = os.path.splitext(local_image_path)[1].lower()
        mime_type = 'image/tiff' if file_ext == '.tif' else 'image/jpeg'
        
        # WWTP analysis prompt
        prompt = """
You are an expert Remote Sensing Analyst specializing in critical infrastructure. Analyze the provided satellite image to identify if it contains a Wastewater Treatment Plant (WWTP).

### 1. IDENTIFICATION LOGIC (The "Functional Signature")
To confirm a WWTP, do not just look for round tanks. You must find a *process flow* that distinguishes it from oil/gas storage or agriculture. Use these advanced visual cues:

* **The "Dirty-to-Clean" Gradient:**
    * Look for a color change across the tanks. The input side (Aeration/PST) often looks brownish, turbulent, or frothy. The output side (Clarifiers) often looks darker/clearer or "calm."
    * *Oil/Gas tanks* are uniform in color/finish. *WWTP tanks* vary because the water quality changes as it moves through them.
* **The Shadow Test (Height vs. Depth):**
    * *Digesters* are often tall, domed cylinders that cast long shadows.
    * *Clarifiers/PSTs* are ground-level or sunken; they cast little to no shadow and often show internal "bridges" or scraper arms.
* **Connectivity:**
    * Look for open channels (water flumes) connecting the tanks. Oil tanks use narrow pipes; WWTPs often use visible open water channels between rectangular and circular units.
    * Look for discharge points (outfall) into a nearby river, lake, or ocean.

### 2. COUNTING RULES (Circular Tanks Only)
* Count **only** distinct, stand-alone circular structures that appear to be part of the treatment process (PSTs, Clarifiers, Thickeners, Digesters).
* Do **not** count small manholes, silos, or circular bushes.
* If `is_wwtp` is NO, the count should be 0.

### 3. OUTPUT FORMAT
You must return a strictly valid JSON object containing ONLY these three keys. Do not include markdown formatting (like ```json), intro text, or explanations.

{
  "is_wwtp": boolean, // true if it is a WWTP, false if it is not (or if ambiguous).
  "circular_tank_count": integer, // The count of circular tanks identified.
  "description": string // A concise summary (max 2 sentences) describing the specific visual features (e.g., "Cluster of 3 circular clarifiers and rectangular aeration basins with visible water channels nearby.")
}
"""
        
        print("Sending image to Gemini for analysis...")
        
        # Generate content with image and prompt using new API
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=[
                types.Content(
                    parts=[
                        types.Part(text=prompt),
                        types.Part(
                            inline_data=types.Blob(
                                mime_type=mime_type,
                                data=image_data
                            )
                        )
                    ]
                )
            ]
        )
        
        # Parse the response
        result = parse_gemini_response(response)
        
        print(f"✓ Gemini analysis complete: WWTP={result['is_wwtp']}, Tanks={result['circular_tank_count']}")
        
        return result
        
    except ValueError as e:
        # API key error
        print(f"✗ Gemini API key error: {str(e)}")
        return {
            'is_wwtp': None,
            'circular_tank_count': 0,
            'description': "AI analysis unavailable: API key not configured",
            'error': str(e)
        }
        
    except FileNotFoundError as e:
        print(f"✗ Image file error: {str(e)}")
        return {
            'is_wwtp': None,
            'circular_tank_count': 0,
            'description': "AI analysis failed: Image file not found",
            'error': str(e)
        }
        
    except Exception as e:
        # Any other error (network, API limits, parsing, etc.)
        print(f"✗ Gemini analysis error: {str(e)}")
        return {
            'is_wwtp': None,
            'circular_tank_count': 0,
            'description': f"AI analysis failed: {str(e)}",
            'error': str(e)
        }


def parse_gemini_response(response):
    """
    Parse and validate the JSON response from Gemini.
    
    Gemini may return the JSON wrapped in markdown code blocks or with extra text.
    This function extracts the JSON and validates it.
    
    Args:
        response: Gemini API response object
        
    Returns:
        dict: Parsed and validated response with keys:
            - is_wwtp (bool)
            - circular_tank_count (int)
            - description (str)
            - error (None if successful)
            
    Raises:
        ValueError: If response cannot be parsed or is invalid
    """
    try:
        # Get the text response
        response_text = response.text.strip()
        print(f"Raw Gemini response:\n{response_text}\n")
        
        # Try to extract JSON from markdown code blocks if present
        if "```json" in response_text:
            # Extract content between ```json and ```
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
        elif "```" in response_text:
            # Extract content between ``` and ```
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
        else:
            # Assume the entire response is JSON
            json_str = response_text
        
        # Parse JSON
        data = json.loads(json_str)
        
        # Validate required keys
        required_keys = ['is_wwtp', 'circular_tank_count', 'description']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key in Gemini response: {key}")
        
        # Validate types
        if not isinstance(data['is_wwtp'], bool):
            raise ValueError(f"'is_wwtp' must be a boolean, got: {type(data['is_wwtp'])}")
        
        if not isinstance(data['circular_tank_count'], int):
            raise ValueError(f"'circular_tank_count' must be an integer, got: {type(data['circular_tank_count'])}")
        
        if not isinstance(data['description'], str):
            raise ValueError(f"'description' must be a string, got: {type(data['description'])}")
        
        # Return validated data
        return {
            'is_wwtp': data['is_wwtp'],
            'circular_tank_count': data['circular_tank_count'],
            'description': data['description'],
            'error': None
        }
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse Gemini response as JSON: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error validating Gemini response: {str(e)}")


# Test function for standalone execution
def test_gemini_analysis():
    """
    Test function to verify Gemini integration.
    Requires a test image and valid API key in .env file.
    """
    print("="*80)
    print("GEMINI ANALYSIS MODULE TEST")
    print("="*80)
    
    # Check for test image (use forward slashes or raw strings to avoid escape issues)
    test_image_paths = [
        r"D:\Akila IWMI\wwtp_bot\wwtp_bot\backend\Data\satellite_32.426178_35.966435.tif",
        "../Data/satellite_32.566846_35.933951.tif",
        "../Data/satellite_image.tif",
        "Data/satellite_image.tif"
    ]
    
    test_image = None
    for path in test_image_paths:
        if os.path.exists(path):
            test_image = path
            break
    
    if test_image is None:
        print("\n✗ No test image found. Please run wwtp_analysis.py first to download a satellite image.")
        print(f"  Looking for images in: {test_image_paths}")
        return
    
    print(f"\nUsing test image: {test_image}")
    
    # Run analysis
    result = analyze_image_with_gemini(test_image)
    
    # Display results
    print("\n" + "="*80)
    print("GEMINI ANALYSIS RESULTS")
    print("="*80)
    print(f"WWTP Detected: {result['is_wwtp']}")
    print(f"Circular Tank Count: {result['circular_tank_count']}")
    print(f"Description: {result['description']}")
    
    if result['error']:
        print(f"Error: {result['error']}")
    
    print("="*80)


if __name__ == "__main__":
    test_gemini_analysis()
