"""
WhatsApp Bot for WWTP Analysis
Integrates Twilio WhatsApp with WWTP satellite image analysis pipeline
"""

import os
import re
import uvicorn
from fastapi import FastAPI, Form, Response, Request
from twilio.twiml.messaging_response import MessagingResponse
from wwtp_analysis import analyze_wwtp

app = FastAPI()

# Session storage to track user states
# Key: phone number (From field), Value: {lat, lon, timestamp}
user_sessions = {}


def parse_location_data(message):
    """
    Parse latitude, longitude, and height from WhatsApp message.
    
    Expected format: "latitude,longitude,height" (e.g., "32.566846,35.933951,5")
    
    Args:
        message (str): The incoming message text
        
    Returns:
        tuple: (latitude, longitude, height) or (None, None, None) if parsing fails
    """
    # Pattern: three numbers separated by commas
    # Each number can be positive/negative with optional decimal
    pattern = r'^\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*$'
    match = re.match(pattern, message.strip())
    
    if match:
        try:
            lat = float(match.group(1))
            lon = float(match.group(2))
            height = float(match.group(3))
            return lat, lon, height
        except ValueError:
            return None, None, None
    
    return None, None, None


def parse_height_only(message):
    """
    Parse height value from a message (for second step after location sharing).
    
    Args:
        message (str): The incoming message text
        
    Returns:
        float: Height value or None if parsing fails
    """
    # Try to parse a single number
    pattern = r'^\s*(-?\d+\.?\d*)\s*$'
    match = re.match(pattern, message.strip())
    
    if match:
        try:
            height = float(match.group(1))
            return height
        except ValueError:
            return None
    
    return None


def validate_coordinates(lat, lon):
    """
    Validate that coordinates are within valid ranges.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if lat is None or lon is None:
        return False, "Could not parse coordinates."
    
    if not (-90 <= lat <= 90):
        return False, f"❌ Invalid latitude: {lat}. Latitude must be between -90 and 90."
    
    if not (-180 <= lon <= 180):
        return False, f"❌ Invalid longitude: {lon}. Longitude must be between -180 and 180."
    
    return True, None


def validate_location_data(lat, lon, height):
    """
    Validate that coordinates and height are within valid ranges.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        height (float): Height in meters
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if lat is None or lon is None or height is None:
        return False, (
            "Hi,\n\n"
            "Please submit the location in this exact format:\n"
            "latitude,longitude,height\n\n"
            "Example: 32.566846,35.933951,5\n\n"
            "Make sure to:\n"
            "✓ Use commas to separate values\n"
            "✓ Include all three values (latitude, longitude, height in meters)\n"
            "✓ Use decimal numbers"
        )
    
    if not (-90 <= lat <= 90):
        return False, f"❌ Invalid latitude: {lat}. Latitude must be between -90 and 90."
    
    if not (-180 <= lon <= 180):
        return False, f"❌ Invalid longitude: {lon}. Longitude must be between -180 and 180."
    
    if height <= 0:
        return False, f"❌ Invalid height: {height}. Height must be a positive number (in meters)."
    
    return True, None


def calculate_tank_volume(circular_tanks, height):
    """
    Calculate volume for each circular tank using the formula V = π × r² × h.
    
    Args:
        circular_tanks (list): List of circular tank data with measurements
        height (float): Height of the tanks in meters
        
    Returns:
        dict: Dictionary containing:
            - total_volume (float): Total volume of all tanks in cubic meters
            - tank_volumes (list): List of individual tank volumes with details
    """
    if not circular_tanks:
        return {
            'total_volume': 0,
            'tank_volumes': []
        }
    
    import math
    tank_volumes = []
    total_volume = 0
    
    for tank in circular_tanks:
        radius = tank['measurements']['radius_meters']
        # V = π × r² × h
        volume = math.pi * (radius ** 2) * height
        
        tank_volumes.append({
            'tank_id': tank['object_id'],
            'radius': radius,
            'volume': round(volume, 2)
        })
        
        total_volume += volume
    
    return {
        'total_volume': round(total_volume, 2),
        'tank_volumes': tank_volumes
    }


def format_wwtp_response(results, height=None, volume_data=None):
    """
    Format the WWTP analysis results into a user-friendly WhatsApp message.
    
    Args:
        results (dict): Results from analyze_wwtp function
        height (float): Height of tanks in meters (optional)
        volume_data (dict): Volume calculation data (optional)
        
    Returns:
        str: Formatted message for WhatsApp
    """
    if not results['success']:
        return f"❌ Analysis failed: {results['error']}"
    
    # Check if any WWTP-related objects were detected
    if not results['wwtp_detected'] and len(results['all_detections']) == 0:
        return "🔍 There are no WWTP in this location."
    
    # Build response message
    message_parts = ["✅ *WWTP Analysis Results*\n"]
    
    # Add detection summary
    detection_counts = results['detection_counts']
    if detection_counts:
        message_parts.append("📊 *Detected Objects:*")
        for class_name, count in detection_counts.items():
            message_parts.append(f"  • {class_name}: {count}")
        message_parts.append("")
    
    # Add circular tank details
    circular_tanks = results['circular_tanks']
    if circular_tanks:
        message_parts.append(f"🔵 *Circular Tanks ({len(circular_tanks)}):*")
        for i, tank in enumerate(circular_tanks, 1):
            measurements = tank['measurements']
            tank_info = (
                f"  {i}. Radius: {measurements['radius_meters']}m "
                f"(Diameter: {measurements['diameter_meters']}m)"
            )
            
            # Add volume information if available
            if volume_data and volume_data['tank_volumes']:
                for vol_info in volume_data['tank_volumes']:
                    if vol_info['tank_id'] == tank['object_id']:
                        tank_info += f"\n     Volume: {vol_info['volume']}m³"
                        break
            
            message_parts.append(tank_info)
        message_parts.append("")
    
    # Add volume summary if available
    if volume_data and volume_data['total_volume'] > 0:
        message_parts.append("💧 *Volume Summary:*")
        message_parts.append(f"  • Tank Height: {height}m")
        message_parts.append(f"  • Volume Estimated: {volume_data['total_volume']}m³")
        message_parts.append("")
    
    # Add note about image
    message_parts.append("💾 *Note:* Annotated image saved locally for future reference.")
    
    return "\n".join(message_parts)


@app.post("/whatsapp")
async def whatsapp_webhook(request: Request, Body: str = Form(""), Latitude: str = Form(None), Longitude: str = Form(None), From: str = Form("")):
    """
    Main webhook endpoint for Twilio WhatsApp messages.
    
    Supports two input methods:
    1. Text format: "latitude,longitude,height" (e.g., "32.566846,35.933951,5")
    2. Two-step flow:
       - Step 1: Share WhatsApp location → Bot asks for height
       - Step 2: Send height → Bot analyzes with stored location
    """
    # Debug: Get all form data to see what Twilio sends
    form_data = await request.form()
    
    print(f"\n{'='*80}")
    print("NEW WHATSAPP MESSAGE RECEIVED")
    print(f"{'='*80}")
    print(f"All form data received: {dict(form_data)}")
    print(f"Body: {Body}")
    print(f"Latitude: {Latitude}")
    print(f"Longitude: {Longitude}")
    print(f"From: {From}")
    
    # Initialize response
    resp = MessagingResponse()
    
    # SCENARIO 1: User shared location via WhatsApp
    if Latitude and Longitude:
        try:
            lat = float(Latitude)
            lon = float(Longitude)
            print(f"✓ Received location share: ({lat}, {lon})")
            
            # Validate coordinates
            is_valid, error_msg = validate_coordinates(lat, lon)
            if not is_valid:
                resp.message(f"❌ {error_msg}")
                return Response(content=str(resp), media_type="application/xml")
            
            # Store location in session
            import time
            user_sessions[From] = {
                'lat': lat,
                'lon': lon,
                'timestamp': time.time()
            }
            
            # Ask user for height
            response_text = (
                "📍 Location received!\n\n"
                f"Latitude: {lat}\n"
                f"Longitude: {lon}\n\n"
                "Please send the tank height in meters.\n\n"
                "Example: 5"
            )
            print(f"Asking user for height: {response_text}")
            resp.message(response_text)
            return Response(content=str(resp), media_type="application/xml")
            
        except (ValueError, TypeError) as e:
            print(f"Failed to parse location share: {e}")
            resp.message("❌ Could not parse location data. Please try sharing your location again.")
            return Response(content=str(resp), media_type="application/xml")
    
    # SCENARIO 2: Check if user has a stored location and is sending height
    if From in user_sessions:
        session = user_sessions[From]
        
        # Try to parse height from message
        height = parse_height_only(Body)
        
        if height is not None and height > 0:
            # User sent valid height, use stored location
            lat = session['lat']
            lon = session['lon']
            
            print(f"✓ Using stored location ({lat}, {lon}) with height {height}m")
            
            # Clear session after use
            del user_sessions[From]
            
            # Proceed with analysis
            print(f"Starting WWTP analysis for coordinates: ({lat}, {lon}) with height: {height}m")
            
            try:
                # Run WWTP analysis
                results = analyze_wwtp(lat, lon, output_dir="Data")
                
                # Calculate volume if circular tanks were detected
                volume_data = None
                if results['success'] and results['circular_tanks']:
                    volume_data = calculate_tank_volume(results['circular_tanks'], height)
                    print(f"Volume calculation: {volume_data}")
                
                # Format response with volume information
                response_text = format_wwtp_response(results, height=height, volume_data=volume_data)
                
                print(f"\n{'='*80}")
                print("SENDING RESPONSE TO USER")
                print(f"{'='*80}")
                print(response_text)
                
                resp.message(response_text)
                return Response(content=str(resp), media_type="application/xml")
                
            except Exception as e:
                error_text = f"❌ An unexpected error occurred: {str(e)}"
                print(f"ERROR: {error_text}")
                resp.message(error_text)
                return Response(content=str(resp), media_type="application/xml")
        
        elif height is not None and height <= 0:
            # Invalid height
            resp.message("❌ Height must be a positive number (in meters). Please send a valid height.")
            return Response(content=str(resp), media_type="application/xml")
        
        else:
            # Could not parse height, but user has session
            resp.message(
                "❌ Invalid height value!\n\n"
                "Please send only the height in meters as a number.\n\n"
                "Example: 5"
            )
            return Response(content=str(resp), media_type="application/xml")
    
    # SCENARIO 3: User sent text in format "lat,lon,height"
    lat, lon, height = parse_location_data(Body)
    
    if lat is not None and lon is not None and height is not None:
        print(f"Parsed from text message: lat={lat}, lon={lon}, height={height}m")
        
        # Validate location data
        is_valid, error_msg = validate_location_data(lat, lon, height)
        
        if not is_valid:
            print(f"Validation failed: {error_msg}")
            resp.message(error_msg)
            return Response(content=str(resp), media_type="application/xml")
        
        # Send processing message
        print(f"Starting WWTP analysis for coordinates: ({lat}, {lon}) with height: {height}m")
        
        try:
            # Run WWTP analysis
            results = analyze_wwtp(lat, lon, output_dir="Data")
            
            # Calculate volume if circular tanks were detected
            volume_data = None
            if results['success'] and results['circular_tanks']:
                volume_data = calculate_tank_volume(results['circular_tanks'], height)
                print(f"Volume calculation: {volume_data}")
            
            # Format response with volume information
            response_text = format_wwtp_response(results, height=height, volume_data=volume_data)
            
            print(f"\n{'='*80}")
            print("SENDING RESPONSE TO USER")
            print(f"{'='*80}")
            print(response_text)
            
            resp.message(response_text)
            return Response(content=str(resp), media_type="application/xml")
            
        except Exception as e:
            error_text = f"❌ An unexpected error occurred: {str(e)}"
            print(f"ERROR: {error_text}")
            resp.message(error_text)
            return Response(content=str(resp), media_type="application/xml")
    
    # SCENARIO 4: Could not parse any valid input
    print("Failed to parse any valid input")
    error_msg = (
        "Hi,\n\n"
        "Please submit the location in one of these formats:\n\n"
        "1️⃣ Text format:\n"
        "   latitude,longitude,height\n"
        "   Example: 32.566846,35.933951,5\n\n"
        "2️⃣ Location sharing:\n"
        "   Share your location using WhatsApp, then send the height when prompted."
    )
    resp.message(error_msg)
    return Response(content=str(resp), media_type="application/xml")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "WhatsApp WWTP Bot is running", "version": "1.0"}


if __name__ == "__main__":
    print("="*80)
    print("STARTING WHATSAPP WWTP BOT")
    print("="*80)
    print("Make sure:")
    print("  1. ngrok is running: ngrok http 5000")
    print("  2. Twilio webhook is configured with ngrok URL + /whatsapp")
    print("="*80)
    
    # Running on port 5000 to match ngrok setup
    uvicorn.run(app, host="0.0.0.0", port=5000)
