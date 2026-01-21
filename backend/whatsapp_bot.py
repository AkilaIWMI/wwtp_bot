"""
WhatsApp Bot for WWTP Analysis
Integrates Twilio WhatsApp with WWTP satellite image analysis pipeline

New Flow:
1. User sends location (lat,lon or shares location) - NO height
2. Bot downloads satellite image and runs YOLO detection
3. If circular tanks detected, bot prompts for height of each tank one by one
4. After all heights collected, bot shows summary with radius, area, volume
5. Bot runs Gemini AI analysis
"""

import os
import re
import time
import math
import uvicorn
from fastapi import FastAPI, Form, Response, Request
from twilio.twiml.messaging_response import MessagingResponse
from wwtp_analysis import analyze_wwtp
from gemini_analysis import analyze_image_with_gemini

app = FastAPI()

# Session storage to track user states
# Key: phone number (From field)
# Value: {
#     'lat': float,
#     'lon': float,
#     'results': dict (YOLO analysis results),
#     'circular_tanks': list (detected circular tanks),
#     'tank_heights': dict (tank_id -> height),
#     'current_tank_index': int,
#     'annotated_image_path': str,
#     'satellite_image_path': str,
#     'timestamp': float
# }
user_sessions = {}

# Session timeout: 30 minutes
SESSION_TIMEOUT = 1800


def parse_lat_lon_only(message):
    """
    Parse latitude and longitude only (no height) from WhatsApp message.
    
    Expected format: "latitude,longitude" (e.g., "32.566846,35.933951")
    
    Args:
        message (str): The incoming message text
        
    Returns:
        tuple: (latitude, longitude) or (None, None) if parsing fails
    """
    # Pattern: two numbers separated by comma
    # Each number can be positive/negative with optional decimal
    pattern = r'^\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*$'
    match = re.match(pattern, message.strip())
    
    if match:
        try:
            lat = float(match.group(1))
            lon = float(match.group(2))
            return lat, lon
        except ValueError:
            return None, None
    
    return None, None


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


def validate_height_input(message):
    """
    Validate that input is a positive numerical value for height.
    
    Args:
        message (str): The incoming message text
        
    Returns:
        tuple: (height, is_valid, error_message)
    """
    # Try to parse a single number
    pattern = r'^\s*(-?\d+\.?\d*)\s*$'
    match = re.match(pattern, message.strip())
    
    if match:
        try:
            height = float(match.group(1))
            if height <= 0:
                return None, False, "❌ Height must be a positive number (in meters). Please enter a valid height."
            return height, True, None
        except ValueError:
            return None, False, "❌ Invalid input! Please enter only a numerical value for height (in meters)."
    
    return None, False, "❌ Invalid input! Please enter only a numerical value for height (in meters)."


def calculate_tank_volumes(circular_tanks, tank_heights):
    """
    Calculate volume for each circular tank using individual heights.
    
    Formula: V = π × r² × h
    
    Args:
        circular_tanks (list): List of circular tank data with measurements
        tank_heights (dict): Dictionary mapping tank_id to height in meters
        
    Returns:
        dict: Dictionary containing:
            - total_volume (float): Total volume of all tanks in cubic meters
            - tank_data (list): List of individual tank data with radius, area, volume
    """
    if not circular_tanks:
        return {
            'total_volume': 0,
            'tank_data': []
        }
    
    tank_data = []
    total_volume = 0
    
    for tank in circular_tanks:
        tank_id = tank['tank_id']
        radius = tank['measurements']['radius_meters']
        surface_area = tank['measurements']['surface_area_sqm']
        height = tank_heights.get(tank_id, 0)
        
        # V = π × r² × h
        volume = math.pi * (radius ** 2) * height
        
        tank_data.append({
            'tank_id': tank_id,
            'radius': radius,
            'surface_area': round(surface_area, 2),
            'height': height,
            'volume': round(volume, 2)
        })
        
        total_volume += volume
    
    return {
        'total_volume': round(total_volume, 2),
        'tank_data': tank_data
    }


def format_detection_message(results):
    """
    Format initial detection message after YOLO analysis.
    
    Args:
        results (dict): Results from analyze_wwtp function
        
    Returns:
        str: Formatted message for WhatsApp
    """
    if not results['success']:
        return f"❌ Analysis failed: {results['error']}"
    
    wwtp_detected = results['wwtp_detected']
    circular_tanks = results['circular_tanks']
    detection_counts = results['detection_counts']
    
    # Scenario 1: No WWTP and no circular tanks
    if not wwtp_detected and len(circular_tanks) == 0:
        return "🔍 *Analysis Complete*\n\nNo WWTP facilities or circular tanks detected at this location."
    
    # Scenario 2: WWTP detected but no circular tanks
    if wwtp_detected and len(circular_tanks) == 0:
        message_parts = ["✅ *WWTP Detected*\n"]
        message_parts.append("A wastewater treatment plant facility has been identified at this location.")
        message_parts.append("\nHowever, no circular tanks were detected in the satellite imagery.")
        
        # Add detection summary
        if detection_counts:
            message_parts.append("\n📊 *Detected Objects:*")
            for class_name, count in detection_counts.items():
                message_parts.append(f"  • {class_name}: {count}")
        
        return "\n".join(message_parts)
    
    # Scenario 3: Circular tanks detected (with or without WWTP)
    message_parts = ["✅ *Detection Complete*\n"]
    
    # Add detection summary
    if detection_counts:
        message_parts.append("📊 *Detected Objects:*")
        for class_name, count in detection_counts.items():
            message_parts.append(f"  • {class_name}: {count}")
        message_parts.append("")
    
    # Add circular tank info
    message_parts.append(f"🔵 *{len(circular_tanks)} Circular Tank(s) Detected*\n")
    message_parts.append("I've identified the tanks and labeled them with unique IDs (#1, #2, #3, etc.).")
    message_parts.append("\n📏 *Next Step:*")
    message_parts.append("Please provide the height (in meters) for each tank.")
    message_parts.append(f"\n*Tank #1:* Enter height in meters")
    
    return "\n".join(message_parts)


def format_final_summary(circular_tanks, volume_data, gemini_data=None):
    """
    Format final summary with all measurements and AI analysis.
    
    Args:
        circular_tanks (list): List of circular tank data
        volume_data (dict): Volume calculation results
        gemini_data (dict): Gemini AI analysis results (optional)
        
    Returns:
        str: Formatted summary message
    """
    message_parts = ["✅ *WWTP Analysis Complete*\n"]
    
    # Add detailed tank information
    message_parts.append(f"📊 *Tank Measurements Summary ({len(circular_tanks)} tanks):*\n")
    
    for tank_info in volume_data['tank_data']:
        tank_id = tank_info['tank_id']
        message_parts.append(f"*Tank #{tank_id}:*")
        message_parts.append(f"  • Radius: {tank_info['radius']}m")
        message_parts.append(f"  • Surface Area: {tank_info['surface_area']}m²")
        message_parts.append(f"  • Height: {tank_info['height']}m")
        message_parts.append(f"  • Volume: {tank_info['volume']}m³")
        message_parts.append("")
    
    # Add total volume
    if volume_data['total_volume'] > 0:
        message_parts.append("💧 *Total Volume:*")
        message_parts.append(f"  {volume_data['total_volume']}m³")
        message_parts.append("")
    
    # Add Gemini AI analysis if available
    if gemini_data and gemini_data.get('error') is None:
        message_parts.append("🤖 *AI Analysis:*")
        
        # WWTP detection
        is_wwtp = gemini_data.get('is_wwtp')
        if is_wwtp is not None:
            wwtp_status = "Yes ✓" if is_wwtp else "No ✗"
            message_parts.append(f"  • WWTP Detected: {wwtp_status}")
        
        # Circular tank count
        ai_tank_count = gemini_data.get('circular_tank_count', 0)
        message_parts.append(f"  • Circular Tanks (AI Count): {ai_tank_count}")
        
        # Description
        description = gemini_data.get('description', '')
        if description:
            message_parts.append(f"  • Description: {description}")
        
        message_parts.append("")
        
        # AI Disclaimer
        message_parts.append("⚠️ *AI Disclaimer:* AI analysis is experimental and may contain errors. Please verify critical information.")
        message_parts.append("")
    
    # Add note about images
    message_parts.append("💾 *Note:* Annotated images saved locally for future reference.")
    
    return "\n".join(message_parts)


def clean_old_sessions():
    """Remove sessions older than SESSION_TIMEOUT."""
    current_time = time.time()
    to_remove = []
    
    for phone, session in user_sessions.items():
        if current_time - session['timestamp'] > SESSION_TIMEOUT:
            to_remove.append(phone)
    
    for phone in to_remove:
        del user_sessions[phone]
        print(f"Cleaned up old session for {phone}")


@app.post("/whatsapp")
async def whatsapp_webhook(request: Request, Body: str = Form(""), Latitude: str = Form(None), Longitude: str = Form(None), From: str = Form("")):
    """
    Main webhook endpoint for Twilio WhatsApp messages.
    
    New Flow:
    1. User sends location (lat,lon text or WhatsApp location share)
    2. Bot runs YOLO detection immediately
    3. If circular tanks detected, bot asks for heights one by one
    4. After all heights collected, bot shows final summary
    """
    # Debug: Get all form data
    form_data = await request.form()
    
    print(f"\n{'='*80}")
    print("NEW WHATSAPP MESSAGE RECEIVED")
    print(f"{'='*80}")
    print(f"All form data received: {dict(form_data)}")
    print(f"Body: {Body}")
    print(f"Latitude: {Latitude}")
    print(f"Longitude: {Longitude}")
    print(f"From: {From}")
    
    # Clean old sessions periodically
    clean_old_sessions()
    
    # Initialize response
    resp = MessagingResponse()
    
    # --------------------------------------------------
    # SCENARIO 1: User shared location via WhatsApp
    # --------------------------------------------------
    if Latitude and Longitude:
        try:
            lat = float(Latitude)
            lon = float(Longitude)

            lat = round(lat, 4)
            lon = round(lon, 4)
            print(f"✓ Received location share: ({lat}, {lon})")
            
            # Validate coordinates
            is_valid, error_msg = validate_coordinates(lat, lon)
            if not is_valid:
                resp.message(f"❌ {error_msg}")
                return Response(content=str(resp), media_type="application/xml")
            
            # Run WWTP analysis immediately
            print(f"Starting WWTP analysis for coordinates: ({lat}, {lon})")
            results = analyze_wwtp(lat, lon, output_dir="Data")
            
            # Check results and handle accordingly
            circular_tanks = results.get('circular_tanks', [])
            
            # Send detection message (only ONE message per response)
            detection_message = format_detection_message(results)
            
            if len(circular_tanks) == 0:
                # No circular tanks detected - send message and end
                resp.message(detection_message)
                return Response(content=str(resp), media_type="application/xml")
            
            # Circular tanks detected - start height collection
            # Store session
            user_sessions[From] = {
                'lat': lat,
                'lon': lon,
                'results': results,
                'circular_tanks': circular_tanks,
                'tank_heights': {},
                'current_tank_index': 0,
                'annotated_image_path': results.get('annotated_image_path'),
                'satellite_image_path': results.get('satellite_image_path'),
                'timestamp': time.time()
            }
            
            # Send detection message (already created above)
            resp.message(detection_message)
            return Response(content=str(resp), media_type="application/xml")
            
        except Exception as e:
            print(f"Failed to process location share: {e}")
            resp.message(f"❌ Error processing location: {str(e)}")
            return Response(content=str(resp), media_type="application/xml")
    
    # --------------------------------------------------
    # SCENARIO 2: User is in height collection flow
    # --------------------------------------------------
    if From in user_sessions:
        session = user_sessions[From]
        circular_tanks = session['circular_tanks']
        current_index = session['current_tank_index']
        
        # Validate height input
        height, is_valid, error_msg = validate_height_input(Body)
        
        if not is_valid:
            # Invalid height - ask again for the same tank
            current_tank_id = circular_tanks[current_index]['tank_id']
            resp.message(f"{error_msg}\n\n*Tank #{current_tank_id}:* Please enter a valid height in meters.")
            return Response(content=str(resp), media_type="application/xml")
        
        # Valid height - store it
        current_tank_id = circular_tanks[current_index]['tank_id']
        session['tank_heights'][current_tank_id] = height
        print(f"✓ Stored height {height}m for Tank #{current_tank_id}")
        
        # Move to next tank
        session['current_tank_index'] += 1
        
        # Check if we need more heights
        if session['current_tank_index'] < len(circular_tanks):
            # Ask for next tank height
            next_tank_id = circular_tanks[session['current_tank_index']]['tank_id']
            resp.message(f"✓ Tank #{current_tank_id}: {height}m recorded.\n\n*Tank #{next_tank_id}:* Enter height in meters")
            return Response(content=str(resp), media_type="application/xml")
        
        # All heights collected - generate final summary
        print(f"\n{'='*80}")
        print("All heights collected. Generating final summary...")
        print(f"{'='*80}")
        
        # Calculate volumes
        volume_data = calculate_tank_volumes(circular_tanks, session['tank_heights'])
        print(f"Volume calculation: {volume_data}")
        
        # Run Gemini AI analysis
        gemini_data = None
        if session['satellite_image_path']:
            print(f"\n{'='*80}")
            print("Running Gemini AI Analysis")
            print(f"{'='*80}")
            try:
                gemini_data = analyze_image_with_gemini(session['satellite_image_path'])
            except Exception as e:
                print(f"Gemini analysis failed: {e}")
        
        # Format final summary
        final_message = format_final_summary(circular_tanks, volume_data, gemini_data)
        
        print(f"\n{'='*80}")
        print("SENDING FINAL SUMMARY TO USER")
        print(f"{'='*80}")
        print(final_message)
        
        # Clear session
        del user_sessions[From]
        
        resp.message(final_message)
        return Response(content=str(resp), media_type="application/xml")
    
    # --------------------------------------------------
    # SCENARIO 3: User sent text in format "lat,lon"
    # --------------------------------------------------
    lat, lon = parse_lat_lon_only(Body)
    
    if lat is not None and lon is not None:
        print(f"Parsed from text message: lat={lat}, lon={lon}")
        
        # Validate coordinates
        is_valid, error_msg = validate_coordinates(lat, lon)
        
        if not is_valid:
            print(f"Validation failed: {error_msg}")
            resp.message(error_msg)
            return Response(content=str(resp), media_type="application/xml")
        
        # Run WWTP analysis immediately
        print(f"Starting WWTP analysis for coordinates: ({lat}, {lon})")
        
        try:
            results = analyze_wwtp(lat, lon, output_dir="Data")
            
            # Check results
            circular_tanks = results.get('circular_tanks', [])
            
            # Send detection message (only ONE message per response)
            detection_message = format_detection_message(results)
            
            if len(circular_tanks) == 0:
                # No circular tanks - send message and end
                resp.message(detection_message)
                return Response(content=str(resp), media_type="application/xml")
            
            # Circular tanks detected - start height collection
            user_sessions[From] = {
                'lat': lat,
                'lon': lon,
                'results': results,
                'circular_tanks': circular_tanks,
                'tank_heights': {},
                'current_tank_index': 0,
                'annotated_image_path': results.get('annotated_image_path'),
                'satellite_image_path': results.get('satellite_image_path'),
                'timestamp': time.time()
            }
            
            # Send detection message (already created above)
            resp.message(detection_message)
            return Response(content=str(resp), media_type="application/xml")
            
        except Exception as e:
            error_text = f"❌ An unexpected error occurred: {str(e)}"
            print(f"ERROR: {error_text}")
            resp.message(error_text)
            return Response(content=str(resp), media_type="application/xml")
    
    # --------------------------------------------------
    # SCENARIO 4: Could not parse any valid input
    # --------------------------------------------------
    print("Failed to parse any valid input")
    error_msg = (
        "Hi!\n\n"
        "Please submit your location in one of these formats:\n\n"
        "1. Text format: latitude,longitude\n"
        "   Example: 32.566846,35.933951\n\n"
        "2. Location sharing: Share your location using WhatsApp."
    )
    resp.message(error_msg)
    
    # Debug: Print the XML response being sent
    twiml_xml = str(resp)
    print(f"\n{'='*80}")
    print("XML RESPONSE TO TWILIO:")
    print(twiml_xml)
    print(f"{'='*80}\n")
    
    return Response(content=twiml_xml, media_type="application/xml")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "WhatsApp WWTP Bot is running", "version": "2.0"}


if __name__ == "__main__":
    print("="*80)
    print("STARTING WHATSAPP WWTP BOT - VERSION 2.0")
    print("="*80)
    print("New Flow:")
    print("  1. User sends location (no height)")
    print("  2. Bot downloads satellite image and runs YOLO")
    print("  3. Bot prompts for height of each circular tank")
    print("  4. Bot shows final summary with volumes")
    print("="*80)
    print("Make sure:")
    print("  1. ngrok is running: ngrok http 5000")
    print("  2. Twilio webhook is configured with ngrok URL + /whatsapp")
    print("="*80)
    
    # Running on port 5000 to match ngrok setup
    uvicorn.run(app, host="0.0.0.0", port=5000)
