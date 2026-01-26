# WWTP WhatsApp Bot

A WhatsApp-based chatbot for automated detection and measurement of circular tanks in Wastewater Treatment Plant (WWTP) facilities using satellite imagery, YOLO object detection, and Google Gemini AI analysis.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Data Storage](#data-storage)
- [CI/CD Pipeline](#cicd-pipeline)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project provides an intelligent WhatsApp chatbot that allows users to analyze WWTP facilities by simply sharing their location. The bot:
1. Downloads high-resolution satellite imagery for the specified location
2. Performs YOLO object detection to identify circular tanks
3. Analyzes the imagery using Google Gemini AI
4. Calculates tank dimensions (radius, surface area, volume) based on user-provided height data
5. Stores results and images in Google Cloud Storage
6. Generates Excel reports with detailed measurements

## ✨ Features

- **Location-based Analysis**: Users can share GPS coordinates or text-based location data
- **Automated Satellite Imagery Download**: Fetches high-resolution imagery from satellite sources
- **YOLO Object Detection**: Identifies and annotates circular tanks in WWTP facilities
- **AI-Powered Analysis**: Uses Google Gemini to provide detailed insights
- **Tank Measurements**: Calculates radius, surface area, and volume for detected tanks
- **Image Collection**: Allows users to upload three additional site photos after analysis
- **Real-time Progress Updates**: Sends "Processing..." messages during analysis
- **Cloud Storage Integration**: Stores all images and results in GCP buckets
- **Excel Report Generation**: Creates per-user Excel files with tank measurements
- **Session Management**: Handles multi-message conversations with timeout handling
- **Scalable Architecture**: Hybrid deployment with Cloud Run and VM workers

## 🏗️ Architecture

### Hybrid Cloud Architecture

```
┌─────────────────┐
│  WhatsApp User  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Twilio API     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  GCP Cloud Run Service          │
│  (FastAPI + WhatsApp Bot)       │
│  - Message handling             │
│  - Gemini AI integration        │
│  - Session management           │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Google Cloud Storage           │
│  - Satellite imagery            │
│  - YOLO predictions             │
│  - User-uploaded images         │
│  - Excel reports                │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  VM Worker (Optional)           │
│  - YOLO detection processing    │
│  - Heavy computation tasks      │
└─────────────────────────────────┘
```

### Component Breakdown

1. **Cloud Run Service**: Hosts the FastAPI application handling WhatsApp webhooks
2. **GCP Storage**: Stores images, predictions, and Excel reports
3. **VM Worker**: Optional compute instance for YOLO processing (referenced in workflow)
4. **Twilio**: WhatsApp messaging gateway

## 🛠️ Technology Stack

### Backend
- **Python 3.11**: Core programming language
- **FastAPI**: Web framework for API endpoints
- **Uvicorn**: ASGI server

### AI/ML
- **YOLO (YOLOv8)**: Object detection for circular tanks
- **Google Gemini AI**: Image analysis and insights
- **GDAL**: Geospatial data processing

### Cloud Services
- **Google Cloud Platform (GCP)**
  - Cloud Run: Serverless container hosting
  - Cloud Storage: Object storage
  - Artifact Registry: Docker image repository
- **Twilio**: WhatsApp Business API

### DevOps
- **Docker**: Containerization
- **GitHub Actions**: CI/CD automation
- **ngrok**: Local development tunneling (optional)

### Data Processing
- **pandas**: Data manipulation
- **openpyxl**: Excel file generation
- **Pillow**: Image processing

## 📦 Prerequisites

Before setting up the project, ensure you have:

1. **Google Cloud Platform Account**
   - Project with billing enabled
   - Service account with necessary permissions
   - Cloud Run, Cloud Storage, and Artifact Registry APIs enabled

2. **Twilio Account**
   - WhatsApp Business API access
   - Account SID and Auth Token

3. **Development Environment**
   - Python 3.11 or higher
   - Docker and Docker Compose
   - Git

4. **API Keys**
   - Google Gemini API key
   - GCP service account JSON key

## 🚀 Installation

### Local Development Setup

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd wwtp_bot
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate Virtual Environment**
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source .venv/bin/activate
     ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Install GDAL (Windows)**
   ```bash
   pip install GDAL-3.8.4-cp311-cp311-win_amd64.whl
   ```

6. **Configure Environment Variables**
   - Copy `.env.example` to `.env`
   - Fill in all required credentials (see [Configuration](#configuration))

### Docker Setup

1. **Build Docker Image**
   ```bash
   docker-compose build
   ```

2. **Run Container**
   ```bash
   docker-compose up
   ```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# GCP Configuration
GCP_PROJECT_ID=your-gcp-project-id
GCP_BUCKET_NAME=your-bucket-name
GCP_BUCKET_PATH=bot-dump
GCP_SERVICE_ACCOUNT_KEY_PATH=path/to/service-account-key.json

# Google AI
GOOGLE_API_KEY=your-gemini-api-key

# Twilio Configuration
TWILIO_ACCOUNT_SID=your-twilio-account-sid
TWILIO_AUTH_TOKEN=your-twilio-auth-token
TWILIO_WHATSAPP_NUMBER=whatsapp:+1234567890

# Application Settings
PORT=5000
```

### GitHub Secrets

For CI/CD deployment, configure the following secrets in your GitHub repository:

| Secret Name | Description |
|-------------|-------------|
| `GCP_PROJECT_ID` | Your GCP project ID |
| `GCP_SA_KEY` | Service account JSON key (entire content) |
| `GCP_BUCKET_NAME` | GCS bucket name for storage |
| `GCP_BUCKET_PATH` | Folder path within bucket (e.g., `bot-dump`) |
| `GOOGLE_API_KEY` | Google Gemini API key |
| `TWILIO_ACCOUNT_SID` | Twilio account identifier |
| `TWILIO_AUTH_TOKEN` | Twilio authentication token |
| `TWILIO_WHATSAPP_NUMBER` | Twilio WhatsApp number (format: `whatsapp:+1234567890`) |

## 🚢 Deployment

### Automatic Deployment (CI/CD)

The project uses GitHub Actions for automated deployment to GCP Cloud Run.

#### Workflow Trigger
Deployment is triggered automatically on push to the `main` branch.

#### Deployment Steps

1. **Checkout Code**: Retrieves the latest code from the repository
2. **GCP Authentication**: Authenticates using service account credentials
3. **Docker Build & Push**: 
   - Builds Docker image with SHA tag
   - Pushes to GCP Artifact Registry (`us-central1-docker.pkg.dev`)
4. **Cloud Run Deployment**:
   - Service: `whatsapp-bot-service`
   - Region: `us-central1`
   - Resources: 4GB RAM, 2 CPUs
   - Timeout: 600 seconds
   - Port: 5000
5. **Worker Code Upload**: Uploads Python backend files to GCS bucket
6. **VM Restart** (Optional): Restarts worker VM if configured

#### Manual Deployment

To deploy manually:

```bash
# Authenticate with GCP
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Build and push Docker image
REGION=us-central1
PROJECT_ID=your-project-id
REPO_NAME=whatsapp-repo
SERVICE=whatsapp-bot-service
TAG=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$SERVICE:latest

docker build -t $TAG .
docker push $TAG

# Deploy to Cloud Run
gcloud run deploy $SERVICE \
  --image $TAG \
  --region $REGION \
  --memory 4Gi \
  --cpu 2 \
  --timeout 600s \
  --port 5000 \
  --set-env-vars "GCP_PROJECT_ID=$PROJECT_ID,..."
```

## 📱 Usage

### User Interaction Flow

1. **Send Location**
   - User sends GPS location via WhatsApp
   - OR sends text coordinates (e.g., "6.9271, 79.8612")

2. **Processing**
   - Bot sends "Processing..." message
   - Downloads satellite imagery
   - Runs YOLO detection
   - Performs Gemini AI analysis

3. **Image Preview**
   - Bot sends annotated image showing detected tanks
   - Prompts user for tank height data

4. **Height Input**
   - User provides height for each detected tank
   - Bot calculates radius, surface area, and volume

5. **AI Summary**
   - Gemini provides detailed analysis
   - Final summary sent to user

6. **Image Collection**
   - Bot requests 3 additional site photos
   - User uploads images (5-minute session timeout)
   - Confirmation message sent upon completion

### Example Conversation

```
User: [Shares Location]

Bot: Processing your request. Downloading satellite imagery and performing analysis...

Bot: [Sends annotated image with detected tanks]
     "I found 3 circular tanks. Please provide the height (in meters) for each tank.
     Tank 1 height:"

User: 5.2

Bot: "Tank 2 height:"

User: 4.8

Bot: "Tank 3 height:"

User: 5.0

Bot: [Sends calculation results and Gemini analysis]

Bot: "Please upload 3 images of the WWTP site."

User: [Uploads 3 images]

Bot: "Thank you! All images received and saved."
```

## 📁 Project Structure

```
wwtp_bot/
├── backend/
│   ├── whatsapp_bot.py          # Main WhatsApp bot logic
│   ├── satellite_downloader.py  # Satellite imagery download
│   ├── yolo_detector.py         # YOLO object detection
│   ├── gemini_analyzer.py       # Gemini AI integration
│   ├── gcp_storage.py           # GCS upload/download utilities
│   ├── session_manager.py       # User session handling
│   └── excel_generator.py       # Excel report generation
├── .github/
│   └── workflows/
│       └── deploy.yml           # GitHub Actions CI/CD workflow
├── .venv/                       # Virtual environment (local)
├── .env                         # Environment variables (not committed)
├── .env.example                 # Environment variables template
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker container definition
├── docker-compose.yml           # Docker Compose configuration
├── GDAL-3.8.4-...whl           # GDAL wheel file (Windows)
└── README.md                    # This file
```

## 🔌 API Endpoints

### Health Check
- **Endpoint**: `GET /`
- **Response**: `{"status": "ok", "service": "WhatsApp Bot"}`
- **Purpose**: Verify service is running

### WhatsApp Webhook
- **Endpoint**: `POST /webhook/whatsapp`
- **Content-Type**: `application/x-www-form-urlencoded`
- **Response**: XML (TwiML)
- **Purpose**: Handle incoming WhatsApp messages from Twilio

## 💾 Data Storage

### GCS Bucket Structure

```
gs://your-bucket-name/bot-dump/
├── images/                                    # Downloaded satellite imagery
│   └── {timestamp}_{phone_number}.jpg
├── predictions/                               # YOLO annotated images
│   └── {timestamp}_{phone_number}_pred.jpg
├── loc_wwtp_img/                             # User-uploaded images
│   └── {submission_id}_{timestamp}/
│       ├── image_1.jpg
│       ├── image_2.jpg
│       └── image_3.jpg
├── results/                                   # Tank measurements (per user)
│   └── tank_measurements_{phone_number}.xlsx
└── results_wwtp_loc/                         # Master records
    └── location_images_master.xlsx
```

### Excel File Schemas

#### `tank_measurements_{phone_number}.xlsx`
| Column | Description |
|--------|-------------|
| Phone Number | User's WhatsApp number |
| Timestamp | Analysis timestamp |
| WWTP Latitude | Original location latitude |
| WWTP Longitude | Original location longitude |
| Tank Center Longitude | Individual tank longitude |
| Tank Center Latitude | Individual tank latitude |
| Tank Height (m) | User-provided height |
| Tank Radius (m) | Calculated radius |
| Surface Area (m²) | Calculated surface area |
| Volume (m³) | Calculated volume |

#### `location_images_master.xlsx`
| Column | Description |
|--------|-------------|
| Submission ID | Unique submission identifier |
| Phone Number | User's WhatsApp number |
| Image Folder Path | GCS path to uploaded images |

## 🔄 CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/deploy.yml`) automates the entire deployment process:

### Pipeline Stages

1. **Build Stage**
   - Checkout source code
   - Authenticate with GCP
   - Configure Docker for Artifact Registry
   - Build Docker image with commit SHA tag
   - Push image to registry

2. **Deploy Stage**
   - Deploy to Cloud Run with environment variables
   - Configure compute resources (4GB RAM, 2 CPUs)
   - Set timeout and port settings

3. **Post-Deploy Stage**
   - Upload backend Python files to GCS
   - Upload requirements.txt
   - Optional: Restart worker VM

### Deployment Monitoring

- View logs: `gcloud run logs read whatsapp-bot-service --region us-central1`
- Check service status: `gcloud run services describe whatsapp-bot-service --region us-central1`
- Monitor GitHub Actions: Repository → Actions tab

## 🐛 Troubleshooting

### Common Issues

#### 1. Docker Permission Denied
**Problem**: `uvicorn: permission denied` error when running container

**Solution**: Check Dockerfile permissions and ensure uvicorn is properly installed in the container environment

#### 2. Twilio Content-Type Error
**Problem**: WhatsApp bot not responding, Twilio expects `application/xml`

**Solution**: Ensure `media_type="application/xml"` is set in the response in `whatsapp_bot.py`

#### 3. GDAL Installation Issues
**Problem**: GDAL fails to install via pip

**Solution**: 
- Windows: Use the provided `.whl` file
- Linux/Docker: Install system dependencies first (`libgdal-dev`)

#### 4. Session Timeout
**Problem**: User image upload session expires

**Solution**: Session timeout is 5 minutes. Ensure users upload all 3 images within this timeframe.

#### 5. GCS Upload Failures
**Problem**: Images fail to upload to Google Cloud Storage

**Solution**: 
- Verify service account has Storage Object Creator role
- Check GCS bucket permissions
- Retry logic (2 attempts) is built-in

### Debug Mode

Enable debug logging by modifying the application:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Access Logs

**Cloud Run Logs**:
```bash
gcloud run logs read whatsapp-bot-service --region us-central1 --limit 50
```

**Local Logs**:
- Check console output when running via `docker-compose up`

### Local Development with ngrok

For testing WhatsApp webhooks locally:

```bash
# Start ngrok tunnel
ngrok http 5000

# Copy the ngrok URL and configure in Twilio sandbox
# Example: https://weakishly-ungrounded-alex.ngrok-free.dev/webhook/whatsapp
```

## 🤝 Contributing

### Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make changes and test locally
3. Run tests (if available): `pytest`
4. Commit with descriptive messages
5. Push and create a pull request
6. Wait for CI/CD checks to pass
7. Request review from maintainers

### Code Style

- Follow PEP 8 for Python code
- Use type hints where applicable
- Add docstrings to functions and classes
- Keep functions focused and modular

### Testing

- Test locally with Docker Compose
- Use ngrok for webhook testing
- Verify all environment variables are properly set
- Test with actual WhatsApp messages before deploying

## 👥 Maintainers

- Akila
- Contact: akilapremarathna0@gmail.com

## 🙏 Acknowledgments

- **Google Gemini AI** for image analysis capabilities
- **Twilio** for WhatsApp Business API
- **Ultralytics YOLO** for object detection
- **Google Cloud Platform** for cloud infrastructure

---

**Last Updated**: January 2026

For questions or issues, please open a GitHub issue or contact the maintainers.




