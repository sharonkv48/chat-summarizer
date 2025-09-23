from fastapi import FastAPI, HTTPException
import requests
import datetime
from typing import List, Dict
import os

app = FastAPI(title="Slack Data Pipeline", version="1.0.0")

# Configuration - Built-in tokens
SLACK_TOKEN = "xoxb-9561597131411-9551575917063-efEYI6uNXgqUskqO1rEujOYo"
DEFAULT_CHANNEL_ID = "C09GN1EHH6W"
SLACK_BASE_URL = "https://slack.com/api"

def fetch_slack_messages(channel_id: str) -> List[Dict]:
    """Fetch raw messages from Slack API"""
    headers = {"Authorization": f"Bearer {SLACK_TOKEN}"}
    url = f"{SLACK_BASE_URL}/conversations.history?channel={channel_id}"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("ok"):
            raise Exception(f"Slack API error: {data.get('error', 'Unknown error')}")
            
        return data.get("messages", [])
        
    except requests.exceptions.Timeout:
        raise Exception("Slack API timeout")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Slack API connection error: {str(e)}")

def clean_message(message: Dict) -> Dict:
    """Clean and format a single message"""
    ts = message.get("ts")
    
    # Convert timestamp to readable format
    if ts:
        timestamp_float = float(ts)
        dt = datetime.datetime.fromtimestamp(timestamp_float, tz=datetime.timezone.utc)
        readable_date = dt.strftime('%Y-%m-%d %H:%M:%S')  # Removed UTC
    else:
        readable_date = "Unknown"
    
    return {
        "timestamp": readable_date,
        "user": message.get("user", "Unknown"),
        "text": message.get("text", "")
    }

def clean_messages(messages: List[Dict]) -> List[Dict]:
    """Clean all messages"""
    cleaned = []
    for msg in messages:
        # Skip unwanted message types if needed
        if msg.get('type') != 'message':
            continue
            
        cleaned_msg = clean_message(msg)
        cleaned.append(cleaned_msg)
    
    return cleaned

@app.get("/")
def root():
    """Root endpoint with basic info"""
    return {
        "message": "Slack Data Pipeline API",
        "default_channel": DEFAULT_CHANNEL_ID,
        "docs_url": "/docs"
    }

@app.get("/messages/{channel_id}")
def get_messages(channel_id: str):
    """
    Fetch and return cleaned messages from a Slack channel
    - channel_id: Slack channel ID (e.g., C09GN1EHH6W)
    """
    try:
        # Validate inputs
        if not channel_id.startswith('C'):
            raise HTTPException(status_code=400, detail="Invalid channel ID format")
        
        # Fetch from Slack
        raw_messages = fetch_slack_messages(channel_id)
        
        # Clean and format
        cleaned_messages = clean_messages(raw_messages)
        
        return cleaned_messages
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/messages")
def get_default_channel_messages():
    """Get messages from the default channel"""
    return get_messages(DEFAULT_CHANNEL_ID)

@app.get("/health")
def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)