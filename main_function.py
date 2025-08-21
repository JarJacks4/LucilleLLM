import functions_framework
import os
import json
from datetime import datetime
import uuid

# Cloud Functions entry point
@functions_framework.http
def lucillellm(request):
    """Simple Cloud Functions entry point for LucilleLLM"""
    # Handle CORS
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)
    
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST',
        'Access-Control-Allow-Headers': 'Content-Type'
    }
    
    try:
        # Simple health check
        if request.path == '/health' or request.path == '/':
            return (json.dumps({
                "status": "healthy",
                "message": "LucilleLLM Cloud Function is running",
                "timestamp": datetime.now().isoformat()
            }), 200, headers)
        
        # Simple chat endpoint
        if request.path == '/chat' and request.method == 'POST':
            data = request.get_json()
            if not data:
                return (json.dumps({"error": "No JSON data provided"}), 400, headers)
            
            message = data.get('message', '')
            session_id = data.get('session_id', str(uuid.uuid4()))
            
            if not message:
                return (json.dumps({"error": "Message is required"}), 400, headers)
            
            # Simple response for testing
            response = {
                "session_id": session_id,
                "response": f"Hello! I'm Lucille, your self-care assistant. You said: '{message}'. This is a test response from Cloud Functions.",
                "conversation": [message, f"Hello! I'm Lucille, your self-care assistant. You said: '{message}'. This is a test response from Cloud Functions."]
            }
            
            return (json.dumps(response), 200, headers)
        
        # Default response
        return (json.dumps({
            "message": "LucilleLLM Cloud Function",
            "endpoints": ["/health", "/chat"],
            "timestamp": datetime.now().isoformat()
        }), 200, headers)
        
    except Exception as e:
        print(f"Error in function: {e}")
        return (json.dumps({"error": str(e)}), 500, headers)
