"""
LucilleLLM - Self-Care Chatbot API

A FastAPI-based chatbot that provides self-care advice and wellbeing support.
Features OpenAI integration, FAISS vector search, and Firebase session management.
"""

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Optional
from dotenv import load_dotenv
import os
import uuid
from collections import defaultdict
from datetime import datetime

# OpenAI and LangChain imports
from openai import OpenAI, APIConnectionError, RateLimitError
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.schema import AIMessage, HumanMessage
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
import tiktoken
import httpx

# Local imports
from firebase_service import get_firebase_service

# Load environment variables
load_dotenv()

def get_openai_api_key():
    """Get OpenAI API key from environment or Google Secret Manager"""
    # Try local environment first
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key.strip()
    
    # Try Google Secret Manager in production
    try:
        from google.cloud import secretmanager
        import google.auth
        
        client = secretmanager.SecretManagerServiceClient()
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        
        # If no project ID set, try to infer it
        if not project_id:
            try:
                creds, inferred_project = google.auth.default()
                if inferred_project:
                    project_id = inferred_project
            except Exception:
                pass
        
        if project_id:
            secret_name = f"projects/{project_id}/secrets/openai-api-key/versions/latest"
            response = client.access_secret_version(request={"name": secret_name})
            return response.payload.data.decode("UTF-8").strip()
    except Exception as e:
        print(f"Warning: Could not load from Secret Manager: {e}")
    
    return None

# Initialize OpenAI API key
openai_api_key = get_openai_api_key()
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY not set. Please set it in environment or Google Secret Manager.")

os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize OpenAI client with proper timeout configuration
_httpx_client = httpx.Client(
    timeout=httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=30.0),
)

openai_client = OpenAI(
    api_key=openai_api_key,
    http_client=_httpx_client
)

# Configuration
OPENAI_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
FAISS_INDEX_PATH = './faiss_vecdb'

# Initialize FastAPI app
app = FastAPI(
    title="LucilleLLM API",
    description="Self-care chatbot API providing wellbeing support and advice",
    version="1.0.0"
)

# CORS middleware - Enhanced for FlutterFlow compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for FlutterFlow
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize embeddings and vector store
embeddings_model = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=openai_api_key,
)

def load_faiss_vectorstore(local_path: str) -> FAISS:
    """Load FAISS index from disk"""
    os.makedirs(local_path, exist_ok=True)
    vs = FAISS.load_local(local_path, embeddings=embeddings_model, allow_dangerous_deserialization=True)
    print(f"✅ Loaded FAISS vector store (dimension: {vs.index.d})")
    return vs

# Load vector store
VectorStore = load_faiss_vectorstore(FAISS_INDEX_PATH)

# Initialize tokenizer for token counting
try:
    tokenizer = tiktoken.encoding_for_model(OPENAI_MODEL)
except KeyError:
    tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count tokens in text"""
    return len(tokenizer.encode(text))

# Session management
session_histories: Dict[str, BaseChatMessageHistory] = defaultdict(ChatMessageHistory)
session_summaries: Dict[str, str] = defaultdict(str)

# Initialize summarization chain
summary_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, request_timeout=120, max_retries=3)
summarize_chain = load_summarize_chain(llm=summary_llm, chain_type="refine")

# Session validation function
def validate_session_id(session_id: str) -> str:
    """Validate and normalize session ID"""
    if not session_id or session_id == "unique_identifier" or session_id == "null" or session_id == "":
        return str(uuid.uuid4())
    return session_id

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    conversation: List[str]
    status: str = "success"
    timestamp: str = ""
    message_count: int = 0

# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with Firebase session management"""
    try:
        # Validate and normalize session ID
        session_id = validate_session_id(request.session_id)
        prompt = request.message
        
        # Initialize Firebase service
        firebase_service = get_firebase_service()

        # Load existing session from Firebase if it exists
        try:
            existing_session = firebase_service.get_chat_session(session_id)
            if existing_session:
                summary = existing_session.get('summary', '')
                session_summaries[session_id] = summary
                print(f"📱 Loaded existing session {session_id} from Firebase")
            else:
                summary = session_summaries.get(session_id, "")
        except Exception as e:
            print(f"⚠️ Failed to load session from Firebase: {e}")
            summary = session_summaries.get(session_id, "")
            existing_session = None

        # Create system prompt with summary if it exists
        system_prompt = (
            "You are Lucille, a self-care expert and helpful assistant. "
            "Always format your responses in **Markdown** using bold, italic, lists, and line breaks for better readability. "
            "You are NOT a medical doctor, so always add a disclaimer where needed and refrain from giving medical advice. "
            "If someone is suicidal, refer them to suicide helplines immediately."
        )
        
        if summary:
            system_prompt = f"{summary}\n\n{system_prompt}"

        # Get chat history for context
        chat_history = session_histories.get(session_id, ChatMessageHistory())
        history_messages = []
        for msg in chat_history.messages[-6:]:  # Last 6 messages for context
            if isinstance(msg, HumanMessage):
                history_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history_messages.append({"role": "assistant", "content": msg.content})

        # Create messages for OpenAI
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        # Call OpenAI API
        try:
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            bot_response = response.choices[0].message.content
        except APIConnectionError as e:
            print(f"❌ OpenAI network error: {e}")
            raise HTTPException(status_code=502, detail=f"OpenAI network error: {e}")
        except RateLimitError as e:
            print(f"❌ OpenAI rate limit: {e}")
            raise HTTPException(status_code=429, detail="OpenAI rate limit hit")
        except Exception as e:
            print(f"❌ OpenAI API call failed: {e}")
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")

        # Update chat history
        chat_history.add_user_message(prompt)
        chat_history.add_ai_message(bot_response)
        session_histories[session_id] = chat_history

        # Reconstruct conversation for response
        conversation_strings = [prompt, bot_response]

        # Summarize if over token limit
        full_text = f"{prompt} {bot_response}"
        if count_tokens(full_text) > 8000:
            print("🔁 Summarizing chat history...")
            try:
                documents = [Document(page_content=full_text)]
                refined_summary = summarize_chain.run(documents)
                session_summaries[session_id] = refined_summary
                
                # Update summary in Firebase
                firebase_service.update_session_summary(session_id, refined_summary)
            except Exception as e:
                print(f"⚠️ Summarization failed: {e}")

        # Store/Update session in Firebase
        try:
            current_messages = chat_history.messages
            current_summary = session_summaries.get(session_id, "")
            
            if existing_session:
                firebase_service.update_chat_session(session_id, current_messages, current_summary)
            else:
                firebase_service.store_chat_session(session_id, current_messages, current_summary)
        except Exception as e:
            print(f"⚠️ Failed to store session in Firebase: {e}")

        return ChatResponse(
            session_id=session_id,
            response=bot_response,
            conversation=conversation_strings,
            status="success",
            timestamp=datetime.now().isoformat(),
            message_count=len(chat_history.messages)
        )

    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint to verify Firebase connectivity"""
    try:
        firebase_service = get_firebase_service()
        # Try a simple operation to verify connectivity
        sessions = firebase_service.list_sessions(limit=1)
        return {
            "status": "healthy",
            "firebase": "connected",
            "vector_store": "loaded",
            "model": OPENAI_MODEL
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "firebase": "disconnected",
            "error": str(e)
        }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint that generates a new session ID"""
    session_id = str(uuid.uuid4())
    return JSONResponse(content={
        "session_id": session_id,
        "status": "success",
        "message": "New session created",
        "timestamp": datetime.now().isoformat()
    })

# Chat interface
@app.get("/chat-interface", response_class=HTMLResponse)
async def chat_interface():
    """Simple HTML chat interface"""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Lucille - Self-Care Chatbot</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .chat-container { border: 1px solid #ddd; height: 400px; overflow-y: auto; padding: 20px; margin: 20px 0; }
            .input-container { display: flex; gap: 10px; }
            input[type="text"] { flex: 1; padding: 10px; font-size: 16px; }
            button { padding: 10px 20px; font-size: 16px; background: #007bff; color: white; border: none; cursor: pointer; }
        </style>
    </head>
    <body>
        <h1>Lucille - Self-Care Chatbot</h1>
        <p>Welcome! I'm here to help with self-care and wellbeing advice.</p>
        <div class="chat-container" id="chatContainer"></div>
        <div class="input-container">
            <input type="text" id="messageInput" placeholder="Type your message here..." />
            <button onclick="sendMessage()">Send</button>
        </div>
        
        <script>
            let sessionId = Math.random().toString(36).substring(7);
            
            async function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                if (!message) return;
                
                const chatContainer = document.getElementById('chatContainer');
                chatContainer.innerHTML += `<div><strong>You:</strong> ${message}</div>`;
                input.value = '';
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: message, session_id: sessionId })
                    });
                    
                    const data = await response.json();
                    chatContainer.innerHTML += `<div><strong>Lucille:</strong> ${data.response}</div>`;
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                } catch (error) {
                    chatContainer.innerHTML += `<div style="color: red;"><strong>Error:</strong> ${error.message}</div>`;
                }
            }
            
            document.getElementById('messageInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') sendMessage();
            });
        </script>
    </body>
    </html>
    """)

# Session management endpoints
@app.get("/chat/{session_id}", response_model=ChatResponse)
async def get_chat_history(session_id: str):
    """Retrieve chat history for a session"""
    try:
        # Validate session ID
        session_id = validate_session_id(session_id)
        firebase_service = get_firebase_service()
        session_data = firebase_service.get_chat_session(session_id)
        
        if not session_data:
            raise HTTPException(status_code=404, detail="No chat history found.")

        messages = session_data.get('messages', [])
        conversation_history = [msg.get('content', '') for msg in messages]

        return ChatResponse(
            session_id=session_id,
            response="Chat history retrieved successfully",
            conversation=conversation_history,
            status="success",
            timestamp=datetime.now().isoformat(),
            message_count=len(conversation_history)
        )
    except Exception as e:
        print(f"❌ Error retrieving chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session"""
    try:
        # Validate session ID
        session_id = validate_session_id(session_id)
        firebase_service = get_firebase_service()
        success = firebase_service.delete_chat_session(session_id)
        
        if success:
            # Clear from memory
            if session_id in session_histories:
                del session_histories[session_id]
            if session_id in session_summaries:
                del session_summaries[session_id]
            
            return JSONResponse(content={
                "session_id": session_id,
                "status": "success",
                "message": f"Session {session_id} deleted successfully",
                "timestamp": datetime.now().isoformat()
            })
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        print(f"❌ Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/")
async def list_sessions(limit: int = 100):
    """List recent chat sessions"""
    try:
        firebase_service = get_firebase_service()
        sessions = firebase_service.list_sessions(limit)
        return JSONResponse(content={
            "sessions": sessions, 
            "count": len(sessions),
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        print(f"❌ Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Session validation endpoint
@app.get("/session/{session_id}/validate")
async def validate_session(session_id: str):
    """Validate if a session exists"""
    try:
        # Validate session ID format
        session_id = validate_session_id(session_id)
        firebase_service = get_firebase_service()
        session_data = firebase_service.get_chat_session(session_id)
        
        return JSONResponse(content={
            "session_id": session_id,
            "valid": session_data is not None,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "session_id": session_id,
                "valid": False,
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

# Exception handlers
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "error_code": 422,
            "message": "Validation error",
            "details": jsonable_encoder(exc.errors()),
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error_code": exc.status_code,
            "message": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error_code": 500,
            "message": "Internal server error",
            "details": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# Development server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)