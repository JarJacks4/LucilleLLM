"""
LucilleLLM - Self-Care Chatbot API

A FastAPI-based chatbot that provides self-care advice and wellbeing support.
Features OpenAI integration, FAISS vector search, and Firebase session management.

RAG Enhancement (suyash-rag-enhancement branch):
- Added RAG (Retrieval-Augmented Generation) to /chat endpoint
- Retrieves relevant context from FAISS vector store based on user queries
- Augments LLM prompts with retrieved self-care knowledge base content
- Improves response accuracy and relevance for self-care topics

Streaming Enhancement (suyash-streaming-chat branch):
- Added /chat/stream endpoint for real-time token streaming
- Uses Server-Sent Events (SSE) for true streaming responses
- Provides ChatGPT-like typing experience with RAG integration
"""

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Optional, AsyncGenerator
from dotenv import load_dotenv
import os
import uuid
import json
import asyncio
from collections import defaultdict
from datetime import datetime
import logging
import sys
import numpy as np
import pickle

# OpenAI and LangChain imports
from openai import OpenAI, APIConnectionError, RateLimitError
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
# from langchain_community.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
import tiktoken
import httpx

# Local imports
from firebase_service import get_firebase_service

# Load environment variables
load_dotenv()

# Configure logging for production (Cloud Run compatible)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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

# Load document texts for RAG
try:
    with open('texts.pkl', 'rb') as file:
        DOCS = pickle.load(file)
    logger.info(f"✅ Loaded {len(DOCS)} documents for RAG")
except FileNotFoundError:
    logger.warning("⚠️ texts.pkl not found. RAG retrieval will be disabled.")
    DOCS = []
except Exception as e:
    logger.error(f"❌ Failed to load texts.pkl: {e}")
    DOCS = []

# Initialize tokenizer for token counting
try:
    tokenizer = tiktoken.encoding_for_model(OPENAI_MODEL)
except KeyError:
    tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count tokens in text"""
    return len(tokenizer.encode(text))

def retrieve_relevant_context(query: str, k: int = 5, similarity_threshold: float = 1.1) -> str:
    """
    Retrieve relevant context from FAISS vector store for RAG.
    
    Args:
        query: User's query
        k: Number of top results to retrieve
        similarity_threshold: Maximum distance threshold (lower is more similar)
    
    Returns:
        Combined context string from retrieved documents
    """
    if not DOCS or len(DOCS) == 0:
        logger.warning("No documents available for RAG retrieval")
        return ""
    
    try:
        # Embed the query using OpenAI embeddings
        query_embedding = embeddings_model.embed_query(query)
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Search in FAISS index
        distances, indices = VectorStore.index.search(query_vector, k)
        
        # Filter by similarity threshold and collect relevant docs
        retrieved_contexts = []
        for idx, distance in zip(indices[0], distances[0]):
            if distance <= similarity_threshold and 0 <= idx < len(DOCS):
                retrieved_contexts.append(DOCS[idx].page_content)
        
        if retrieved_contexts:
            logger.info(f"🔍 Retrieved {len(retrieved_contexts)} relevant documents for query")
            return "\n\n".join(retrieved_contexts)
        else:
            logger.info(f"No documents found within similarity threshold ({similarity_threshold})")
            return ""
            
    except Exception as e:
        logger.error(f"❌ Error retrieving context: {e}")
        return ""

# Session management
session_histories: Dict[str, BaseChatMessageHistory] = defaultdict(ChatMessageHistory)
session_summaries: Dict[str, str] = defaultdict(str)

# Initialize summarization LLM
summary_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, request_timeout=120, max_retries=3)

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

        # 🔍 RAG: Retrieve relevant context from vector store
        retrieved_context = retrieve_relevant_context(prompt, k=5, similarity_threshold=1.1)
        
        # Create system prompt with RAG context, summary, and base instructions
        system_prompt = (
            "You are Lucille, a self-care expert and helpful assistant. "
            "Always format your responses in **Markdown** using bold, italic, lists, and line breaks for better readability. "
            "You are NOT a medical doctor, so always add a disclaimer where needed and refrain from giving medical advice. "
            "If someone is suicidal, refer them to suicide helplines immediately."
        )
        
        # Add retrieved context if available
        if retrieved_context:
            system_prompt = (
                f"You are Lucille, a self-care expert and helpful assistant. "
                f"Use the following context from the self-care knowledge base to inform your response:\n\n"
                f"--- KNOWLEDGE BASE CONTEXT ---\n{retrieved_context}\n--- END CONTEXT ---\n\n"
                f"Always format your responses in **Markdown** using bold, italic, lists, and line breaks for better readability. "
                f"You are NOT a medical doctor, so always add a disclaimer where needed and refrain from giving medical advice. "
                f"If someone is suicidal, refer them to suicide helplines immediately. "
                f"If the context is relevant to the user's question, incorporate it naturally into your response. "
                f"If the context is not relevant, rely on your general knowledge while staying true to your role."
            )
        
        # Add conversation summary if it exists
        if summary:
            system_prompt = f"Previous conversation summary:\n{summary}\n\n{system_prompt}"

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
                # Create a simple summary using the LLM directly
                summary_prompt = f"Please summarize the following conversation history in a concise way, focusing on key topics and decisions:\n\n{full_text}"
                refined_summary = summary_llm.invoke(summary_prompt).content
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

        payload = ChatResponse(
            session_id=session_id,
            response=bot_response,
            conversation=conversation_strings,
            status="success",
            timestamp=datetime.now().isoformat(),
            message_count=len(chat_history.messages)
        )
        return JSONResponse(content=payload.model_dump())

    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Streaming chat endpoint
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint with RAG enhancement.
    Returns Server-Sent Events (SSE) for real-time token streaming.
    
    Response format:
    - data: {"delta": "token", "session_id": "...", "type": "content"}
    - data: {"type": "done", "session_id": "...", "message_count": N}
    - data: [DONE]
    """
    try:
        # Validate and normalize session ID
        session_id = validate_session_id(request.session_id)
        prompt = request.message
        
        logger.info(f"🔄 Streaming chat request for session {session_id}")
        
        # Initialize Firebase service
        firebase_service = get_firebase_service()

        # Load existing session from Firebase if it exists
        try:
            existing_session = firebase_service.get_chat_session(session_id)
            if existing_session:
                summary = existing_session.get('summary', '')
                session_summaries[session_id] = summary
                logger.info(f"📱 Loaded existing session {session_id} from Firebase")
            else:
                summary = session_summaries.get(session_id, "")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load session from Firebase: {e}")
            summary = session_summaries.get(session_id, "")
            existing_session = None

        # 🔍 RAG: Retrieve relevant context from vector store
        retrieved_context = retrieve_relevant_context(prompt, k=5, similarity_threshold=1.1)
        
        # Create system prompt with RAG context, summary, and base instructions
        system_prompt = (
            "You are Lucille, a self-care expert and helpful assistant. "
            "Always format your responses in **Markdown** using bold, italic, lists, and line breaks for better readability. "
            "You are NOT a medical doctor, so always add a disclaimer where needed and refrain from giving medical advice. "
            "If someone is suicidal, refer them to suicide helplines immediately."
        )
        
        # Add retrieved context if available
        if retrieved_context:
            system_prompt = (
                f"You are Lucille, a self-care expert and helpful assistant. "
                f"Use the following context from the self-care knowledge base to inform your response:\n\n"
                f"--- KNOWLEDGE BASE CONTEXT ---\n{retrieved_context}\n--- END CONTEXT ---\n\n"
                f"Always format your responses in **Markdown** using bold, italic, lists, and line breaks for better readability. "
                f"You are NOT a medical doctor, so always add a disclaimer where needed and refrain from giving medical advice. "
                f"If someone is suicidal, refer them to suicide helplines immediately. "
                f"If the context is relevant to the user's question, incorporate it naturally into your response. "
                f"If the context is not relevant, rely on your general knowledge while staying true to your role."
            )
        
        # Add conversation summary if it exists
        if summary:
            system_prompt = f"Previous conversation summary:\n{summary}\n\n{system_prompt}"

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

        async def event_generator() -> AsyncGenerator[str, None]:
            """Generate SSE events with streaming tokens from OpenAI"""
            full_response = ""
            
            try:
                # Call OpenAI API with streaming enabled
                stream = openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.7,
                    stream=True  # Enable streaming
                )
                
                # Stream tokens as they arrive
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        full_response += token
                        
                        # Send token as SSE event
                        event_data = json.dumps({
                            "delta": token,
                            "session_id": session_id,
                            "type": "content"
                        })
                        yield f"data: {event_data}\n\n"
                
                # Update chat history after streaming completes
                chat_history.add_user_message(prompt)
                chat_history.add_ai_message(full_response)
                session_histories[session_id] = chat_history
                
                # Store/Update session in Firebase (async-safe)
                try:
                    current_messages = chat_history.messages
                    current_summary = session_summaries.get(session_id, "")
                    
                    if existing_session:
                        firebase_service.update_chat_session(session_id, current_messages, current_summary)
                    else:
                        firebase_service.store_chat_session(session_id, current_messages, current_summary)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to store session in Firebase: {e}")
                
                # Summarize if over token limit
                full_text = f"{prompt} {full_response}"
                if count_tokens(full_text) > 8000:
                    logger.info("🔁 Summarizing chat history...")
                    try:
                        summary_prompt = f"Please summarize the following conversation history in a concise way, focusing on key topics and decisions:\n\n{full_text}"
                        refined_summary = summary_llm.invoke(summary_prompt).content
                        session_summaries[session_id] = refined_summary
                        firebase_service.update_session_summary(session_id, refined_summary)
                    except Exception as e:
                        logger.warning(f"⚠️ Summarization failed: {e}")
                
                # Send completion event with full response (matching /chat format)
                done_data = json.dumps({
                    "type": "done",
                    "session_id": session_id,
                    "response": full_response,
                    "conversation": [prompt, full_response],
                    "status": "success",
                    "message_count": len(chat_history.messages),
                    "timestamp": datetime.now().isoformat()
                })
                yield f"data: {done_data}\n\n"
                yield "data: [DONE]\n\n"
                
                logger.info(f"✅ Streaming completed for session {session_id}")
                
            except APIConnectionError as e:
                logger.error(f"❌ OpenAI network error: {e}")
                error_data = json.dumps({"type": "error", "message": f"OpenAI network error: {e}"})
                yield f"data: {error_data}\n\n"
                yield "data: [DONE]\n\n"
            except RateLimitError as e:
                logger.error(f"❌ OpenAI rate limit: {e}")
                error_data = json.dumps({"type": "error", "message": "OpenAI rate limit hit"})
                yield f"data: {error_data}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"❌ Streaming error: {e}")
                error_data = json.dumps({"type": "error", "message": str(e)})
                yield f"data: {error_data}\n\n"
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error in streaming chat endpoint: {e}")
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