from fastapi import FastAPI, Request, HTTPException, Cookie, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import uuid
import pickle
import os
import firebase_admin
from collections import defaultdict
from firebase_admin import credentials, storage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import tiktoken
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from firebase_service import get_firebase_service


# ✅ Load environment variables from .env
load_dotenv()

def get_openai_api_key():
    """Get OpenAI API key from environment or Google Secret Manager"""
    # Try local environment first
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    # Try Google Secret Manager in production
    try:
        from google.cloud import secretmanager
        client = secretmanager.SecretManagerServiceClient()
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        if project_id:
            secret_name = f"projects/{project_id}/secrets/openai-api-key/versions/latest"
            response = client.access_secret_version(request={"name": secret_name})
            return response.payload.data.decode("UTF-8")
    except Exception as e:
        print(f"Warning: Could not load from Secret Manager: {e}")
    
    return None

# Set OpenAI API key (env var or Secret Manager). Try to infer project ID if missing.
def _resolve_openai_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    # Try Secret Manager with GOOGLE_CLOUD_PROJECT or inferred project id
    try:
        from google.cloud import secretmanager
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            try:
                import google.auth
                creds, inferred_project = google.auth.default()
                if inferred_project:
                    project_id = inferred_project
            except Exception:
                pass
        if project_id:
            client_sm = secretmanager.SecretManagerServiceClient()
            name = f"projects/{project_id}/secrets/openai-api-key/versions/latest"
            resp = client_sm.access_secret_version(request={"name": name})
            return resp.payload.data.decode("utf-8")
    except Exception as e:
        print(f"Warning: Could not resolve OPENAI_API_KEY via Secret Manager: {e}")
    return ""

openai_api_key = _resolve_openai_key()
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
else:
    print("Warning: No OpenAI API key found!")

openai_model = "gpt-4o-mini"
# ✅ FastAPI setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.swaggerhub.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

"""
Embeddings and Vector DB setup
- Use OpenAI embeddings (text-embedding-3-small, 1536 dims)
- Ensure FAISS index matches embedding dimension; if not, rebuild from texts.pkl
"""

# OpenAI embeddings
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=openai_api_key,
)
print("✅ Using OpenAI embeddings (text-embedding-3-small)")

folder_prefix = 'faiss_vecdb/'
local_download_path = './faiss_vecdb'
os.makedirs(local_download_path, exist_ok=True)

def load_faiss_only(local_path: str) -> FAISS:
    """Load FAISS index from disk without probing or rebuilding.
    Keeps startup fast for Cloud Run. Rebuild locally if needed and include the files in the image.
    """
    vs = FAISS.load_local(local_path, embeddings=embeddings_model, allow_dangerous_deserialization=True)
    print(f"✅ Loaded FAISS (dimension: {vs.index.d})")
    return vs

VectorStore = load_faiss_only(local_download_path)



# Using tokenizer to limit the token size of the prompts
try:
    tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
except KeyError:
    tokenizer = tiktoken.get_encoding("cl100k_base")  # fallback for GPT-4 models

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

session_summaries: Dict[str, str] = defaultdict(str)

summary_llm = ChatOpenAI(model=openai_model, temperature=0, request_timeout=120, max_retries=3)
summarize_chain = load_summarize_chain(
    llm=summary_llm,
    chain_type="refine"
)


retriever = VectorStore.as_retriever(
    # using MMR: Maximal marginal relevance
    search_type="mmr",  
    search_kwargs={
        "k": 3,
        "fetch_k": 10,         # how many initial candidates to consider
        "lambda_mult": 0.7     # higher = more relevance, lower = more diversity
    }
)
retriever_tool = create_retriever_tool(
    retriever,
    "selfcare_search",
    "Search for information about self-care and wellbeing. For any questions about self-care and wellbeing, you MUST use this tool!",
)
tools = [retriever_tool]

llm = ChatOpenAI(model=openai_model, temperature=0, request_timeout=120, max_retries=3)
system_message_prompt = SystemMessagePromptTemplate.from_template(
    # "You are a self-care expert and helpful assistant. Your name is Lucille and you answer people's queries regarding self care and well being. But you are NOT a medical doctor so always add a disclaimer where required and refrain from giving medical advice. If someone is suicidal, refer them to suicide helplines."
    "You are a self-care expert and helpful assistant named Lucille. "
    "Always format your responses in **Markdown** using bold, italic, lists, and line breaks where appropriate for better readability. "
    "You are NOT a medical doctor, so always add a disclaimer where needed and refrain from giving medical advice. "
    "If someone is suicidal, refer them to suicide helplines immediately."

)
human_message_prompt = HumanMessagePromptTemplate.from_template("{input}")
chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    MessagesPlaceholder(variable_name="chat_history"),
    human_message_prompt,
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_tool_calling_agent(llm, tools, chat_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

session_histories: Dict[str, BaseChatMessageHistory] = defaultdict(ChatMessageHistory)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: session_histories[session_id],
    input_messages_key="input",
    history_messages_key="chat_history",
)

class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    conversation: List[str]
    class Config:
        arbitrary_types_allowed = True

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        prompt = request.message
        session_id = request.session_id
        
        # Initialize Firebase service
        firebase_service = get_firebase_service()

        # Step 1: Load existing session from Firebase if it exists
        try:
            existing_session = firebase_service.get_chat_session(session_id)
            if existing_session:
                # Load summary from Firebase
                summary = existing_session.get('summary', '')
                session_summaries[session_id] = summary
                print(f"📱 Loaded existing session {session_id} from Firebase")
            else:
                summary = session_summaries.get(session_id, "")
        except Exception as e:
            print(f"⚠️ Failed to load session from Firebase: {e}")
            summary = session_summaries.get(session_id, "")
            existing_session = None

        # Step 2: Inject summary if it exists
        if summary:
            system_prompt_with_summary = SystemMessage(content=summary + "\n\n" + 
                "You are a self-care expert and helpful assistant named Lucille. "
                "Always format your responses in **Markdown** using bold, italic, lists, and line breaks where appropriate for better readability. "
                "You are NOT a medical doctor, so always add a disclaimer where needed and refrain from giving medical advice. "
                "If someone is suicidal, refer them to suicide helplines immediately."
            )
            chat_prompt.messages[0] = system_prompt_with_summary  # update first prompt dynamically

        # Step 3: Run Lucille
        resp = agent_with_chat_history.invoke(
            {"input": prompt},
            config={"configurable": {"session_id": session_id}},
        )
        bot_response = resp['output']

        # Step 4: Reconstruct full conversation
        conversation_strings = [
            m.content if isinstance(m, (HumanMessage, AIMessage)) else str(m)
            for m in resp['chat_history']
        ]
        full_text = "\n".join(conversation_strings)

        # Step 5: Summarize if over token limit
        if count_tokens(full_text) > 8000:
            print("🔁 Summarizing chat history...")
            documents = [Document(page_content=chunk) for chunk in conversation_strings]
            refined_summary = summarize_chain.run(documents)
            session_summaries[session_id] = refined_summary  # Store the summary
            
            # Update summary in Firebase
            firebase_service.update_session_summary(session_id, refined_summary)

        # Step 6: Store/Update session in Firebase
        try:
            current_messages = session_histories[session_id].messages
            current_summary = session_summaries.get(session_id, "")
            
            if existing_session:
                # Update existing session
                firebase_service.update_chat_session(session_id, current_messages, current_summary)
            else:
                # Create new session
                firebase_service.store_chat_session(session_id, current_messages, current_summary)
        except Exception as e:
            print(f"⚠️ Failed to store session in Firebase: {e}")
            # Continue without Firebase storage

        # Step 7: Return response
        return ChatResponse(
            session_id=session_id,
            response=bot_response,
            conversation=conversation_strings
        )

    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    session_id = str(uuid.uuid4())
    response = JSONResponse(content={"session_id": session_id})
    response.set_cookie(key="session_id", value=session_id)
    return response

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
            "timestamp": "2024-01-01T00:00:00Z"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "firebase": "disconnected",
            "error": str(e),
            "timestamp": "2024-01-01T00:00:00Z"
        }

@app.get("/chats/", response_class=HTMLResponse)
async def chat_page(request: Request):
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html><head><title>Chat with Lucille</title></head>
    <body><h2>Lucille Chat Interface</h2><p>Use the POST `/chat` endpoint to talk to Lucille.</p></body>
    </html>
    """)

@app.get("/chat/{session_id}", response_model=ChatResponse)
async def get_chat_history(session_id: str):
    try:
        # Initialize Firebase service
        firebase_service = get_firebase_service()
        
        # Get session from Firebase
        session_data = firebase_service.get_chat_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="No chat history found.")

        # Extract conversation from Firebase data
        messages = session_data.get('messages', [])
        conversation_history = [msg.get('content', '') for msg in messages]

        return ChatResponse(
            session_id=session_id,
            response="Chat history retrieved successfully from Firebase",
            conversation=conversation_history
        )
    except Exception as e:
        print(f"❌ Error retrieving chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session from Firebase"""
    try:
        firebase_service = get_firebase_service()
        success = firebase_service.delete_chat_session(session_id)
        
        if success:
            # Also clear from memory
            if session_id in session_histories:
                del session_histories[session_id]
            if session_id in session_summaries:
                del session_summaries[session_id]
            
            return {"message": f"Session {session_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found or could not be deleted")
    except Exception as e:
        print(f"❌ Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/")
async def list_sessions(limit: int = 100):
    """List recent chat sessions"""
    try:
        firebase_service = get_firebase_service()
        sessions = firebase_service.list_sessions(limit)
        return {"sessions": sessions, "count": len(sessions)}
    except Exception as e:
        print(f"❌ Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors()}),
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)

@app.post("/chat-simple", response_model=ChatResponse)
async def chat_simple(request: ChatRequest):
    """Simplified chat endpoint that bypasses the agent"""
    try:
        prompt = request.message
        session_id = request.session_id
        
        # Simple direct LLM call
        try:
            response = llm.invoke(f"You are Lucille, a self-care expert. User: {prompt}")
            bot_response = response.content
        except Exception as e:
            print(f"❌ OpenAI API call failed: {e}")
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

        return ChatResponse(
            session_id=session_id,
            response=bot_response,
            conversation=[prompt, bot_response]
        )

    except Exception as e:
        print(f"❌ Error in simple chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

