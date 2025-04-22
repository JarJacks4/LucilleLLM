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
from firebase_admin import credentials, storage
from langchain_community.embeddings import HuggingFaceEmbeddings
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
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ✅ Load environment variables from .env
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ✅ Firebase setup from env
firebase_creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
cred = credentials.Certificate(firebase_creds_path)
firebase_admin.initialize_app(cred, {
    'storageBucket': 'escape-ujuzxr.appspot.com'
})
bucket = storage.bucket()

# ✅ FastAPI setup
app = FastAPI()
client = OpenAI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.swaggerhub.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Embeddings and Vector DB
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
hf = HuggingFaceEmbeddings(model_name=embedding_model_name)

folder_prefix = 'faiss_vecdb/'
local_download_path = './faiss_vecdb'
os.makedirs(local_download_path, exist_ok=True)

VectorStore = FAISS.load_local(
    local_download_path, embeddings=hf, allow_dangerous_deserialization=True
)
print("FAISS vectorstore loaded successfully")

with open('texts.pkl', 'rb') as file:
    docs = pickle.load(file)

retriever = VectorStore.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "selfcare_search",
    "Search for information about self-care and wellbeing. For any questions about self-care and wellbeing, you must use this tool!",
)
tools = [retriever_tool]

llm = ChatOpenAI(model="gpt-4", temperature=0)
system_message_prompt = SystemMessagePromptTemplate.from_template(
    "You are a self-care expert and helpful assistant. Your name is Lucille and you answer people's queries regarding self care and well being. But you are NOT a medical doctor so always add a disclaimer where required and refrain from giving medical advice. If someone is suicidal, refer them to suicide helplines."
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
message_history = ChatMessageHistory()

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: message_history,
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

def search(query: str, k: int = 8, thresh: float = 0.8):
    embedded_query = np.asarray(hf.embed_documents([query]))
    scores, inds = VectorStore.index.search(embedded_query, k=k)
    return [docs[i].page_content for i, s in zip(inds[0], scores[0]) if s <= thresh and s >= 0]

session_conversations = {}

def add_message_to_session(session_id, message):
    session_conversations.setdefault(session_id, []).append(message)

def get_conversation_for_session(session_id):
    return session_conversations.get(session_id, [])

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        prompt = request.message
        session_id = request.session_id
        resp = agent_with_chat_history.invoke(
            {"input": prompt},
            config={"configurable": {"session_id": session_id}},
        )
        bot_response = resp['output']

        conversation_strings = [
            m.content if isinstance(m, (HumanMessage, AIMessage)) else str(m)
            for m in resp['chat_history']
        ]

        add_message_to_session(session_id, bot_response)

        return ChatResponse(
            session_id=session_id,
            response=bot_response,
            conversation=conversation_strings
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    session_id = str(uuid.uuid4())
    response = JSONResponse(content={"session_id": session_id})
    response.set_cookie(key="session_id", value=session_id)
    return response

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
        conversation_history = get_conversation_for_session(session_id)
        if not conversation_history:
            raise HTTPException(status_code=404, detail="No chat history found.")
        return ChatResponse(
            session_id=session_id,
            response="Chat history retrieved successfully",
            conversation=conversation_history
        )
    except Exception as e:
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
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
