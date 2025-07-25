from fastapi import FastAPI, Request, HTTPException, Cookie, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from typing import List, Dict
from dotenv import load_dotenv
import numpy as np
import uuid
import pickle
import os
import firebase_admin
from collections import defaultdict
from firebase_admin import credentials, storage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
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
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.runnables import RunnableLambda
import tiktoken

# ✅ FastAPI setup
app = FastAPI()

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

# Tokenizer for summarization threshold
tokenizer = tiktoken.encoding_for_model("gpt-4o")
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

session_summaries: Dict[str, str] = defaultdict(str)

# ✅ Load HuggingFace model without GPU
model_id = "meta-llama/Llama-3.2-1B"  # Replace with small open model

tokenizer_hf = AutoTokenizer.from_pretrained(model_id)
model_hf = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model_hf,
    tokenizer=tokenizer_hf,
    max_new_tokens=512,
    return_full_text=False,
    temperature=0.7
)

llm = HuggingFacePipeline(pipeline=pipe)
summary_llm = llm

summarize_chain = load_summarize_chain(
    llm=summary_llm,
    chain_type="refine"
)

system_message_prompt = SystemMessagePromptTemplate.from_template(
    "You are a self-care expert and helpful assistant named Lucille. "
    "Always format your responses in **Markdown** using bold, italic, lists, and line breaks where appropriate for better readability. "
    "You are NOT a medical doctor, so always add a disclaimer where needed and refrain from giving medical advice. "
    "If someone is suicidal, refer them to suicide helplines immediately."
)
human_message_prompt = HumanMessagePromptTemplate.from_template("{input}")
chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    MessagesPlaceholder(variable_name="chat_history"),
    human_message_prompt
])

# Wrap prompt + LLM chain
chain = chat_prompt | llm

session_histories: Dict[str, BaseChatMessageHistory] = defaultdict(ChatMessageHistory)

chat_executor = RunnableWithMessageHistory(
    chain,
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

        # Inject summary if it exists
        summary = session_summaries.get(session_id, "")
        if summary:
            system_prompt_with_summary = SystemMessage(content=summary + "\n\n" + system_message_prompt.prompt.template)
            chat_prompt.messages[0] = system_prompt_with_summary

        resp = chat_executor.invoke(
            {"input": prompt},
            config={"configurable": {"session_id": session_id}},
        )
        bot_response = resp['output'] if isinstance(resp, dict) else resp

        conversation_strings = [
            m.content if isinstance(m, (HumanMessage, AIMessage)) else str(m)
            for m in session_histories[session_id].messages
        ]
        full_text = "\n".join(conversation_strings)

        if count_tokens(full_text) > 8000:
            print("🔁 Summarizing chat history...")
            documents = [Document(page_content=chunk) for chunk in conversation_strings]
            refined_summary = summarize_chain.run(documents)
            session_summaries[session_id] = refined_summary

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
        history = session_histories.get(session_id)
        if not history:
            raise HTTPException(status_code=404, detail="No chat history found.")

        conversation_history = [msg.content for msg in history.messages]

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
    import uvicorn
    uvicorn.run("llama_testing:app", host="0.0.0.0", port=8080, reload=True)
