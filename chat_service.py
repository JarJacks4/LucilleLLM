from fastapi import FastAPI, Request, HTTPException, Cookie
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
import uuid
import uvicorn
from openai import OpenAI
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import pickle
import os
from dotenv import load_dotenv  # ✅

# ✅ Load .env variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_key

app = FastAPI()
client = OpenAI()

# In-memory session storage
sessions: Dict[str, List[str]] = dict()

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
hf = HuggingFaceEmbeddings(model_name=embedding_model_name)

path = "faiss_vecdb"
VectorStore = FAISS.load_local(path, embeddings=hf, allow_dangerous_deserialization=True)
print("FAISS vectorstore loaded successfully")

with open('texts.pkl', 'rb') as file:
    docs = pickle.load(file)


class ChatRequest(BaseModel):
    message: str
    session_id: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    conversation: List[str]


def search(query: str, k: int = 8, thresh: float = 0.8):
    """Embed a new query and return top-matching docs."""
    embedded_query = np.asarray(hf.embed_documents([query]))
    scores, inds = VectorStore.index.search(embedded_query, k=k)
    retrieved_examples = []
    for i, s in zip(inds[0], scores[0]):
        if 0 <= s <= thresh:
            retrieved_examples.append(docs[i].page_content)
    return retrieved_examples


def generate_response(prompt, kb, history, model_name="gpt-4"):
    combined_docs = " ".join(kb)
    messages = [
        {
            "role": "system",
            "content": f"""You are a friendly self care expert named Lucille and not a medical doctor. 
            You have to chat with the user in a friendly way and answer questions related to self-care. 
            If the prompt is related to self-care, use the context provided. 
            If not, or if no context is found, politely refuse. 
            Do not give medical advice.
            \nContext: {combined_docs}
            \nConversation history:\n{"\n".join(history)}"""
        },
        {
            "role": "user",
            "content": f"{prompt}\nConversation history:\n" + "\n".join(history)
        }
    ]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=150,
        temperature=0.7,
        n=1
    )
    answer = response.choices[0].message.content
    history.append(f"User: {prompt}\nBot: {answer}")
    return answer


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    prompt = request.message
    session_id = request.session_id
    history = sessions.get(session_id, [])
    kb = search(prompt, thresh=1.0)
    bot_response = generate_response(prompt, kb, history)
    sessions[session_id] = history  # Save updated history

    return ChatResponse(
        session_id=session_id,
        response=bot_response,
        conversation=history
    )


@app.get("/")
async def root():
    session_id = str(uuid.uuid4())
    sessions[session_id] = []
    response = JSONResponse(content={"session_id": session_id})
    response.set_cookie(key="session_id", value=session_id)
    return response


if __name__ == "__main__":
    uvicorn.run("chat_service:app", host="127.0.0.1", port=8000, reload=True)
