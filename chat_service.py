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


os.environ["OPENAI_API_KEY"] = "sk-proj-t5GY3pUWUxjh4mZLGWUxT3BlbkFJmQ2aJzbx24KyMdvAnXjf"
app = FastAPI()
client = OpenAI()

# In-memory session storage
sessions: Dict[str, List[str]] = dict()

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
hf = HuggingFaceEmbeddings(model_name=embedding_model_name)

path = "faiss_vecdb"
VectorStore = FAISS.load_local(path, embeddings=hf, allow_dangerous_deserialization=True)
print("FAISS vectorstore loaded successfully")
# Load the list from the .pkl file
with open('texts.pkl', 'rb') as file:
    docs = pickle.load(file)


class ChatRequest(BaseModel):
    message: str
    session_id: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    conversation: List[str]


def search(query: str, k: int = 8, thresh: int = 0.8):
    """a function that embeds a new query and returns the most probable results"""
    embedded_query = np.asarray(hf.embed_documents([query])) # embed new query
    scores, inds = VectorStore.index.search(embedded_query,k=k)
    retrieved_examples =[]
    for i,s in zip(inds[0],scores[0]):
      if s<=thresh and s>=0:
        retrieved_examples.append(docs[i].page_content)
    return retrieved_examples


def generate_response(prompt, model_name, kb, history=[]):
    combined_docs = " ".join(kb)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a friendly self care expert named Lucille and not a medical doctor. You have to chat with user in a friendly way and answer questions related to self-care. \
            If prompt is related to self care, use the context given by your assistant to answer. If the prompt is related to self-care and no context found refuse to answer. Do not give medical advice. \n Context: " + combined_docs+"\nConversation history for current session: " + "\n".join(history)},
            {"role": "user",
             "content": prompt+" Conversation history:"+"\n".join(history)}
            ],
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7
    )
    resp = response.choices[0].message.content
    history.append(f"User: {prompt}\nBot: {resp}")
    # resp = "this is models response to prompt"
    return resp


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    prompt = request.message
    session_id = request.session_id
    if not sessions.get(session_id):
        sessions[session_id] = []
        history = []
    else:
        # Append user message to the session conversation
        history = sessions[session_id]

    # generate response
    kb = search(prompt, thresh=1)
    # kb = "this is the relevant docs"
    bot_response = generate_response(prompt, kb, history)

    # Append bot response to the session conversation
    sessions[session_id].append(f"User: {prompt} \n Bot: {bot_response}")

    resp = ChatResponse(
        session_id=session_id,
        response=bot_response,
        conversation=sessions[session_id]
    )
    return resp


@app.get("/")
async def root():
    session_id = str(uuid.uuid4())
    sessions[session_id] = []
    response = JSONResponse(content={"session_id": session_id})
    response.set_cookie(key="session_id", value=session_id)
    return response


if __name__ == "__main__":
    uvicorn.run("chat_service:app", host="127.0.0.1", port=8000, reload=True)
