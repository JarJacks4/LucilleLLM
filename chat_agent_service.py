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
from dotenv import load_dotenv  # ✅ Added

from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.agents import create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import AgentExecutor


# ✅ Load from .env instead of hardcoding
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

app = FastAPI()
client = OpenAI()

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
hf = HuggingFaceEmbeddings(model_name=embedding_model_name)

path = "faiss_vecdb"
VectorStore = FAISS.load_local(path, embeddings=hf, allow_dangerous_deserialization=True)
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


def search(query: str, k: int = 8, thresh: int = 0.8):
    embedded_query = np.asarray(hf.embed_documents([query]))
    scores, inds = VectorStore.index.search(embedded_query, k=k)
    retrieved_examples = []
    for i, s in zip(inds[0], scores[0]):
        if s <= thresh and s >= 0:
            retrieved_examples.append(docs[i].page_content)
    return retrieved_examples


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    prompt = request.message
    session_id = request.session_id
    resp = agent_with_chat_history.invoke(
        {"input": prompt},
        config={"configurable": {"session_id": session_id}},
    )
    bot_response = resp['output']

    return ChatResponse(
        session_id=session_id,
        response=bot_response,
        conversation=resp['chat_history']
    )


@app.get("/")
async def root():
    session_id = str(uuid.uuid4())
    response = JSONResponse(content={"session_id": session_id})
    response.set_cookie(key="session_id", value=session_id)
    return response


if __name__ == "__main__":
    uvicorn.run("chat_service:app", host="127.0.0.1", port=8000, reload=True)
