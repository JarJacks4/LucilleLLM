from fastapi import FastAPI, Request, HTTPException, Cookie
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
import uuid
import uvicorn
from openai import OpenAI
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
import os
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

from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi import status
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError
# from langchain.schema import HumanMessage, AIMessage
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
# # Add this middleware to your FastAPI app


os.environ["OPENAI_API_KEY"] = "sk-proj-t5GY3pUWUxjh4mZLGWUxT3BlbkFJmQ2aJzbx24KyMdvAnXjf"
app = FastAPI()
client = OpenAI()

app.add_middleware(
    CORSMiddleware,
    # Replace "*" with specific origins like ["https://example.com"] for more security
    allow_origins=["https://app.swaggerhub.com"],
    allow_credentials=True,
    allow_methods=["*"],  # Or specify allowed methods, e.g., ["GET", "POST"]
    # Or specify allowed headers, e.g., ["Content-Type", "Authorization"]
    allow_headers=["*"],
)


# @app.options("/{full_path:path}")
# async def preflight_handler(full_path: str):
#     response = JSONResponse(content={"message": "Preflight OK"})
#     response.headers["Access-Control-Allow-Origin"] = "*"
#     response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
#     response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
#     response.headers["Access-Control-Allow-Credentials"] = "true"
#     return response


embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
hf = HuggingFaceEmbeddings(model_name=embedding_model_name)

path = "C:/Users/siddh/Downloads/LucilleLLM-main/faiss_vecdb"
VectorStore = FAISS.load_local(
    path, embeddings=hf, allow_dangerous_deserialization=True)
print("FAISS vectorstore loaded successfully")
# Load the list from the .pkl file
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
    "You are a self-care expert and helpful assistant. Your name is Lucille and you answer people's queries regarding self care and well being. But you are NOT a medical doctor so always add a disclaimer with where required andrefrain from giving medical advise. If someone is suicidal please refer them tto suicide helplines.")
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


def search(query: str, k: int = 8, thresh: int = 0.8):
    """a function that embeds a new query and returns the most probable results"""
    embedded_query = np.asarray(hf.embed_documents([query]))  # embed new query
    scores, inds = VectorStore.index.search(embedded_query, k=k)
    retrieved_examples = []
    for i, s in zip(inds[0], scores[0]):
        if s <= thresh and s >= 0:
            retrieved_examples.append(docs[i].page_content)
    return retrieved_examples


session_conversations = {}

# Function to add a message to a session's conversation history


def add_message_to_session(session_id, message):
    if session_id not in session_conversations:
        session_conversations[session_id] = []
    session_conversations[session_id].append(message)

# Function to retrieve a session's conversation history


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

        # Convert chat history messages to strings
        conversation_strings = []
        for message in resp['chat_history']:
            if isinstance(message, (HumanMessage, AIMessage)):
                conversation_strings.append(message.content)
            else:
                conversation_strings.append(str(message))

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
    html_content = """
   <!DOCTYPE html>
    <html>
    <head>
        <title>Chat with Lucille</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            #chat-container {
                max-width: 800px;
                margin: 0 auto;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                padding: 20px;
            }
            #chat-messages {
                height: 500px;
                overflow-y: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 20px;
            }
            #message-input {
                width: calc(100% - 120px);
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-right: 10px;
            }
            #send-button {
                padding: 10px 20px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            #send-button:hover {
                background-color: #0056b3;
            }
            .message {
                margin-bottom: 10px;
                padding: 10px;
                border-radius: 5px;
            }
            .user-message {
                background-color: #e3f2fd;
            }
            .bot-message {
                background-color: #f5f5f5;
            }
        </style>
    </head>
    <body>
        <div id="chat-container">
            <div id="chat-messages"></div>
            <div style="display: flex;">
                <input type="text" id="message-input" placeholder="Type your message...">
                <button id="send-button">Send</button>
            </div>
        </div>

        <script>
            let sessionId = null;

            async function getSessionId() {
                const response = await fetch('/');
                const data = await response.json();
                sessionId = data.session_id;
            }

            async function sendMessage() {
                const messageInput = document.getElementById('message-input');
                const message = messageInput.value.trim()
                if (!message) return;

                const chatMessages = document.getElementById('chat-messages');
                chatMessages.innerHTML += `<div class="message user-message"><strong>You:</strong> ${message}</div>`;
                messageInput.value = '';

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            message: message,
                            session_id: sessionId
                        })
                    });

                    const data = await response.json();
                    chatMessages.innerHTML += `<div class="message bot-message"><strong>Lucille:</strong> ${data.response}</div>`;
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                } catch (error) {
                    console.error('Error:', error);
                    chatMessages.innerHTML += `<div class="message bot-message" style="color: red;">Error: Could not send message</div>`;
                }
            }

            document.getElementById('send-button').addEventListener('click', sendMessage);
            document.getElementById('message-input').addEventListener('keypress', (e) => {
                if (e.key === 'Enter') sendMessage();
            });

            getSessionId();
        </script>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)


@app.get("/chat/{session_id}", response_model=ChatResponse)
async def get_chat_history(session_id: str):
    try:
        # Get the conversation history for the provided session_id
        conversation_history = get_conversation_for_session(session_id)

        # If no conversation history exists, raise a 404 error
        if not conversation_history:
            raise HTTPException(
                status_code=404, detail="No chat history found for this session.")

        # Join the conversation history into a single string (or use a list of strings)
        conversation_strings = []
        for message in conversation_history:
            conversation_strings.append(message)

        # Assuming you are getting the bot response from the agent with chat history
        bot_response = "Chat history retrieved successfully"

        return ChatResponse(
            session_id=session_id,
            response=bot_response,
            conversation=conversation_strings
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}")


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
    uvicorn.run("main:app",
                host="0.0.0.0", port=8080, reload=True)


