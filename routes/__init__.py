"""
LucilleLLM - Route Modules

APIRouter-based route organization.
Each module exports a `router` that main.py includes.

Migration note: The /chat endpoints remain in main.py due to their
deep integration with local state (VectorStore, embeddings, etc.).
All other endpoints are organized into route modules here.
"""
