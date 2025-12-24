# RAG Enhancement for LucilleLLM

## Overview
This branch (`suyash-rag-enhancement`) adds **Retrieval-Augmented Generation (RAG)** functionality to the `/chat` endpoint in `main.py`.

## What Changed?

### Before
- The `/chat` endpoint was calling OpenAI directly with only the conversation history
- The FAISS vector store was loaded but **never used**
- Responses relied solely on GPT-4's general knowledge

### After
- The `/chat` endpoint now retrieves relevant context from the FAISS vector store for each user query
- Retrieved self-care knowledge is injected into the system prompt
- The LLM now has access to your custom self-care knowledge base when generating responses

## Technical Implementation

### 1. Added Dependencies
```python
import numpy as np
import pickle
```

### 2. Document Loading
```python
# Load document texts for RAG
with open('texts.pkl', 'rb') as file:
    DOCS = pickle.load(file)
```

### 3. RAG Retrieval Function
```python
def retrieve_relevant_context(query: str, k: int = 5, similarity_threshold: float = 0.85) -> str
```
- Embeds the user's query using OpenAI embeddings
- Searches FAISS index for top-k most similar documents
- Filters by similarity threshold
- Returns combined context string

### 4. Integration in `/chat` Endpoint
```python
# Retrieve relevant context
retrieved_context = retrieve_relevant_context(prompt, k=5, similarity_threshold=0.85)

# Augment system prompt with retrieved context
if retrieved_context:
    system_prompt = f"Use the following context from the self-care knowledge base...\n{retrieved_context}"
```

## Configuration Parameters

### `k` (default: 5)
Number of top documents to retrieve from the vector store.
- Higher = More context, potentially more relevant information
- Lower = Faster, less noise

### `similarity_threshold` (default: 0.85)
Maximum distance threshold for considering a document relevant.
- Lower values = More strict (only very similar documents)
- Higher values = More lenient (includes less similar documents)
- Range: 0.0 (perfect match) to infinity

## Benefits

1. **Improved Accuracy**: Responses are grounded in your specific self-care knowledge base
2. **Consistency**: All responses leverage the same curated information
3. **Transparency**: Retrieved context can be logged for debugging
4. **Flexibility**: Easy to tune retrieval parameters for optimal performance

## Testing

To test the RAG functionality:

1. Ask a question related to content in your self-care knowledge base
2. Check logs for: `🔍 Retrieved X relevant documents for query`
3. Compare response quality with and without RAG

## Deployment Notes

**Important**: Make sure `texts.pkl` is included in your deployment:
- Add to Docker image
- Upload to Cloud Storage if needed
- The app will gracefully handle missing `texts.pkl` (RAG disabled with warning)

## Next Steps

Potential enhancements:
- [ ] Add metadata filtering (by category, source, etc.)
- [ ] Implement hybrid search (semantic + keyword)
- [ ] Add citation support (show which docs were used)
- [ ] Tune similarity threshold based on query type
- [ ] Cache embeddings for frequently asked questions

