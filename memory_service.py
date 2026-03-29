"""
LucilleLLM - Memory Service

Manages episodic, semantic, and factual memories per user.
Features: CRUD, auto-extraction from conversations, semantic search,
and memory formatting for system prompt injection.

Follows the singleton pattern from firebase_service.py / user_service.py.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from firebase_admin import firestore
from openai import OpenAI

from firebase_service import get_firebase_service
from models import Memory, MemoryType

logger = logging.getLogger(__name__)

# ── Extraction Prompt ─────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = (
    "You extract memorable facts from a therapy chatbot conversation. "
    "Given the user's message and the assistant's response, identify any NEW "
    "information worth remembering about the user.\n\n"
    "Return a JSON array of memory objects. Each object has:\n"
    '- "content": a concise statement about the user (e.g. "User has a dog named Max")\n'
    '- "memory_type": one of ["episodic", "semantic", "fact"]\n'
    '  - episodic = specific events/experiences (e.g. "User went hiking last weekend")\n'
    '  - semantic = general knowledge/preferences (e.g. "User prefers mornings")\n'
    '  - fact = concrete facts (e.g. "User lives in NYC")\n'
    '- "importance": integer 1-10 (1=trivial small talk, 10=critical life event)\n'
    '- "tags": list of 1-3 short tags (e.g. ["family", "health"])\n\n'
    "Rules:\n"
    "- Only extract information ABOUT THE USER, not general advice given by the assistant.\n"
    "- Skip greetings, filler, and anything too vague.\n"
    "- If there is nothing worth remembering, return an empty array [].\n"
    "- Return ONLY valid JSON, no explanation.\n"
)

VALID_MEMORY_TYPES = {"episodic", "semantic", "fact"}


class MemoryService:
    """
    Service for user memory CRUD, extraction, and search.

    Patterns (matching existing services):
    - Singleton via module-level global + getter
    - self.db is None guard on every method
    - try/except with graceful degradation
    """

    COLLECTION = "user_memories"
    MAX_MEMORIES_PER_USER = 200

    def __init__(self, openai_client: OpenAI, embedding_model: str = "text-embedding-3-small"):
        self._client = openai_client
        self._embedding_model = embedding_model
        self._firebase = get_firebase_service()

    @property
    def db(self):
        return self._firebase.db

    # ── CREATE ───────────────────────────────────────

    def store_memory(self, user_id: str, memory_data: dict) -> Optional[str]:
        """
        Store a single memory document in Firestore.

        Args:
            user_id: The user this memory belongs to.
            memory_data: dict with content, memory_type, importance, tags, etc.

        Returns:
            memory_id on success, None on failure.
        """
        if self.db is None:
            return None

        try:
            memory_id = memory_data.get("memory_id") or str(uuid.uuid4())
            memory_data["memory_id"] = memory_id
            memory_data["user_id"] = user_id
            memory_data.setdefault("created_at", datetime.now().isoformat())
            memory_data.setdefault("last_accessed", datetime.now().isoformat())
            memory_data.setdefault("access_count", 0)
            memory_data.setdefault("source", "auto_extracted")

            # Pre-compute embedding for semantic search
            embedding = self._embed_text(memory_data["content"])
            if embedding is not None:
                memory_data["embedding"] = embedding

            doc_ref = (
                self.db.collection(self.COLLECTION)
                .document(user_id)
                .collection("memories")
                .document(memory_id)
            )
            doc_ref.set(memory_data)

            logger.info(f"Stored memory {memory_id} for user {user_id}")
            return memory_id

        except Exception as e:
            logger.warning(f"Failed to store memory for {user_id}: {e}")
            return None

    # ── READ ─────────────────────────────────────────

    def get_memories(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[dict]:
        """Retrieve memories for a user, optionally filtered by type."""
        if self.db is None:
            return []

        try:
            col_ref = (
                self.db.collection(self.COLLECTION)
                .document(user_id)
                .collection("memories")
            )

            if memory_type and memory_type in VALID_MEMORY_TYPES:
                query = col_ref.where("memory_type", "==", memory_type)
            else:
                query = col_ref

            query = query.order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit)

            memories = []
            for doc in query.stream():
                data = doc.to_dict()
                # Remove embedding from response (too large)
                data.pop("embedding", None)
                memories.append(data)

            return memories

        except Exception as e:
            logger.warning(f"Failed to get memories for {user_id}: {e}")
            return []

    # ── DELETE ────────────────────────────────────────

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """Delete a specific memory."""
        if self.db is None:
            return False

        try:
            doc_ref = (
                self.db.collection(self.COLLECTION)
                .document(user_id)
                .collection("memories")
                .document(memory_id)
            )
            doc = doc_ref.get()
            if not doc.exists:
                return False

            doc_ref.delete()
            logger.info(f"Deleted memory {memory_id} for user {user_id}")
            return True

        except Exception as e:
            logger.warning(f"Failed to delete memory {memory_id}: {e}")
            return False

    # ── SEMANTIC SEARCH ──────────────────────────────

    def search_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> List[dict]:
        """
        Search user memories by semantic similarity.

        Embeds the query, then computes cosine similarity against
        pre-computed memory embeddings stored in Firestore.
        """
        if self.db is None:
            return []

        try:
            query_embedding = self._embed_text(query)
            if query_embedding is None:
                return self.get_memories(user_id, limit=limit)

            # Fetch all memories with embeddings
            col_ref = (
                self.db.collection(self.COLLECTION)
                .document(user_id)
                .collection("memories")
            )
            all_docs = list(col_ref.stream())

            scored: List[Tuple[float, dict]] = []
            query_vec = np.array(query_embedding, dtype=np.float32)

            for doc in all_docs:
                data = doc.to_dict()
                emb = data.get("embedding")
                if emb is None:
                    continue

                mem_vec = np.array(emb, dtype=np.float32)
                # Cosine similarity
                dot = np.dot(query_vec, mem_vec)
                norm = np.linalg.norm(query_vec) * np.linalg.norm(mem_vec)
                similarity = float(dot / norm) if norm > 0 else 0.0

                data.pop("embedding", None)
                data["relevance"] = round(similarity, 4)
                scored.append((similarity, data))

            # Sort by similarity descending
            scored.sort(key=lambda x: x[0], reverse=True)

            results = [item[1] for item in scored[:limit]]

            # Update access metadata for returned memories
            self._touch_memories(user_id, [m["memory_id"] for m in results])

            return results

        except Exception as e:
            logger.warning(f"Memory search failed for {user_id}: {e}")
            return []

    # ── AUTO-EXTRACTION ──────────────────────────────

    def extract_memories(
        self,
        user_id: str,
        user_message: str,
        assistant_response: str,
    ) -> List[str]:
        """
        Use LLM to extract memorable facts from a conversation turn.
        Stores extracted memories and returns list of memory_ids.
        """
        try:
            conversation_text = (
                f"User: {user_message}\n"
                f"Assistant: {assistant_response}"
            )

            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": conversation_text},
                ],
                max_tokens=300,
                temperature=0,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content
            data = json.loads(raw)

            # Handle both {"memories": [...]} and plain [...]
            if isinstance(data, dict):
                memories_list = data.get("memories", data.get("items", []))
                if not isinstance(memories_list, list):
                    memories_list = []
            elif isinstance(data, list):
                memories_list = data
            else:
                memories_list = []

            stored_ids = []
            for mem in memories_list:
                if not isinstance(mem, dict) or not mem.get("content"):
                    continue

                # Validate memory_type
                mem_type = mem.get("memory_type", "semantic")
                if mem_type not in VALID_MEMORY_TYPES:
                    mem_type = "semantic"

                # Validate importance
                importance = mem.get("importance", 5)
                try:
                    importance = max(1, min(10, int(importance)))
                except (TypeError, ValueError):
                    importance = 5

                tags = mem.get("tags", [])
                if not isinstance(tags, list):
                    tags = []

                # Check for duplicates before storing
                if self._is_duplicate(user_id, mem["content"]):
                    continue

                memory_data = {
                    "content": mem["content"],
                    "memory_type": mem_type,
                    "importance": importance,
                    "tags": tags,
                    "source": "auto_extracted",
                }

                mid = self.store_memory(user_id, memory_data)
                if mid:
                    stored_ids.append(mid)

            if stored_ids:
                logger.info(
                    f"Extracted {len(stored_ids)} memories for user {user_id}"
                )

            return stored_ids

        except Exception as e:
            logger.warning(f"Memory extraction failed for {user_id}: {e}")
            return []

    # ── PROMPT FORMATTING ────────────────────────────

    def format_memories_for_prompt(self, memories: List[dict]) -> str:
        """
        Format retrieved memories as a text block for system prompt injection.
        Only includes non-empty memories, sorted by relevance/importance.
        """
        if not memories:
            return ""

        parts = []
        for mem in memories:
            content = mem.get("content", "")
            mem_type = mem.get("memory_type", "semantic")
            importance = mem.get("importance", 5)

            if not content:
                continue

            label = {
                "episodic": "Event",
                "semantic": "Known",
                "fact": "Fact",
            }.get(mem_type, "Note")

            if importance >= 8:
                parts.append(f"[{label} - Important] {content}")
            else:
                parts.append(f"[{label}] {content}")

        if not parts:
            return ""

        return (
            "--- USER MEMORIES ---\n"
            + "\n".join(parts)
            + "\n--- END MEMORIES ---"
        )

    # ── CONSOLIDATION ────────────────────────────────

    def consolidate_memories(self, user_id: str) -> int:
        """
        Consolidate old, low-importance memories that haven't been accessed.
        Removes memories with importance <= 2 and access_count == 0
        that are older than 30 days.
        Returns the number of memories removed.
        """
        if self.db is None:
            return 0

        try:
            col_ref = (
                self.db.collection(self.COLLECTION)
                .document(user_id)
                .collection("memories")
            )

            cutoff = datetime.now().isoformat()
            removed = 0

            for doc in col_ref.stream():
                data = doc.to_dict()
                importance = data.get("importance", 5)
                access_count = data.get("access_count", 0)
                created_at = data.get("created_at", "")

                if importance <= 2 and access_count == 0 and created_at:
                    try:
                        created = datetime.fromisoformat(created_at)
                        age_days = (datetime.now() - created).days
                        if age_days > 30:
                            doc.reference.delete()
                            removed += 1
                    except (ValueError, TypeError):
                        continue

            if removed:
                logger.info(
                    f"Consolidated {removed} old memories for user {user_id}"
                )
            return removed

        except Exception as e:
            logger.warning(f"Memory consolidation failed for {user_id}: {e}")
            return 0

    # ── PRIVATE HELPERS ──────────────────────────────

    def _embed_text(self, text: str) -> Optional[List[float]]:
        """Compute embedding for a text string using OpenAI."""
        try:
            response = self._client.embeddings.create(
                model=self._embedding_model,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None

    def _touch_memories(self, user_id: str, memory_ids: List[str]) -> None:
        """Update last_accessed and access_count for retrieved memories."""
        if self.db is None or not memory_ids:
            return

        try:
            now = datetime.now().isoformat()
            for mid in memory_ids:
                doc_ref = (
                    self.db.collection(self.COLLECTION)
                    .document(user_id)
                    .collection("memories")
                    .document(mid)
                )
                doc_ref.update({
                    "last_accessed": now,
                    "access_count": firestore.Increment(1),
                })
        except Exception:
            pass  # Non-critical

    def _is_duplicate(self, user_id: str, content: str) -> bool:
        """
        Quick check: does a memory with very similar content already exist?
        Uses embedding similarity to detect near-duplicates.
        """
        try:
            new_emb = self._embed_text(content)
            if new_emb is None:
                return False

            col_ref = (
                self.db.collection(self.COLLECTION)
                .document(user_id)
                .collection("memories")
            )

            new_vec = np.array(new_emb, dtype=np.float32)

            for doc in col_ref.stream():
                data = doc.to_dict()
                emb = data.get("embedding")
                if emb is None:
                    continue

                existing_vec = np.array(emb, dtype=np.float32)
                dot = np.dot(new_vec, existing_vec)
                norm = np.linalg.norm(new_vec) * np.linalg.norm(existing_vec)
                similarity = float(dot / norm) if norm > 0 else 0.0

                if similarity > 0.92:
                    logger.info(
                        f"Skipping duplicate memory (sim={similarity:.3f}): "
                        f"{content[:60]}"
                    )
                    return True

            return False

        except Exception:
            return False


# ── Singleton ─────────────────────────────────────────

_memory_service: Optional[MemoryService] = None


def get_memory_service(
    openai_client: OpenAI = None,
    embedding_model: str = "text-embedding-3-small",
) -> MemoryService:
    """Get or create MemoryService singleton instance.
    openai_client must be provided on first call."""
    global _memory_service
    if _memory_service is None:
        if openai_client is None:
            raise ValueError(
                "openai_client must be provided on first call to get_memory_service()"
            )
        _memory_service = MemoryService(openai_client, embedding_model)
    return _memory_service
