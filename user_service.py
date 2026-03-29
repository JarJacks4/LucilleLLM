"""
LucilleLLM - User Profile Service

Firebase CRUD operations for user profiles with 5 behavioral layers.
Follows the same singleton pattern as firebase_service.py.
"""

from firebase_admin import firestore
from typing import Optional, List
from datetime import datetime
import logging
import uuid

from firebase_service import get_firebase_service

logger = logging.getLogger(__name__)


class UserService:
    """
    Service for user profile CRUD operations in Firestore.

    Patterns (matching firebase_service.py):
    - Singleton via module-level global + getter
    - self.db is None guard on every method
    - try/except with graceful degradation
    - Returns bool / None / [] on failure
    """

    COLLECTION = "user_profiles"
    MOOD_HISTORY_CAP = 50

    def __init__(self):
        self._firebase = get_firebase_service()

    @property
    def db(self):
        return self._firebase.db

    # ── CREATE ───────────────────────────────────────

    def create_user_profile(self, profile_data: dict) -> Optional[str]:
        """
        Create a new user profile document in Firestore.

        Args:
            profile_data: dict from UserProfile.model_dump()

        Returns:
            user_id on success, None on failure
        """
        if self.db is None:
            print("⚠️ Firebase not available, skipping profile creation")
            return None

        try:
            user_id = profile_data.get("user_id")
            if not user_id:
                user_id = str(uuid.uuid4())
                profile_data["user_id"] = user_id

            profile_data["created_at"] = firestore.SERVER_TIMESTAMP
            profile_data["updated_at"] = firestore.SERVER_TIMESTAMP

            doc_ref = self.db.collection(self.COLLECTION).document(user_id)
            doc_ref.set(profile_data)

            print(f"✅ Created user profile {user_id}")
            return user_id

        except Exception as e:
            print(f"❌ Failed to create user profile: {e}")
            return None

    # ── READ ─────────────────────────────────────────

    def get_user_profile(self, user_id: str) -> Optional[dict]:
        """Retrieve a user profile by user_id."""
        if self.db is None:
            print("⚠️ Firebase not available, cannot retrieve profile")
            return None

        try:
            doc = self.db.collection(self.COLLECTION).document(user_id).get()
            if doc.exists:
                data = doc.to_dict()
                # Convert Firestore timestamps to ISO strings
                for field in ("created_at", "updated_at"):
                    val = data.get(field)
                    if val and hasattr(val, "isoformat"):
                        data[field] = val.isoformat()
                print(f"📱 Retrieved user profile {user_id}")
                return data
            else:
                print(f"⚠️ User profile {user_id} not found")
                return None

        except Exception as e:
            print(f"❌ Failed to retrieve user profile {user_id}: {e}")
            return None

    # ── UPDATE ───────────────────────────────────────

    def update_user_profile(self, user_id: str, update_data: dict) -> bool:
        """
        Update specific layers of a user profile.
        update_data should contain only the layer keys to replace,
        e.g. {"cognitive": {...}, "affective": {...}}
        """
        if self.db is None:
            print("⚠️ Firebase not available, skipping profile update")
            return False

        try:
            update_data["updated_at"] = firestore.SERVER_TIMESTAMP
            doc_ref = self.db.collection(self.COLLECTION).document(user_id)
            doc_ref.update(update_data)
            print(f"✅ Updated user profile {user_id}")
            return True

        except Exception as e:
            print(f"❌ Failed to update user profile {user_id}: {e}")
            return False

    # ── DELETE ────────────────────────────────────────

    def delete_user_profile(self, user_id: str) -> bool:
        """Delete a user profile."""
        if self.db is None:
            print("⚠️ Firebase not available, cannot delete profile")
            return False

        try:
            self.db.collection(self.COLLECTION).document(user_id).delete()
            print(f"✅ Deleted user profile {user_id}")
            return True

        except Exception as e:
            print(f"❌ Failed to delete user profile {user_id}: {e}")
            return False

    # ── MOOD APPEND ──────────────────────────────────

    def append_mood_entry(self, user_id: str, mood_entry: dict) -> bool:
        """
        Append a mood entry to the user's affective.mood_history.
        Caps the list at MOOD_HISTORY_CAP entries (drops oldest).
        Also updates affective.current_mood.
        """
        if self.db is None:
            return False

        try:
            doc_ref = self.db.collection(self.COLLECTION).document(user_id)
            doc = doc_ref.get()
            if not doc.exists:
                print(f"⚠️ User profile {user_id} not found for mood update")
                return False

            data = doc.to_dict()
            affective = data.get("affective", {})
            mood_history = affective.get("mood_history", [])

            mood_history.append(mood_entry)
            if len(mood_history) > self.MOOD_HISTORY_CAP:
                mood_history = mood_history[-self.MOOD_HISTORY_CAP:]

            doc_ref.update({
                "affective.current_mood": mood_entry.get("mood", "neutral"),
                "affective.mood_history": mood_history,
                "updated_at": firestore.SERVER_TIMESTAMP
            })
            print(f"✅ Recorded mood '{mood_entry.get('mood')}' for user {user_id}")
            return True

        except Exception as e:
            print(f"❌ Failed to append mood for {user_id}: {e}")
            return False

    # ── PROFILE-TO-PROMPT FORMATTER ──────────────────

    def format_profile_for_prompt(self, profile_data: dict) -> str:
        """
        Convert a user profile dict into a concise text block
        suitable for injecting into the LLM system prompt.
        Only includes non-empty fields to save tokens.
        """
        parts = []

        # Persona layer
        persona = profile_data.get("persona", {})
        name = persona.get("display_name", "")
        if name:
            parts.append(f"User's name: {name}")

        comm_pref = persona.get("communication_preference", "")
        if comm_pref:
            parts.append(f"Communication preference: {comm_pref}")

        traits = persona.get("personality_traits", [])
        if traits:
            parts.append(f"Personality: {', '.join(traits)}")

        interests = persona.get("interests", [])
        if interests:
            parts.append(f"Interests: {', '.join(interests)}")

        age = persona.get("age_range", "")
        if age:
            parts.append(f"Age range: {age}")

        # Affective layer
        affective = profile_data.get("affective", {})
        mood = affective.get("current_mood", "")
        if mood and mood != "neutral":
            parts.append(f"Current mood: {mood}")

        triggers = affective.get("emotional_triggers", [])
        if triggers:
            parts.append(f"Emotional triggers to be mindful of: {', '.join(triggers)}")

        # Recent mood trend from auto-detected entries
        mood_history = affective.get("mood_history", [])
        if mood_history:
            recent = mood_history[-3:]
            trend_parts = []
            for entry in recent:
                m = entry.get("mood", "neutral")
                i = entry.get("intensity", 5)
                if entry.get("detected_via") == "text_auto":
                    trend_parts.append(f"{m} ({i}/10)")
            if trend_parts:
                parts.append(f"Recent mood trend: {' -> '.join(trend_parts)}")

        # Motivational layer
        motivational = profile_data.get("motivational", {})
        values = motivational.get("core_values", [])
        if values:
            parts.append(f"Core values: {', '.join(values)}")

        goals = motivational.get("goals", [])
        active_goals = [g["title"] for g in goals if g.get("status") == "active"]
        if active_goals:
            parts.append(f"Active goals: {', '.join(active_goals)}")

        motivations = motivational.get("motivations", [])
        if motivations:
            parts.append(f"Motivations: {', '.join(motivations)}")

        # Behavioral layer
        behavioral = profile_data.get("behavioral", {})
        sleep = behavioral.get("sleep_pattern", "")
        if sleep:
            parts.append(f"Sleep pattern: {sleep}")

        exercise = behavioral.get("exercise_frequency", "")
        if exercise:
            parts.append(f"Exercise frequency: {exercise}")

        habits = behavioral.get("habits", [])
        active_habits = [h["name"] for h in habits if h.get("status") == "active"]
        if active_habits:
            parts.append(f"Active habits: {', '.join(active_habits)}")

        # Cognitive layer
        cognitive = profile_data.get("cognitive", {})
        distortions = cognitive.get("cognitive_distortions", [])
        if distortions:
            parts.append(f"Known cognitive patterns to watch for: {', '.join(distortions)}")

        if not parts:
            return ""

        return "--- USER PROFILE ---\n" + "\n".join(parts) + "\n--- END PROFILE ---"


# ── Singleton ─────────────────────────────────────────

_user_service = None


def get_user_service() -> UserService:
    """Get or create UserService singleton instance"""
    global _user_service
    if _user_service is None:
        _user_service = UserService()
    return _user_service
