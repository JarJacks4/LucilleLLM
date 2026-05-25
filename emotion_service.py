"""
LucilleLLM - Emotion & Intent Detection Service

Detects emotion and intent from user text messages using OpenAI gpt-4o-mini.
Follows the singleton pattern from firebase_service.py and user_service.py.
"""

import base64
import json
import logging
from typing import Optional

from openai import OpenAI

from models import EmotionDetectionResult

logger = logging.getLogger(__name__)

# Valid values for validation
VALID_EMOTIONS = {
    "happy", "sad", "anxious", "angry", "fearful", "disgusted",
    "surprised", "neutral", "hopeless", "lonely", "overwhelmed", "grateful",
}
VALID_INTENTS = {
    "venting", "seeking_advice", "crisis", "casual_chat",
    "doing_exercise", "reflecting",
}

DETECTION_SYSTEM_PROMPT = (
    "You are an emotion and intent classifier for a therapy chatbot. "
    "Analyze the user's message and return a JSON object with exactly these fields:\n"
    '- "emotion": one of [happy, sad, anxious, angry, fearful, disgusted, '
    "surprised, neutral, hopeless, lonely, overwhelmed, grateful]\n"
    '- "intensity": integer 1-10 (1=barely noticeable, 10=overwhelming)\n'
    '- "intent": one of [venting, seeking_advice, crisis, casual_chat, '
    "doing_exercise, reflecting]\n"
    '- "confidence": float 0.0-1.0 (your confidence in the classification)\n\n'
    "Return ONLY valid JSON, no explanation."
)

IMAGE_DETECTION_SYSTEM_PROMPT = (
    "You are a facial-emotion classifier for a self-care app. "
    "Examine the face(s) in the image and identify the single dominant emotion. "
    "Return a JSON object with exactly these fields:\n"
    '- "emotion": one of [happy, sad, anxious, angry, fearful, disgusted, '
    "surprised, neutral, hopeless, lonely, overwhelmed, grateful]\n"
    '- "intensity": integer 1-10 (1=barely noticeable, 10=overwhelming)\n'
    '- "confidence": float 0.0-1.0 (your confidence in the classification)\n'
    '- "face_detected": boolean (true only if at least one human face is visible)\n\n'
    'If no human face is visible, set face_detected=false, emotion="neutral", '
    "intensity=1, and confidence=0.0. "
    "Return ONLY valid JSON, no explanation."
)


class EmotionService:
    """
    Service for detecting emotion and intent from user text.

    Patterns (matching firebase_service.py / user_service.py):
    - Singleton via module-level global + getter
    - try/except with graceful degradation
    - Returns fallback EmotionDetectionResult on failure
    """

    def __init__(self, openai_client: OpenAI, model: str = "gpt-4o-mini",
                 vision_client: Optional[OpenAI] = None,
                 vision_model: Optional[str] = None):
        self._client = openai_client
        self._model = model
        # Vision client/model for image-based mood detection. Falls back to the
        # main text client/model when a dedicated one isn't supplied.
        self._vision_client = vision_client or openai_client
        self._vision_model = vision_model or model

    def detect(self, user_message: str) -> EmotionDetectionResult:
        """
        Detect emotion and intent from a user message.

        Args:
            user_message: The raw text of the user's chat message.

        Returns:
            EmotionDetectionResult with emotion, intensity, intent, confidence.
            Returns safe defaults on any failure.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": DETECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=80,
                temperature=0,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content
            data = json.loads(raw)

            # Validate and clamp values
            emotion = data.get("emotion", "neutral")
            if emotion not in VALID_EMOTIONS:
                emotion = "neutral"

            intensity = data.get("intensity", 5)
            try:
                intensity = int(intensity)
                intensity = max(1, min(10, intensity))
            except (TypeError, ValueError):
                intensity = 5

            intent = data.get("intent", "casual_chat")
            if intent not in VALID_INTENTS:
                intent = "casual_chat"

            confidence = data.get("confidence", 0.5)
            try:
                confidence = float(confidence)
                confidence = max(0.0, min(1.0, confidence))
            except (TypeError, ValueError):
                confidence = 0.5

            result = EmotionDetectionResult(
                emotion=emotion,
                intensity=intensity,
                intent=intent,
                confidence=round(confidence, 2),
            )

            logger.info(
                f"🎭 Detected emotion={result.emotion} "
                f"(intensity={result.intensity}, confidence={result.confidence}), "
                f"intent={result.intent}"
            )
            return result

        except Exception as e:
            logger.warning(f"⚠️ Emotion detection failed, using defaults: {e}")
            return EmotionDetectionResult()

    def detect_from_image(self, image_bytes: bytes,
                          mime_type: str = "image/jpeg") -> EmotionDetectionResult:
        """
        Detect the dominant emotion from a face image using an OpenAI vision model.

        Replaces the legacy ViT classifier. Intent is always 'casual_chat' since
        it can't be inferred from a still image. Returns safe neutral defaults on
        any failure or when no face is detected.

        Args:
            image_bytes: Raw bytes of the uploaded image.
            mime_type: MIME type of the image (e.g. 'image/jpeg', 'image/png').

        Returns:
            EmotionDetectionResult with emotion, intensity, and confidence.
        """
        try:
            if mime_type not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
                mime_type = "image/jpeg"
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            data_url = f"data:{mime_type};base64,{b64}"

            response = self._vision_client.chat.completions.create(
                model=self._vision_model,
                messages=[
                    {"role": "system", "content": IMAGE_DETECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text",
                         "text": "Classify the dominant facial emotion in this image."},
                        {"type": "image_url",
                         "image_url": {"url": data_url, "detail": "low"}},
                    ]},
                ],
                max_tokens=100,
                temperature=0,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content
            data = json.loads(raw)

            emotion = data.get("emotion", "neutral")
            if emotion not in VALID_EMOTIONS:
                emotion = "neutral"

            intensity = data.get("intensity", 5)
            try:
                intensity = max(1, min(10, int(intensity)))
            except (TypeError, ValueError):
                intensity = 5

            confidence = data.get("confidence", 0.5)
            try:
                confidence = max(0.0, min(1.0, float(confidence)))
            except (TypeError, ValueError):
                confidence = 0.5

            result = EmotionDetectionResult(
                emotion=emotion,
                intensity=intensity,
                intent="casual_chat",
                confidence=round(confidence, 2),
            )
            logger.info(
                f"🖼️ Image emotion={result.emotion} "
                f"(intensity={result.intensity}, confidence={result.confidence})"
            )
            return result

        except Exception as e:
            logger.warning(f"⚠️ Image emotion detection failed, using defaults: {e}")
            return EmotionDetectionResult()

    def format_detection_for_prompt(self, result: EmotionDetectionResult) -> str:
        """
        Format detection result as a text block for system prompt injection.
        Includes therapy-aware instructions based on detected intent.
        """
        parts = []

        if result.emotion != "neutral":
            parts.append(
                f"Detected emotion in this message: {result.emotion} "
                f"(intensity {result.intensity}/10)"
            )

        if result.intent == "crisis":
            parts.append(
                "CRITICAL: User may be in crisis. Prioritize safety. "
                "Provide helpline numbers immediately."
            )
        elif result.intent == "venting":
            parts.append(
                "User intent: venting. Focus on active listening and validation. "
                "Avoid jumping to solutions unless asked."
            )
        elif result.intent == "seeking_advice":
            parts.append(
                "User intent: seeking advice. Provide actionable self-care suggestions."
            )
        elif result.intent == "reflecting":
            parts.append(
                "User intent: reflecting. Ask clarifying questions and mirror back "
                "their thoughts supportively."
            )
        elif result.intent == "doing_exercise":
            parts.append(
                "User intent: doing an exercise. Guide and encourage them through it."
            )
        # casual_chat gets no special instruction

        if not parts:
            return ""

        return (
            "--- DETECTED EMOTIONAL CONTEXT ---\n"
            + "\n".join(parts)
            + "\n--- END EMOTIONAL CONTEXT ---"
        )


# ── Singleton ─────────────────────────────────────────

_emotion_service: Optional[EmotionService] = None


def get_emotion_service(
    openai_client: OpenAI = None, model: str = "gpt-4o-mini",
    vision_client: OpenAI = None, vision_model: str = None,
) -> EmotionService:
    """Get or create EmotionService singleton instance.
    openai_client must be provided on first call."""
    global _emotion_service
    if _emotion_service is None:
        if openai_client is None:
            raise ValueError("openai_client must be provided on first call to get_emotion_service()")
        _emotion_service = EmotionService(
            openai_client, model,
            vision_client=vision_client, vision_model=vision_model,
        )
    return _emotion_service
