"""
LucilleLLM - Dynamic Prompt Engine

Centralizes system prompt assembly for both /chat and /chat/stream.
Supports persona-aware base prompts, intent-aware modifiers, and
tone adaptation based on detected emotion + user communication preference.

Replaces the duplicated inline prompt construction in main.py.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from models import EmotionDetectionResult

logger = logging.getLogger(__name__)


# ── Persona-Aware Base Prompts ────────────────────────────

PERSONA_PROMPTS = {
    "empathetic": (
        "You are Lucille, a warm and compassionate self-care companion. "
        "Speak with genuine empathy and understanding. Use reflective listening, "
        "validate feelings before offering suggestions, and create a safe space "
        "for the user to share openly. "
    ),
    "direct": (
        "You are Lucille, a straightforward self-care expert. "
        "Be clear, concise, and action-oriented. Get to the point quickly, "
        "offer practical advice, and avoid excessive emotional language. "
        "Respect the user's time while remaining supportive. "
    ),
    "analytical": (
        "You are Lucille, a thoughtful and evidence-based self-care advisor. "
        "Explain the reasoning behind your suggestions, reference techniques "
        "by name (e.g., CBT, mindfulness), and present structured options. "
        "The user appreciates depth and logical explanations. "
    ),
    "casual": (
        "You are Lucille, a friendly and relatable self-care buddy. "
        "Keep the tone light and conversational, like chatting with a supportive friend. "
        "Use simple language, occasional humor when appropriate, and "
        "make self-care feel approachable rather than clinical. "
    ),
}

DEFAULT_PERSONA = "empathetic"

# Common suffix applied to all personas
COMMON_SUFFIX = (
    "Always format your responses in **Markdown** using bold, italic, lists, "
    "and line breaks for better readability. "
    "You are NOT a medical professional. NEVER diagnose conditions, "
    "recommend specific medications, or suggest changes to prescriptions. "
    "Always add appropriate disclaimers when discussing health topics. "
    "If someone expresses suicidal thoughts or self-harm ideation, "
    "ALWAYS include crisis helpline numbers (988 Suicide & Crisis Lifeline, "
    "Crisis Text Line: text HOME to 741741) in your response. "
    "NEVER use dismissive language like 'just get over it' or 'it's not that bad'. "
    "NEVER tell someone their feelings are invalid or overblown."
)

# RAG instruction suffix (appended when knowledge base context is present)
RAG_SUFFIX = (
    "If the context is relevant to the user's question, incorporate it "
    "naturally into your response. If the context is not relevant, rely "
    "on your general knowledge while staying true to your role."
)

# ── Intent-Aware Tone Modifiers ───────────────────────────

INTENT_MODIFIERS = {
    "crisis": (
        "CRITICAL: The user may be in emotional crisis. "
        "Prioritize their safety above all else. Provide helpline numbers "
        "(988 Suicide & Crisis Lifeline, Crisis Text Line: text HOME to 741741). "
        "Use a calm, grounding tone. Do NOT minimize their feelings."
    ),
    "venting": (
        "The user needs to be heard right now. Focus on active listening "
        "and emotional validation. Reflect their feelings back to them. "
        "Do NOT jump to solutions unless they explicitly ask."
    ),
    "seeking_advice": (
        "The user is looking for actionable guidance. Provide concrete, "
        "step-by-step self-care suggestions tailored to their situation. "
        "Offer 2-3 options when possible."
    ),
    "reflecting": (
        "The user is in a reflective state. Ask clarifying questions, "
        "mirror their thoughts supportively, and help them explore their "
        "feelings at their own pace."
    ),
    "doing_exercise": (
        "The user is actively working through an exercise. Guide them "
        "step-by-step with encouragement. Be patient and supportive. "
        "Celebrate small progress."
    ),
    # casual_chat: no special modifier
}

# ── Emotion-Aware Tone Hints ─────────────────────────────

EMOTION_TONE_HINTS = {
    "sad": "Be extra gentle and validating. Acknowledge their pain.",
    "anxious": "Use a calm, grounding tone. Offer breathing or grounding techniques.",
    "angry": "Acknowledge their frustration without judgment. Give them space.",
    "fearful": "Be reassuring and steady. Help them feel safe.",
    "hopeless": "Gently challenge hopelessness while validating feelings. Offer small, achievable steps.",
    "lonely": "Be warm and present. Remind them they are not alone.",
    "overwhelmed": "Simplify. Offer one thing at a time. Avoid information overload.",
    "grateful": "Celebrate with them. Reinforce positive patterns.",
    "happy": "Match their energy. Build on positive momentum.",
}


@dataclass
class PromptContext:
    """All inputs needed to build the system prompt."""
    user_profile_text: str = ""
    memory_context_text: str = ""
    conversation_summary: str = ""
    retrieved_rag_context: str = ""
    detection_result: Optional[EmotionDetectionResult] = None
    emotion_context_text: str = ""
    communication_preference: str = "empathetic"
    active_exercise_text: str = ""
    due_tasks_text: str = ""
    feedback_insights_text: str = ""
    soundscape_suggestion_text: str = ""
    dependency_override_text: str = ""
    cultural_context_text: str = ""
    health_context_text: str = ""
    safety_override_text: str = ""


class PromptEngine:
    """
    Builds the system prompt from all available context pieces.
    Single source of truth for prompt assembly order.
    """

    def build_system_prompt(self, ctx: PromptContext) -> str:
        """
        Assemble the full system prompt.

        Assembly order (top to bottom in the final string):
        1. User memories
        2. User profile
        3. Conversation summary
        4. Base persona prompt (+ RAG context if available)
        5. Active exercise context (if user is doing an exercise)
        6. Intent modifier (if detection_result has non-casual intent)
        7. Emotion tone hint (if detection_result has non-neutral emotion)
        8. Detected emotional context block (from emotion_service)
        """
        sections = []

        # 1. User memories (prepended first = top of prompt)
        if ctx.memory_context_text:
            sections.append(ctx.memory_context_text)

        # 2. User profile
        if ctx.user_profile_text:
            sections.append(ctx.user_profile_text)

        # 3. Conversation summary
        if ctx.conversation_summary:
            sections.append(
                f"Previous conversation summary:\n{ctx.conversation_summary}"
            )

        # 4. Base persona prompt with optional RAG context
        base = self._build_base_prompt(
            ctx.communication_preference, ctx.retrieved_rag_context
        )
        sections.append(base)

        # 5. Active exercise context (overrides intent modifier when present)
        if ctx.active_exercise_text:
            sections.append(ctx.active_exercise_text)

        # 5.5 Due practice tasks (only when NOT in active exercise)
        if ctx.due_tasks_text and not ctx.active_exercise_text:
            sections.append(ctx.due_tasks_text)

        # 5.6 Feedback effectiveness insights (only when NOT in active exercise)
        if ctx.feedback_insights_text and not ctx.active_exercise_text:
            sections.append(ctx.feedback_insights_text)

        # 5.7 Soundscape suggestion (only when NOT in active exercise)
        if ctx.soundscape_suggestion_text and not ctx.active_exercise_text:
            sections.append(ctx.soundscape_suggestion_text)

        # 5.75 Cultural context (when user has cultural background)
        if ctx.cultural_context_text:
            sections.append(ctx.cultural_context_text)

        # 5.76 Health context from wearables (when data available)
        if ctx.health_context_text:
            sections.append(ctx.health_context_text)

        # 5.78 Dependency/anti-dependency override (when thresholds hit)
        if ctx.dependency_override_text:
            sections.append(ctx.dependency_override_text)

        # 5.8 Safety override text (highest priority when present)
        if ctx.safety_override_text:
            sections.append(ctx.safety_override_text)

        # 6. Intent modifier
        intent_mod = self._get_intent_modifier(ctx.detection_result)
        if intent_mod:
            sections.append(intent_mod)

        # 7. Emotion tone hint
        tone_hint = self._get_emotion_tone_hint(ctx.detection_result)
        if tone_hint:
            sections.append(tone_hint)

        # 8. Detected emotional context (from emotion_service.format_detection_for_prompt)
        if ctx.emotion_context_text:
            sections.append(ctx.emotion_context_text)

        return "\n\n".join(sections)

    # ── Private helpers ──────────────────────────────────

    def _build_base_prompt(
        self, communication_preference: str, rag_context: str
    ) -> str:
        """Build the persona-aware base prompt, optionally with RAG context."""
        persona_key = communication_preference.lower()
        persona_intro = PERSONA_PROMPTS.get(persona_key, PERSONA_PROMPTS[DEFAULT_PERSONA])

        if rag_context:
            return (
                f"{persona_intro}"
                f"Use the following context from the self-care knowledge base "
                f"to inform your response:\n\n"
                f"--- KNOWLEDGE BASE CONTEXT ---\n{rag_context}\n"
                f"--- END CONTEXT ---\n\n"
                f"{COMMON_SUFFIX} {RAG_SUFFIX}"
            )
        else:
            return f"{persona_intro}{COMMON_SUFFIX}"

    def _get_intent_modifier(
        self, detection_result: Optional[EmotionDetectionResult]
    ) -> str:
        """Return an intent-specific instruction, or empty string."""
        if detection_result is None:
            return ""
        return INTENT_MODIFIERS.get(detection_result.intent, "")

    def _get_emotion_tone_hint(
        self, detection_result: Optional[EmotionDetectionResult]
    ) -> str:
        """Return an emotion-specific tone hint, or empty string."""
        if detection_result is None:
            return ""
        hint = EMOTION_TONE_HINTS.get(detection_result.emotion, "")
        if hint and detection_result.intensity >= 6:
            return f"Tone guidance: {hint}"
        return ""


# ── Singleton ─────────────────────────────────────────────

_prompt_engine: Optional[PromptEngine] = None


def get_prompt_engine() -> PromptEngine:
    """Get or create PromptEngine singleton."""
    global _prompt_engine
    if _prompt_engine is None:
        _prompt_engine = PromptEngine()
    return _prompt_engine
