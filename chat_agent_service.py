"""
LucilleLLM - Chat Agent Service (OpenAI Function Calling)

Defines 20 tools that GPT-4o-mini can invoke during conversations:
  - 12 wrapping existing services (therapy, soundscape, memory, progress, etc.)
  - 4 self-contained tools (breathing, journaling, sleep, grounding)
  - 2 anti-dependency & cultural tools (wellbeing check, international resources)
  - 2 GDPR compliance tools (data export, data deletion)

Uses OpenAI's native function calling — NOT LangChain @tool.
user_id is injected server-side; the LLM never sees it.

Follows the singleton pattern from other services.
"""

import json
import logging
import random
from datetime import datetime
from typing import Any, Callable, Optional

from feedback_service import get_feedback_service
from memory_service import get_memory_service
from models import PracticeTask
from progress_service import get_progress_service
from safety_service import get_safety_service
from soundscape_service import get_soundscape_service
from therapy_service import get_therapy_service
from user_service import get_user_service

logger = logging.getLogger(__name__)


# ── Tools that require server-side user_id injection ──────

TOOLS_REQUIRING_USER_ID = {
    "start_exercise",
    "search_memories",
    "get_user_progress",
    "get_effectiveness",
    "log_mood",
    "get_mood_history",
    "create_task",
    "get_due_tasks",
    "get_wellbeing_check",
    "get_international_resources",
    "request_data_export",
    "request_data_deletion",
    "get_health_summary",
}


# ── OpenAI Tool Definitions (21 tools) ───────────────────

TOOL_DEFINITIONS = [
    # 1. recommend_exercise
    {
        "type": "function",
        "function": {
            "name": "recommend_exercise",
            "description": (
                "Recommend therapy exercises (CBT, ACT, DBT, MI) based on the "
                "user's current emotional state and intent. Use when the user seems "
                "like they could benefit from a structured exercise."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "emotion": {
                        "type": "string",
                        "description": "The user's current emotion (e.g. 'anxious', 'sad', 'overwhelmed', 'angry', 'hopeless', 'lonely', 'neutral')",
                    },
                    "intent": {
                        "type": "string",
                        "description": "The user's intent (e.g. 'seeking_advice', 'doing_exercise', 'reflecting', 'venting', 'casual_chat')",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of exercises to return (default: 3)",
                    },
                },
                "required": ["emotion", "intent"],
            },
        },
    },
    # 2. start_exercise
    {
        "type": "function",
        "function": {
            "name": "start_exercise",
            "description": (
                "Start a specific therapy exercise session for the user. "
                "Use after recommending exercises when the user chooses one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "exercise_id": {
                        "type": "string",
                        "description": "The exercise_id to start (e.g. 'cbt_thought_record', 'dbt_distress_tolerance')",
                    },
                },
                "required": ["exercise_id"],
            },
        },
    },
    # 3. recommend_soundscape
    {
        "type": "function",
        "function": {
            "name": "recommend_soundscape",
            "description": (
                "Recommend ambient soundscapes for relaxation, focus, or "
                "exercise accompaniment based on the user's mood."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "emotion": {
                        "type": "string",
                        "description": "The user's current emotion",
                    },
                    "intent": {
                        "type": "string",
                        "description": "The user's intent",
                    },
                    "exercise_id": {
                        "type": "string",
                        "description": "Optional exercise_id to pair with a soundscape",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum soundscapes to return (default: 3)",
                    },
                },
                "required": ["emotion", "intent"],
            },
        },
    },
    # 4. search_memories
    {
        "type": "function",
        "function": {
            "name": "search_memories",
            "description": (
                "Search the user's stored memories for relevant past context. "
                "Use when the user references something from a past conversation "
                "or you need historical context about them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant memories",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum memories to return (default: 5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    # 5. get_user_progress
    {
        "type": "function",
        "function": {
            "name": "get_user_progress",
            "description": (
                "Get the user's progress summary including exercise completion "
                "rates, streaks, and practice stats. Use when the user asks about "
                "their progress or you want to celebrate milestones."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # 6. get_effectiveness
    {
        "type": "function",
        "function": {
            "name": "get_effectiveness",
            "description": (
                "Get the user's effectiveness profile showing which therapy "
                "modalities and exercises have been most/least helpful. "
                "Use to personalize recommendations."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # 7. get_crisis_resources
    {
        "type": "function",
        "function": {
            "name": "get_crisis_resources",
            "description": (
                "Get crisis helpline numbers and resources. Use whenever the "
                "user expresses significant distress, self-harm ideation, "
                "or asks for help resources."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # 8. search_knowledge_base
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the self-care knowledge base for relevant information. "
                "Use when the user asks factual questions about self-care, "
                "wellbeing techniques, or mental health topics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for the knowledge base",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to retrieve (default: 5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    # 9. log_mood
    {
        "type": "function",
        "function": {
            "name": "log_mood",
            "description": (
                "Log a mood entry for the user. Use when the user explicitly "
                "reports their mood or you want to track an emotional state they described."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mood": {
                        "type": "string",
                        "description": "The mood to log (e.g. 'anxious', 'happy', 'sad', 'calm')",
                    },
                    "intensity": {
                        "type": "integer",
                        "description": "Intensity from 1-10 (1=barely noticeable, 10=overwhelming)",
                    },
                    "context": {
                        "type": "string",
                        "description": "What triggered or accompanies this mood",
                    },
                },
                "required": ["mood", "intensity"],
            },
        },
    },
    # 10. get_mood_history
    {
        "type": "function",
        "function": {
            "name": "get_mood_history",
            "description": (
                "Retrieve the user's recent mood history. Use when the user "
                "asks about mood patterns or you want to discuss trends."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of mood entries to return (default: 10)",
                    },
                },
                "required": [],
            },
        },
    },
    # 11. create_task
    {
        "type": "function",
        "function": {
            "name": "create_task",
            "description": (
                "Create a practice task or homework assignment for the user. "
                "Use when suggesting between-session practice."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_exercise_id": {
                        "type": "string",
                        "description": "The exercise this task relates to",
                    },
                    "title": {
                        "type": "string",
                        "description": "Short task title",
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed instructions for the practice",
                    },
                    "due_date": {
                        "type": "string",
                        "description": "ISO date string for when this task is due",
                    },
                    "target_count": {
                        "type": "integer",
                        "description": "How many times to practice (default: 1)",
                    },
                },
                "required": ["source_exercise_id", "title"],
            },
        },
    },
    # 12. get_due_tasks
    {
        "type": "function",
        "function": {
            "name": "get_due_tasks",
            "description": (
                "Get the user's due or overdue practice tasks. "
                "Use to check in on homework or when the user asks about tasks."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # ── 4 New Self-Contained Tools ────────────────────────
    # 13. breathing_timer
    {
        "type": "function",
        "function": {
            "name": "breathing_timer",
            "description": (
                "Get structured breathing exercise patterns. Use when the user "
                "needs help calming down, is anxious, or asks for breathing exercises."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "enum": ["4-7-8", "box", "diaphragmatic", "resonant"],
                        "description": (
                            "Breathing pattern type. '4-7-8' for relaxation, "
                            "'box' for focus, 'diaphragmatic' for deep calming, "
                            "'resonant' for heart rate variability."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    # 14. journal_prompt
    {
        "type": "function",
        "function": {
            "name": "journal_prompt",
            "description": (
                "Get a context-aware journaling prompt based on the user's "
                "current emotional state. Use when suggesting journaling or "
                "when the user wants to write/reflect."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "emotion": {
                        "type": "string",
                        "description": "The user's current emotion to tailor the prompt",
                    },
                    "intent": {
                        "type": "string",
                        "description": "The user's intent (e.g. 'reflecting', 'venting', 'seeking_advice')",
                    },
                },
                "required": [],
            },
        },
    },
    # 15. sleep_hygiene_tips
    {
        "type": "function",
        "function": {
            "name": "sleep_hygiene_tips",
            "description": (
                "Get targeted sleep hygiene advice. Use when the user mentions "
                "sleep difficulties, insomnia, or asks for sleep tips."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "issue": {
                        "type": "string",
                        "enum": [
                            "falling_asleep",
                            "staying_asleep",
                            "sleep_quality",
                            "sleep_schedule",
                            "general",
                        ],
                        "description": "The specific sleep issue the user is experiencing",
                    },
                },
                "required": [],
            },
        },
    },
    # 16. grounding_exercise
    {
        "type": "function",
        "function": {
            "name": "grounding_exercise",
            "description": (
                "Get a 5-4-3-2-1 grounding exercise script. Use when the user "
                "is experiencing anxiety, panic, dissociation, or feeling overwhelmed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intensity": {
                        "type": "string",
                        "enum": ["mild", "moderate", "severe"],
                        "description": "How intense the user's distress appears to be",
                    },
                },
                "required": [],
            },
        },
    },
    # 17. get_wellbeing_check
    {
        "type": "function",
        "function": {
            "name": "get_wellbeing_check",
            "description": (
                "Get a wellbeing check for the user including their usage patterns "
                "and dependency risk assessment. Use when the user asks about their "
                "app usage, when you sense they might be relying too heavily on the app, "
                "or to check if a break would be beneficial."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    # 18. get_international_resources
    {
        "type": "function",
        "function": {
            "name": "get_international_resources",
            "description": (
                "Get crisis resources appropriate for the user's country or culture. "
                "Use when the user is not in the US, mentions a non-US location, "
                "or when providing crisis resources to a user with a known cultural background."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "country_code": {
                        "type": "string",
                        "description": (
                            "ISO 3166-1 alpha-2 country code (e.g. 'US', 'GB', 'IN', 'AU'). "
                            "Defaults to user's profile country if not provided."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    # ── 2 Phase 13 Tools (GDPR Compliance) ────────────────
    # 19. request_data_export
    {
        "type": "function",
        "function": {
            "name": "request_data_export",
            "description": (
                "Export all of the user's data for portability (GDPR Article 20). "
                "Use when the user asks to download, export, or see all their data. "
                "Returns a summary of exportable data categories and directs them "
                "to the full export endpoint."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    # 20. request_data_deletion
    {
        "type": "function",
        "function": {
            "name": "request_data_deletion",
            "description": (
                "Request deletion of all the user's data (GDPR Article 17 — Right to Erasure). "
                "Use when the user asks to delete their account, erase their data, or "
                "be forgotten. IMPORTANT: This requires explicit confirmation from the user. "
                "The 'confirm' parameter MUST be set to true by the user before proceeding."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "confirm": {
                        "type": "boolean",
                        "description": (
                            "Must be true to proceed with deletion. If false or missing, "
                            "return a warning explaining that all data will be permanently "
                            "deleted and ask for explicit confirmation."
                        ),
                    },
                },
                "required": ["confirm"],
            },
        },
    },
    # 21. get_health_summary
    {
        "type": "function",
        "function": {
            "name": "get_health_summary",
            "description": (
                "Get a summary of the user's recent health metrics from wearable data, "
                "including sleep patterns, activity levels, and trends. "
                "Use when discussing the user's physical health, sleep, or activity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Number of days to summarize (default: 7)",
                    },
                },
                "required": [],
            },
        },
    },
]


# ── Static Content for Self-Contained Tools ──────────────

BREATHING_PATTERNS = {
    "4-7-8": {
        "name": "4-7-8 Relaxation Breath",
        "steps": [
            "Exhale completely through your mouth, making a whoosh sound.",
            "Close your mouth and inhale quietly through your nose for 4 seconds.",
            "Hold your breath for 7 seconds.",
            "Exhale completely through your mouth for 8 seconds, making a whoosh sound.",
            "This is one cycle. Repeat for 4 cycles total.",
        ],
        "rounds": 4,
        "total_seconds_per_cycle": 19,
        "purpose": "Activates the parasympathetic nervous system. Best for falling asleep, reducing anxiety, and managing anger.",
    },
    "box": {
        "name": "Box Breathing (Navy SEAL technique)",
        "steps": [
            "Inhale slowly through your nose for 4 seconds.",
            "Hold your breath for 4 seconds.",
            "Exhale slowly through your mouth for 4 seconds.",
            "Hold your breath (lungs empty) for 4 seconds.",
            "This is one cycle. Repeat for 4-6 cycles.",
        ],
        "rounds": 5,
        "total_seconds_per_cycle": 16,
        "purpose": "Balances the autonomic nervous system. Best for focus, stress management, and grounding.",
    },
    "diaphragmatic": {
        "name": "Diaphragmatic (Belly) Breathing",
        "steps": [
            "Place one hand on your chest, the other on your belly.",
            "Inhale slowly through your nose for 4 seconds, feeling your belly rise (chest stays still).",
            "Pause briefly at the top of the breath.",
            "Exhale slowly through pursed lips for 6 seconds, feeling your belly fall.",
            "Repeat for 5-10 minutes.",
        ],
        "rounds": 10,
        "total_seconds_per_cycle": 10,
        "purpose": "Engages the diaphragm fully. Best for deep relaxation, lowering blood pressure, and chronic stress.",
    },
    "resonant": {
        "name": "Resonant (Coherent) Breathing",
        "steps": [
            "Inhale slowly through your nose for 5 seconds.",
            "Exhale slowly through your nose for 5 seconds.",
            "Maintain this steady 5-5 rhythm without forcing.",
            "Continue for 5-10 minutes.",
        ],
        "rounds": 12,
        "total_seconds_per_cycle": 10,
        "purpose": "Optimizes heart rate variability (HRV). Best for emotional regulation, sustained calm, and long-term resilience.",
    },
}

JOURNAL_PROMPTS = {
    "sad": [
        "What is one small thing that brought you comfort today, even briefly?",
        "Write a letter to yourself from the perspective of someone who loves you.",
        "What would you say to a friend who was feeling the way you feel right now?",
        "Describe a time when you felt sad before and what eventually helped you feel better.",
        "What are three things you're grateful for, even on a hard day like today?",
    ],
    "anxious": [
        "What is the worst case scenario you're worried about? How likely is it, really?",
        "List everything on your mind right now — get it all out, unfiltered.",
        "What are three things you can control right now, and three things you cannot?",
        "Describe your anxiety as if it were a character. What does it look like? What is it trying to protect you from?",
        "What would your life look like if this worry resolved itself tomorrow?",
    ],
    "angry": [
        "What boundary was crossed that led to this anger?",
        "Write out everything you want to say to the person or situation, uncensored.",
        "Underneath the anger, what other emotion might be hiding? (hurt, fear, disappointment?)",
        "What would you need to happen to feel at peace with this situation?",
        "How has anger served you in the past? When has it not served you?",
    ],
    "overwhelmed": [
        "List everything on your plate right now. Circle the top 3 priorities.",
        "What is one small task you could complete in the next 5 minutes?",
        "If you could delegate or drop one responsibility, what would it be?",
        "Describe your ideal day tomorrow — what would make it manageable?",
        "What did you accomplish today that you're not giving yourself credit for?",
    ],
    "lonely": [
        "Who is someone you could reach out to today, even with a simple text?",
        "Describe a time when you felt deeply connected to someone. What made it special?",
        "What activities make you feel most like yourself?",
        "Write about what kind of friendship or connection you wish you had.",
        "What is one kind thing you could do for yourself right now?",
    ],
    "hopeless": [
        "What is one tiny thing that went right today, even if it seems insignificant?",
        "Write about a time you overcame something you thought was impossible.",
        "If hope were a place, what would it look like? Describe it in detail.",
        "What advice would your future self — who has gotten through this — give you?",
        "What is one small step you could take tomorrow toward something you care about?",
    ],
    "happy": [
        "What contributed to your good mood today? How can you create more of that?",
        "Write a thank-you letter to someone who made a difference in your life.",
        "What strengths did you use today that you're proud of?",
        "Describe this moment of happiness in vivid detail so you can revisit it later.",
        "How can you share this good energy with someone else?",
    ],
    "neutral": [
        "What's on your mind today? Write freely for 5 minutes without editing.",
        "What is one thing you've been putting off that you could address this week?",
        "Describe your ideal self-care routine. How close are you to living it?",
        "What is something new you'd like to learn or try?",
        "Reflect on the past week — what went well and what would you change?",
    ],
}

SLEEP_TIPS = {
    "falling_asleep": {
        "issue": "Difficulty Falling Asleep",
        "tips": [
            "Try the 4-7-8 breathing technique (inhale 4s, hold 7s, exhale 8s) — it activates your parasympathetic nervous system.",
            "Keep your bedroom cool (60-67°F / 15-19°C) — your body needs to cool down to fall asleep.",
            "Stop screens 30-60 minutes before bed, or use a blue-light filter.",
            "If you can't fall asleep after 20 minutes, get up and do something quiet in dim light, then return when you feel sleepy.",
            "Try a body scan meditation: slowly focus on relaxing each body part from toes to head.",
        ],
        "avoid": [
            "Avoid caffeine after 2 PM (it has a 6-hour half-life).",
            "Avoid heavy meals within 2-3 hours of bedtime.",
            "Don't watch the clock — it increases anxiety about not sleeping.",
        ],
    },
    "staying_asleep": {
        "issue": "Waking Up During the Night",
        "tips": [
            "Keep a consistent sleep and wake time, even on weekends.",
            "Avoid alcohol before bed — it fragments sleep in the second half of the night.",
            "If you wake up anxious, keep a notepad by your bed to write down thoughts.",
            "Use white noise or a fan to mask environmental sounds.",
            "Check your room for light sources — even small LEDs can disrupt sleep.",
        ],
        "avoid": [
            "Avoid drinking large amounts of fluids in the 2 hours before bed.",
            "Don't check your phone if you wake up — the light resets your circadian clock.",
        ],
    },
    "sleep_quality": {
        "issue": "Poor Sleep Quality",
        "tips": [
            "Aim for 7-9 hours of sleep opportunity (time in bed).",
            "Exercise regularly, but finish workouts at least 3 hours before bed.",
            "Consider your mattress and pillow — they should support neutral spine alignment.",
            "Try a warm bath or shower 1-2 hours before bed — the subsequent cooling signals sleep.",
            "Limit naps to 20 minutes before 3 PM.",
        ],
        "avoid": [
            "Avoid using your bed for work, TV, or scrolling — train your brain that bed = sleep.",
            "Avoid stimulating activities close to bedtime (intense exercise, arguments, thriller movies).",
        ],
    },
    "sleep_schedule": {
        "issue": "Irregular Sleep Schedule",
        "tips": [
            "Set a fixed wake-up time and stick to it every day, including weekends.",
            "Get bright light exposure within 30 minutes of waking — sunlight is best.",
            "Shift your bedtime gradually (15 minutes earlier/later every 2-3 days).",
            "Create a consistent wind-down routine: same activities in the same order.",
            "Use an alarm for bedtime, not just wake-up.",
        ],
        "avoid": [
            "Avoid sleeping in on weekends — it creates 'social jet lag'.",
            "Avoid napping if you're trying to reset your schedule.",
        ],
    },
    "general": {
        "issue": "General Sleep Hygiene",
        "tips": [
            "Keep a consistent sleep schedule (same bedtime and wake time daily).",
            "Make your bedroom dark, cool, and quiet — invest in blackout curtains if needed.",
            "Develop a 30-minute wind-down routine before bed.",
            "Limit caffeine to the morning and early afternoon.",
            "Regular exercise improves sleep quality, but not too close to bedtime.",
        ],
        "avoid": [
            "Avoid screens in the bedroom.",
            "Avoid using alcohol as a sleep aid.",
            "Avoid large meals, caffeine, and nicotine close to bedtime.",
        ],
    },
}

GROUNDING_SCRIPTS = {
    "mild": {
        "name": "Quick Grounding (Mild Distress)",
        "duration": "2-3 minutes",
        "steps": [
            "Take a slow, deep breath. In for 4 counts, out for 6 counts.",
            "Name 5 things you can see right now. Look around slowly.",
            "Name 4 things you can physically feel (your feet on the floor, the chair beneath you, the air on your skin, your hands in your lap).",
            "Name 3 things you can hear.",
            "Take one more deep breath. You are here. You are safe.",
        ],
        "closing": "You're doing great. This feeling is temporary and it will pass.",
    },
    "moderate": {
        "name": "5-4-3-2-1 Grounding (Moderate Distress)",
        "duration": "5-7 minutes",
        "steps": [
            "Pause and take three slow, deep breaths. Feel your feet firmly on the ground.",
            "5 THINGS YOU CAN SEE: Look around carefully. Name five things in detail — their color, shape, and texture.",
            "4 THINGS YOU CAN TOUCH: Reach out and physically touch four objects. Describe how each one feels.",
            "3 THINGS YOU CAN HEAR: Close your eyes briefly. What three sounds can you identify? Near or far.",
            "2 THINGS YOU CAN SMELL: Breathe in deeply. Can you notice two scents? If not, recall two favorite smells.",
            "1 THING YOU CAN TASTE: Notice any taste in your mouth, or take a sip of water and notice its temperature.",
            "Take three more deep breaths. Feel your body settle. You are present in this moment.",
        ],
        "closing": "You've just brought yourself back to the present moment. Whatever you were feeling is valid, and you handled it. Take a moment to appreciate your own strength.",
    },
    "severe": {
        "name": "Intensive Grounding (Severe Distress / Panic)",
        "duration": "10-15 minutes",
        "steps": [
            "STOP. Press your feet firmly into the ground. Feel the solid surface beneath you.",
            "Hold something cold — ice, a cold can, cold water on your wrists. Focus entirely on the sensation.",
            "Say out loud: 'My name is [your name]. I am in [your location]. Today is [today's date]. I am safe right now.'",
            "5 THINGS YOU CAN SEE: Name them out loud. Describe each one in detail.",
            "4 THINGS YOU CAN TOUCH: Press your hands against different surfaces. Focus on the texture.",
            "3 THINGS YOU CAN HEAR: Listen carefully. Name each sound out loud.",
            "2 THINGS YOU CAN SMELL: Breathe deeply through your nose. Describe what you notice.",
            "1 THING YOU CAN TASTE: Take a sip of water or bite into something. Focus entirely on the flavor.",
            "Now, slowly count backward from 10 to 1, taking a deep breath with each number.",
            "Gently move your fingers and toes. Roll your shoulders. You are coming back to yourself.",
        ],
        "closing": "You just made it through something really hard. That takes incredible strength. If you're still struggling, please reach out to someone you trust or call 988 (Suicide & Crisis Lifeline). You don't have to face this alone.",
        "crisis_note": "If distress persists, please contact: 988 Suicide & Crisis Lifeline (call or text 988) or Crisis Text Line (text HOME to 741741).",
    },
}


# ── ChatAgentService ──────────────────────────────────────


class ChatAgentService:
    """
    Manages OpenAI function calling: tool schema definitions,
    tool execution dispatch, and user_id injection.
    """

    def __init__(self):
        self._therapy_svc = get_therapy_service()
        self._soundscape_svc = get_soundscape_service()
        self._memory_svc = get_memory_service()
        self._progress_svc = get_progress_service()
        self._feedback_svc = get_feedback_service()
        self._safety_svc = get_safety_service()
        self._user_svc = get_user_service()

    def get_tool_definitions(self) -> list:
        """Return the list of 20 OpenAI tool definitions."""
        return TOOL_DEFINITIONS

    def execute_tool_call(
        self,
        tool_name: str,
        tool_args: dict,
        user_id: Optional[str],
        retrieve_context_fn: Optional[Callable] = None,
    ) -> str:
        """
        Execute a single tool call and return the result as a JSON string.

        user_id is injected server-side — the LLM never passes it.
        retrieve_context_fn is a reference to main.py's retrieve_relevant_context.
        """
        try:
            result = self._dispatch(tool_name, tool_args, user_id, retrieve_context_fn)
            return json.dumps(result, default=str)
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {e}")
            return json.dumps({
                "error": f"Tool '{tool_name}' failed: {str(e)}",
                "suggestion": "I couldn't retrieve that information right now. Let me try to help you directly instead.",
            })

    def _dispatch(
        self,
        name: str,
        args: dict,
        user_id: Optional[str],
        retrieve_fn: Optional[Callable],
    ) -> Any:
        """Route tool_name to the appropriate service method."""

        # Validate user_id for tools that require it
        if name in TOOLS_REQUIRING_USER_ID and not user_id:
            return {
                "error": "This feature requires a user profile. Please sign in first."
            }

        # ── 12 Service-Wrapping Tools ─────────────────────

        if name == "recommend_exercise":
            effectiveness_scores = None
            if user_id:
                try:
                    eff = self._feedback_svc.compute_effectiveness(user_id)
                    if eff and eff.total_outcomes >= 2:
                        effectiveness_scores = {
                            "modality_scores": eff.modality_scores,
                            "exercise_scores": eff.exercise_scores,
                        }
                except Exception:
                    pass
            recs = self._therapy_svc.recommend_exercises(
                emotion=args.get("emotion", "neutral"),
                intent=args.get("intent", "casual_chat"),
                limit=args.get("limit", 3),
                effectiveness_scores=effectiveness_scores,
                user_id=user_id,    # Phase 17: pass user_id for RL
            )
            return [r.model_dump() for r in recs]

        elif name == "start_exercise":
            session = self._therapy_svc.start_exercise(
                user_id=user_id,
                exercise_id=args["exercise_id"],
            )
            if session is None:
                return {"error": f"Exercise '{args['exercise_id']}' not found or could not be started."}
            return session.model_dump()

        elif name == "recommend_soundscape":
            recs = self._soundscape_svc.recommend_soundscapes(
                emotion=args.get("emotion", "neutral"),
                intent=args.get("intent", "casual_chat"),
                exercise_id=args.get("exercise_id"),
                limit=args.get("limit", 3),
            )
            return [r.model_dump() for r in recs]

        elif name == "search_memories":
            results = self._memory_svc.search_memories(
                user_id=user_id,
                query=args["query"],
                limit=args.get("limit", 5),
            )
            return results if results else {"message": "No relevant memories found."}

        elif name == "get_user_progress":
            summary = self._progress_svc.get_progress_summary(user_id)
            return summary.model_dump()

        elif name == "get_effectiveness":
            profile = self._feedback_svc.compute_effectiveness(user_id)
            return profile.model_dump()

        elif name == "get_crisis_resources":
            resources = self._safety_svc.get_crisis_resources()
            return [r.model_dump() for r in resources]

        elif name == "search_knowledge_base":
            if retrieve_fn is None:
                return {"error": "Knowledge base search not available."}
            context = retrieve_fn(
                query=args["query"],
                k=args.get("k", 5),
            )
            if context:
                return {"context": context}
            return {"context": "No relevant information found in the knowledge base."}

        elif name == "log_mood":
            mood_entry = {
                "mood": args["mood"],
                "intensity": args.get("intensity", 5),
                "context": args.get("context", ""),
                "recorded_at": datetime.now().isoformat(),
                "detected_via": "tool_call",
                "confidence": 1.0,
            }
            success = self._user_svc.append_mood_entry(user_id, mood_entry)
            return {
                "success": success,
                "mood": args["mood"],
                "intensity": args.get("intensity", 5),
                "message": "Mood logged successfully." if success else "Failed to log mood.",
            }

        elif name == "get_mood_history":
            profile_data = self._user_svc.get_user_profile(user_id)
            if not profile_data:
                return {"mood_history": [], "message": "No profile found."}
            mood_history = profile_data.get("affective", {}).get("mood_history", [])
            limit = args.get("limit", 10)
            return {"mood_history": mood_history[-limit:], "total_entries": len(mood_history)}

        elif name == "create_task":
            task = PracticeTask(
                user_id=user_id,
                source_exercise_id=args["source_exercise_id"],
                source_session_id="",
                modality="",
                title=args["title"],
                description=args.get("description", ""),
                due_date=args.get("due_date", ""),
                target_count=args.get("target_count", 1),
            )
            task_id = self._therapy_svc.create_task(user_id, task)
            if task_id:
                return {"success": True, "task_id": task_id, "title": args["title"]}
            return {"success": False, "error": "Failed to create task."}

        elif name == "get_due_tasks":
            tasks = self._therapy_svc.get_due_tasks(user_id)
            return {"tasks": tasks, "count": len(tasks)}

        # ── 4 Self-Contained Tools ────────────────────────

        elif name == "breathing_timer":
            pattern_key = args.get("pattern", "4-7-8")
            pattern = BREATHING_PATTERNS.get(pattern_key, BREATHING_PATTERNS["4-7-8"])
            return pattern

        elif name == "journal_prompt":
            emotion = args.get("emotion", "neutral")
            prompts = JOURNAL_PROMPTS.get(emotion, JOURNAL_PROMPTS["neutral"])
            selected = random.choice(prompts)
            return {"prompt": selected, "emotion": emotion}

        elif name == "sleep_hygiene_tips":
            issue = args.get("issue", "general")
            tips = SLEEP_TIPS.get(issue, SLEEP_TIPS["general"])
            return tips

        elif name == "grounding_exercise":
            intensity = args.get("intensity", "moderate")
            script = GROUNDING_SCRIPTS.get(intensity, GROUNDING_SCRIPTS["moderate"])
            return script

        # ── 2 Phase 12 Tools (Anti-Dependency & Cultural) ──

        elif name == "get_wellbeing_check":
            from dependency_service import get_dependency_service
            dep_svc = get_dependency_service()
            return dep_svc.get_dependency_insights(user_id)

        elif name == "get_international_resources":
            from cultural_service import get_cultural_service
            cultural_svc = get_cultural_service()
            country = args.get("country_code", "")
            if not country:
                # Try to infer from user profile
                profile_data = self._user_svc.get_user_profile(user_id)
                ctx = cultural_svc.extract_cultural_context(profile_data)
                country = ctx.country_code
            resources = cultural_svc.get_crisis_resources_for_country(country)
            return [r.model_dump() for r in resources]

        # ── 2 Phase 13 Tools (GDPR Compliance) ────────────

        elif name == "request_data_export":
            from compliance_service import get_compliance_service
            compliance_svc = get_compliance_service()
            export = compliance_svc.export_user_data(user_id, ip_address="tool_call")
            # Return a summary rather than the full export (which could be huge)
            return {
                "user_id": export.user_id,
                "export_timestamp": export.export_timestamp,
                "data_categories": {
                    "profile": bool(export.profile),
                    "memories_count": len(export.memories),
                    "chat_sessions_count": len(export.chat_sessions),
                    "response_feedback_count": len(export.response_feedback),
                    "exercise_outcomes_count": len(export.exercise_outcomes),
                    "exercise_sessions_count": len(export.exercise_sessions),
                    "practice_tasks_count": len(export.practice_tasks),
                    "soundscape_sessions_count": len(export.soundscape_sessions),
                    "safety_events_count": len(export.safety_events),
                    "has_interaction_metrics": bool(export.interaction_metrics),
                    "has_consent_record": bool(export.consent),
                },
                "message": (
                    "Your data export is ready. It includes the categories listed above. "
                    "To download the complete export as JSON, use the /users/{user_id}/export endpoint."
                ),
            }

        elif name == "request_data_deletion":
            confirm = args.get("confirm", False)
            if not confirm:
                return {
                    "warning": (
                        "⚠️ DATA DELETION WARNING: This will permanently delete ALL your data "
                        "including your profile, memories, chat history, exercise sessions, "
                        "feedback, soundscape sessions, and consent records. "
                        "Audit logs will be preserved as required by law. "
                        "This action CANNOT be undone. "
                        "Please confirm by saying 'Yes, delete all my data' to proceed."
                    ),
                    "confirmed": False,
                }
            from compliance_service import get_compliance_service
            compliance_svc = get_compliance_service()
            receipt = compliance_svc.delete_all_user_data(user_id, ip_address="tool_call")
            return {
                "user_id": receipt.user_id,
                "deletion_timestamp": receipt.deletion_timestamp,
                "total_documents_deleted": receipt.total_documents_deleted,
                "collections_deleted": receipt.collections_deleted,
                "confirmed": True,
                "message": (
                    "All your data has been permanently deleted. "
                    f"{receipt.total_documents_deleted} documents were removed across "
                    f"{len(receipt.collections_deleted)} collections. "
                    "Audit logs have been preserved as required by law."
                ),
            }

        # ── Phase 16: Wearable Health Tool ────────

        elif name == "get_health_summary":
            from wearable_service import get_wearable_service
            svc = get_wearable_service()
            days = args.get("days", 7)
            summary = svc.get_health_summary(user_id, days=days)
            return summary.model_dump()

        else:
            return {"error": f"Unknown tool: {name}"}


# ── Singleton ─────────────────────────────────────────────

_chat_agent_service: Optional[ChatAgentService] = None


def get_chat_agent_service() -> ChatAgentService:
    """Get or create ChatAgentService singleton."""
    global _chat_agent_service
    if _chat_agent_service is None:
        _chat_agent_service = ChatAgentService()
    return _chat_agent_service
