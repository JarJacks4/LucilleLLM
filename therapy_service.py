"""
LucilleLLM - Therapy Module Service

Provides structured CBT, ACT, DBT, and MI exercise templates.
Features: exercise registry, emotion-based recommendations,
active exercise session tracking, and prompt formatting.

Follows the singleton pattern from other services.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from firebase_service import get_firebase_service
from models import (
    ExerciseRecommendation,
    ExerciseSession,
    ExerciseTemplate,
    PracticeTask,
    TherapyModality,
)

logger = logging.getLogger(__name__)


# ── Exercise Template Registry ────────────────────────────

EXERCISE_REGISTRY: Dict[str, ExerciseTemplate] = {}


def _register(ex: ExerciseTemplate) -> None:
    EXERCISE_REGISTRY[ex.exercise_id] = ex


# ── CBT Exercises ─────────────────────────────────────────

_register(ExerciseTemplate(
    exercise_id="cbt_thought_record",
    modality=TherapyModality.CBT,
    title="Thought Record",
    description="Identify and challenge negative automatic thoughts by examining evidence for and against them.",
    target_emotions=["sad", "anxious", "hopeless", "angry"],
    target_intents=["seeking_advice", "doing_exercise", "reflecting"],
    difficulty="beginner",
    duration_minutes=10,
    steps=[
        "Let's start a Thought Record. What situation is on your mind right now? Describe it briefly.",
        "What automatic thought came to mind in that situation? Try to capture the exact words.",
        "How does that thought make you feel? Rate the intensity of each emotion from 0-100.",
        "Now let's examine the evidence. What facts SUPPORT this thought?",
        "What facts go AGAINST this thought? Think of times the opposite was true.",
        "Based on the evidence, can you write a more balanced, realistic thought?",
        "Re-rate your emotions now. Has anything shifted, even slightly?",
    ],
    system_prompt_addon=(
        "You are guiding the user through a CBT Thought Record exercise. "
        "Be Socratic: ask questions rather than giving answers. Help them discover "
        "the balanced thought themselves. Validate their feelings throughout. "
        "Do NOT rush through steps."
    ),
    practice_prompt="Practice writing a Thought Record daily when you notice a negative automatic thought. Focus on finding balanced alternatives.",
    practice_target_count=3,
    practice_days=3,
))

_register(ExerciseTemplate(
    exercise_id="cbt_cognitive_distortions",
    modality=TherapyModality.CBT,
    title="Cognitive Distortion Spotter",
    description="Learn to identify common thinking traps like catastrophizing, black-and-white thinking, and mind reading.",
    target_emotions=["anxious", "sad", "hopeless", "overwhelmed"],
    target_intents=["seeking_advice", "doing_exercise", "reflecting"],
    difficulty="beginner",
    duration_minutes=8,
    steps=[
        "Think of a stressful thought you've had recently. Share it with me.",
        "Let me help you spot any thinking traps. I'll describe common ones, and you tell me which fits.",
        "Common distortions: **Catastrophizing** (worst-case), **All-or-Nothing** (black/white), **Mind Reading** (assuming others' thoughts), **Fortune Telling** (predicting the worst), **Overgeneralization** (always/never). Which one fits your thought?",
        "Now that we've identified the distortion, let's reframe. What would a friend say about this situation?",
        "Write down the reframed thought. How does it feel compared to the original?",
    ],
    system_prompt_addon=(
        "You are guiding the user through identifying cognitive distortions. "
        "Present distortions clearly with examples. Be encouraging when they "
        "identify one. Help them reframe without dismissing the original feeling."
    ),
    practice_prompt="When you catch a stressful thought, identify the distortion type and try to reframe it.",
    practice_target_count=3,
    practice_days=3,
))

_register(ExerciseTemplate(
    exercise_id="cbt_behavioral_activation",
    modality=TherapyModality.CBT,
    title="Behavioral Activation Planner",
    description="Plan small, achievable activities to counteract low mood and withdrawal patterns.",
    target_emotions=["sad", "hopeless", "lonely"],
    target_intents=["seeking_advice", "doing_exercise"],
    difficulty="beginner",
    duration_minutes=10,
    steps=[
        "When we feel low, we often withdraw from activities. What activities used to bring you joy or a sense of accomplishment?",
        "From that list, pick ONE small activity you could do today or tomorrow. It can be very small.",
        "Let's make it specific. When will you do it? Where? For how long?",
        "What might get in the way? Let's plan for that obstacle.",
        "After you try it, come back and tell me how it went. Even showing up counts as a win.",
    ],
    system_prompt_addon=(
        "You are guiding Behavioral Activation. The key is starting very small. "
        "Validate that low motivation is part of depression, not laziness. "
        "Celebrate any movement toward action."
    ),
    practice_prompt="Do your planned activity and note how it made you feel afterward.",
    practice_target_count=3,
    practice_days=5,
))

# ── ACT Exercises ─────────────────────────────────────────

_register(ExerciseTemplate(
    exercise_id="act_defusion",
    modality=TherapyModality.ACT,
    title="Thought Defusion",
    description="Practice creating distance from unhelpful thoughts by observing them without attachment.",
    target_emotions=["anxious", "overwhelmed", "sad", "hopeless"],
    target_intents=["doing_exercise", "reflecting"],
    difficulty="beginner",
    duration_minutes=8,
    steps=[
        "Name a thought that keeps bothering you. Start with 'I am having the thought that...'",
        "Now say it again, but this time: 'I notice I am having the thought that...' How does adding 'I notice' change things?",
        "Try one more: imagine placing that thought on a leaf floating down a stream. Watch it drift away. Describe what you see.",
        "The thought may come back, and that's okay. The goal isn't to eliminate it, but to hold it more lightly. How are you feeling now?",
    ],
    system_prompt_addon=(
        "You are guiding an ACT defusion exercise. Emphasize that thoughts are "
        "just mental events, not facts. Use metaphors (leaves on a stream, clouds "
        "passing). Never argue with the thought's content."
    ),
    practice_prompt="Practice 'I notice I am having the thought that...' when an unhelpful thought appears.",
    practice_target_count=5,
    practice_days=5,
))

_register(ExerciseTemplate(
    exercise_id="act_values_compass",
    modality=TherapyModality.ACT,
    title="Values Compass",
    description="Clarify your core values and identify small steps to move toward what matters most.",
    target_emotions=["sad", "hopeless", "lonely", "neutral"],
    target_intents=["seeking_advice", "reflecting", "doing_exercise"],
    difficulty="intermediate",
    duration_minutes=15,
    steps=[
        "Let's explore what matters most to you. If you could wave a magic wand and live fully, what areas of life would you focus on? (e.g., relationships, work, health, creativity, community)",
        "Pick the area that feels most important right now. What does living according to that value look like in daily life?",
        "On a scale of 1-10, how closely are you living in alignment with that value right now?",
        "What is ONE small action you could take this week to move closer to that value? Think tiny.",
        "What would it mean to you to take that step? Let's connect it back to your deeper why.",
    ],
    system_prompt_addon=(
        "You are guiding an ACT Values Compass exercise. Help the user connect "
        "to intrinsic values (not goals). Values are directions, not destinations. "
        "Be curious and exploratory."
    ),
    practice_prompt="Take one small values-aligned action this week and reflect on how it felt.",
    practice_target_count=1,
    practice_days=7,
))

# ── DBT Exercises ─────────────────────────────────────────

_register(ExerciseTemplate(
    exercise_id="dbt_distress_tolerance",
    modality=TherapyModality.DBT,
    title="TIPP Distress Tolerance",
    description="Use the TIPP technique (Temperature, Intense exercise, Paced breathing, Paired muscle relaxation) to manage intense emotions.",
    target_emotions=["angry", "anxious", "overwhelmed", "fearful"],
    target_intents=["doing_exercise", "seeking_advice"],
    difficulty="beginner",
    duration_minutes=10,
    steps=[
        "You're feeling intense emotions right now. Let's use the TIPP technique to bring your body's alarm system down. Ready?",
        "**T - Temperature**: Hold something cold (ice cube, cold water on your face) for 30 seconds. This triggers the dive reflex and calms your nervous system. Try it now.",
        "**I - Intense Exercise**: Do 2 minutes of intense movement (jumping jacks, running in place, push-ups). This burns off stress hormones.",
        "**P - Paced Breathing**: Breathe in for 4 counts, out for 6 counts. The longer exhale activates your calm-down system. Let's do 5 rounds together.",
        "**P - Paired Muscle Relaxation**: Tense your fists tightly for 5 seconds, then release. Move to shoulders, then legs. Feel the contrast.",
        "Check in: on a 0-10 scale, where is your distress now compared to when we started?",
    ],
    system_prompt_addon=(
        "You are guiding a DBT TIPP exercise for distress tolerance. "
        "Be calm and direct. Guide each step with timing cues. "
        "Validate that the distress is real while helping them regulate."
    ),
    practice_prompt="Next time you feel intense distress, use one TIPP technique and rate your distress before and after.",
    practice_target_count=2,
    practice_days=3,
))

_register(ExerciseTemplate(
    exercise_id="dbt_opposite_action",
    modality=TherapyModality.DBT,
    title="Opposite Action",
    description="When emotions push you toward unhelpful behaviors, practice doing the opposite to change the emotional experience.",
    target_emotions=["angry", "sad", "fearful", "anxious"],
    target_intents=["seeking_advice", "doing_exercise"],
    difficulty="intermediate",
    duration_minutes=10,
    steps=[
        "What emotion are you experiencing strongly right now? And what is it urging you to do?",
        "Is acting on that urge helpful right now, or would it make things worse in the long run?",
        "If it's not helpful, let's identify the OPPOSITE action. For example: anger urges attack -> opposite is gentle withdrawal or kindness. What's the opposite of your urge?",
        "Try the opposite action, even if it feels unnatural. You don't have to feel it, just do it.",
        "Check in after trying: has the intensity of the emotion shifted at all?",
    ],
    system_prompt_addon=(
        "You are guiding a DBT Opposite Action exercise. Help the user identify "
        "the action urge linked to their emotion, then coach them to try the "
        "opposite. Emphasize that feelings change when behavior changes."
    ),
    practice_prompt="When an emotion urges unhelpful behavior, try the opposite action and note what happens.",
    practice_target_count=3,
    practice_days=5,
))

_register(ExerciseTemplate(
    exercise_id="dbt_mindfulness_observe",
    modality=TherapyModality.DBT,
    title="Mindful Observation",
    description="Practice the DBT mindfulness skill of observing your experience without judgment.",
    target_emotions=["anxious", "overwhelmed", "neutral", "sad"],
    target_intents=["doing_exercise", "reflecting"],
    difficulty="beginner",
    duration_minutes=5,
    steps=[
        "Let's practice mindful observation. Find a comfortable position and take a slow breath.",
        "Notice 3 things you can SEE right now. Describe them without judgment, just facts.",
        "Notice 2 things you can HEAR. Just observe the sounds without labeling them as good or bad.",
        "Notice 1 thing you can FEEL physically (texture, temperature, pressure).",
        "Now turn inward: what emotion are you noticing? Just name it, like 'there is anxiety' or 'there is calm'. No need to change it.",
    ],
    system_prompt_addon=(
        "You are guiding a DBT mindfulness observation exercise. "
        "Use a calm, measured pace. Encourage non-judgmental noticing. "
        "If the user judges, gently redirect: 'Just notice, no need to label good or bad.'"
    ),
    practice_prompt="Do a 2-minute mindful observation check-in: notice 3 things you see, 2 you hear, 1 you feel.",
    practice_target_count=5,
    practice_days=5,
))

# ── MI Exercises ──────────────────────────────────────────

_register(ExerciseTemplate(
    exercise_id="mi_decisional_balance",
    modality=TherapyModality.MI,
    title="Decisional Balance",
    description="Explore the pros and cons of change vs. staying the same to build motivation.",
    target_emotions=["neutral", "sad", "hopeless"],
    target_intents=["seeking_advice", "reflecting", "doing_exercise"],
    difficulty="beginner",
    duration_minutes=12,
    steps=[
        "Let's explore a change you've been thinking about. What is it?",
        "First, what are the BENEFITS of keeping things the way they are? (There are always some.)",
        "Now, what are the COSTS of keeping things the same? What's the downside?",
        "What would be the BENEFITS of making the change?",
        "And the COSTS or challenges of making the change?",
        "Looking at all four quadrants, what stands out to you? Where does the balance tip?",
    ],
    system_prompt_addon=(
        "You are facilitating a Motivational Interviewing decisional balance exercise. "
        "Stay neutral and curious. Do NOT push toward change. Reflect back what the "
        "user says. Use open-ended questions. Honor their autonomy."
    ),
    practice_prompt="Revisit your decisional balance sheet and add any new pros/cons you have thought of.",
    practice_target_count=1,
    practice_days=7,
))

_register(ExerciseTemplate(
    exercise_id="mi_scaling_question",
    modality=TherapyModality.MI,
    title="Motivation Scaling",
    description="Use scaling questions to explore readiness, confidence, and importance of change.",
    target_emotions=["neutral", "sad", "hopeless", "anxious"],
    target_intents=["seeking_advice", "reflecting"],
    difficulty="beginner",
    duration_minutes=8,
    steps=[
        "Think of a change you'd like to make. On a scale of 1-10, how IMPORTANT is this change to you?",
        "Why did you pick that number and not a lower one? (This highlights your existing motivation.)",
        "On a scale of 1-10, how CONFIDENT are you that you could make this change if you decided to?",
        "What would help move your confidence up by just 1 point?",
        "What would be one small first step you could take?",
    ],
    system_prompt_addon=(
        "You are facilitating MI scaling questions. The key insight: asking 'why not lower?' "
        "elicits change talk. Never ask 'why not higher?' as that elicits sustain talk. "
        "Reflect and affirm every response."
    ),
    practice_prompt="Re-rate your motivation and confidence on the change you discussed. What has moved?",
    practice_target_count=1,
    practice_days=7,
))


class TherapyService:
    """
    Service for therapy exercise management.

    Features:
    - Exercise recommendation based on emotion + intent
    - Active exercise session tracking in Firestore
    - Exercise context formatting for prompt injection
    """

    COLLECTION = "exercise_sessions"

    def __init__(self):
        self._firebase = get_firebase_service()

    @property
    def db(self):
        return self._firebase.db

    # ── Exercise Discovery ───────────────────────────────

    def list_exercises(
        self,
        modality: Optional[str] = None,
    ) -> List[dict]:
        """List all available exercises, optionally filtered by modality."""
        results = []
        for ex in EXERCISE_REGISTRY.values():
            if modality and ex.modality.value != modality:
                continue
            results.append({
                "exercise_id": ex.exercise_id,
                "modality": ex.modality.value,
                "title": ex.title,
                "description": ex.description,
                "difficulty": ex.difficulty,
                "duration_minutes": ex.duration_minutes,
                "step_count": len(ex.steps),
            })
        return results

    def get_exercise(self, exercise_id: str) -> Optional[ExerciseTemplate]:
        """Get a specific exercise template by ID."""
        return EXERCISE_REGISTRY.get(exercise_id)

    def recommend_exercises(
        self,
        emotion: str = "neutral",
        intent: str = "casual_chat",
        limit: int = 3,
        effectiveness_scores: Optional[dict] = None,
        user_id: Optional[str] = None,
    ) -> List[ExerciseRecommendation]:
        """
        Recommend exercises based on current emotion, intent, and
        either RL-based Thompson Sampling (Phase 17) or rule-based
        effectiveness adjustments (Phase 7 fallback).
        """
        from config import get_config
        config = get_config()

        # Determine if we should use RL (Thompson Sampling)
        use_rl = False
        thompson_scores: Dict[str, float] = {}
        emotion_group = ""
        cached_arms: Dict[str, dict] = {}

        if config.RL_ENABLED and user_id:
            try:
                from rl_service import get_rl_service, _arm_key
                rl_svc = get_rl_service()
                emotion_group = rl_svc.get_emotion_group(emotion)

                # Sample Thompson scores for all exercises
                all_exercise_ids = list(EXERCISE_REGISTRY.keys())
                thompson_scores = rl_svc.sample_thompson_scores(
                    user_id=user_id,
                    emotion=emotion,
                    exercise_ids=all_exercise_ids,
                )
                # Cache arm states to avoid re-reading in the loop
                cached_arms = rl_svc.load_bandit_state(user_id)
                use_rl = True
                logger.info(
                    f"RL: Using Thompson Sampling for user {user_id}, "
                    f"emotion_group={emotion_group}"
                )
            except Exception as e:
                logger.warning(f"RL sampling failed, falling back: {e}")
                use_rl = False

        scored = []
        for ex in EXERCISE_REGISTRY.values():
            score = 0.0
            reason_parts = []

            # Base scoring: emotion + intent match (unchanged)
            if emotion in ex.target_emotions:
                score += 3
                reason_parts.append(f"helps with {emotion}")

            if intent in ex.target_intents:
                score += 2
                reason_parts.append(f"suits {intent.replace('_', ' ')} intent")

            # Phase 17: Thompson Sampling bonus (replaces Phase 7)
            rl_meta = None
            if use_rl and thompson_scores:
                from rl_service import _arm_key
                from models import RLRecommendationMeta

                ts_score = thompson_scores.get(ex.exercise_id, 0.5)
                bonus = ts_score * config.RL_EXPLORATION_BONUS
                score += bonus

                key = _arm_key(emotion_group, ex.exercise_id)
                arm = cached_arms.get(key, {"alpha": 1.0, "beta": 1.0})

                rl_meta = RLRecommendationMeta(
                    rl_used=True,
                    emotion_group=emotion_group,
                    thompson_score=round(ts_score, 4),
                    arm_alpha=arm.get("alpha", 1.0),
                    arm_beta=arm.get("beta", 1.0),
                )

                if ts_score > 0.7:
                    reason_parts.append(
                        "personalized recommendation based on your history"
                    )

            # Phase 7 fallback: effectiveness_scores adjustments
            elif effectiveness_scores and not use_rl:
                modality_scores = effectiveness_scores.get(
                    "modality_scores", {}
                )
                exercise_scores_dict = effectiveness_scores.get(
                    "exercise_scores", {}
                )

                modality_score = modality_scores.get(ex.modality.value, 0)
                exercise_score = exercise_scores_dict.get(ex.exercise_id, 0)

                if modality_score >= 4.0:
                    score += 2
                    reason_parts.append(
                        f"{ex.modality.value.upper()} has been very "
                        f"effective for you"
                    )
                elif modality_score >= 3.0:
                    score += 1
                    reason_parts.append(
                        f"{ex.modality.value.upper()} has been effective "
                        f"for you"
                    )

                if exercise_score >= 4.0:
                    score += 1
                    reason_parts.append(
                        "you've rated this exercise highly"
                    )

                if 0 < modality_score < 2.0:
                    score -= 1
                    reason_parts.append(
                        f"{ex.modality.value.upper()} has been less "
                        f"effective for you"
                    )

            if score <= 0:
                continue

            reason = "Recommended because it " + " and ".join(reason_parts) + "."
            scored.append((score, ExerciseRecommendation(
                exercise_id=ex.exercise_id,
                modality=ex.modality.value,
                title=ex.title,
                description=ex.description,
                reason=reason,
                difficulty=ex.difficulty,
                duration_minutes=ex.duration_minutes,
                rl_meta=rl_meta,
            )))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:limit]]

    # ── Exercise Session Management ──────────────────────

    def start_exercise(
        self,
        user_id: str,
        exercise_id: str,
    ) -> Optional[ExerciseSession]:
        """Start an exercise session and return the session object."""
        exercise = self.get_exercise(exercise_id)
        if exercise is None:
            return None

        session = ExerciseSession(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            exercise_id=exercise_id,
            modality=exercise.modality.value,
            title=exercise.title,
            current_step=0,
            total_steps=len(exercise.steps),
            status="active",
        )

        # Store in Firestore
        if self.db is not None:
            try:
                doc_ref = (
                    self.db.collection(self.COLLECTION)
                    .document(user_id)
                    .collection("sessions")
                    .document(session.session_id)
                )
                doc_ref.set(session.model_dump())
                logger.info(
                    f"Started exercise '{exercise.title}' for user {user_id}"
                )
            except Exception as e:
                logger.warning(f"Failed to store exercise session: {e}")

        return session

    def get_active_exercise(
        self, user_id: str
    ) -> Optional[ExerciseSession]:
        """Get the user's currently active exercise session, if any."""
        if self.db is None:
            return None

        try:
            col_ref = (
                self.db.collection(self.COLLECTION)
                .document(user_id)
                .collection("sessions")
            )
            query = (
                col_ref.where("status", "==", "active")
                .order_by("started_at", direction="DESCENDING")
                .limit(1)
            )

            for doc in query.stream():
                data = doc.to_dict()
                return ExerciseSession(**data)

            return None

        except Exception as e:
            logger.warning(f"Failed to get active exercise for {user_id}: {e}")
            return None

    def advance_exercise(
        self,
        user_id: str,
        session_id: str,
        user_note: str = "",
    ) -> Optional[ExerciseSession]:
        """
        Advance to the next step of an exercise.
        Returns the updated session, or None if not found.
        """
        if self.db is None:
            return None

        try:
            doc_ref = (
                self.db.collection(self.COLLECTION)
                .document(user_id)
                .collection("sessions")
                .document(session_id)
            )
            doc = doc_ref.get()
            if not doc.exists:
                return None

            data = doc.to_dict()
            session = ExerciseSession(**data)

            if session.status != "active":
                return session

            # Record user note for this step
            if user_note:
                session.notes.append(user_note)

            session.current_step += 1

            # Check if exercise is complete
            if session.current_step >= session.total_steps:
                session.status = "completed"
                session.completed_at = datetime.now().isoformat()

            doc_ref.update({
                "current_step": session.current_step,
                "status": session.status,
                "completed_at": session.completed_at,
                "notes": session.notes,
            })

            # Auto-assign practice task when exercise completes
            if session.status == "completed":
                self._auto_assign_task(user_id, session)

            return session

        except Exception as e:
            logger.warning(f"Failed to advance exercise: {e}")
            return None

    def abandon_exercise(
        self,
        user_id: str,
        session_id: str,
    ) -> bool:
        """Mark an exercise session as abandoned."""
        if self.db is None:
            return False

        try:
            doc_ref = (
                self.db.collection(self.COLLECTION)
                .document(user_id)
                .collection("sessions")
                .document(session_id)
            )
            doc_ref.update({
                "status": "abandoned",
                "completed_at": datetime.now().isoformat(),
            })
            return True

        except Exception as e:
            logger.warning(f"Failed to abandon exercise: {e}")
            return False

    def get_exercise_history(
        self,
        user_id: str,
        limit: int = 20,
    ) -> List[dict]:
        """Get a user's past exercise sessions."""
        if self.db is None:
            return []

        try:
            col_ref = (
                self.db.collection(self.COLLECTION)
                .document(user_id)
                .collection("sessions")
            )
            query = col_ref.order_by(
                "started_at", direction="DESCENDING"
            ).limit(limit)

            results = []
            for doc in query.stream():
                data = doc.to_dict()
                results.append({
                    "session_id": data.get("session_id"),
                    "exercise_id": data.get("exercise_id"),
                    "modality": data.get("modality"),
                    "title": data.get("title"),
                    "status": data.get("status"),
                    "current_step": data.get("current_step"),
                    "total_steps": data.get("total_steps"),
                    "started_at": data.get("started_at"),
                    "completed_at": data.get("completed_at"),
                })
            return results

        except Exception as e:
            logger.warning(f"Failed to get exercise history: {e}")
            return []

    # ── Task / Homework Management ─────────────────────────

    def create_task(self, user_id: str, task: PracticeTask) -> Optional[str]:
        """Store a practice task in Firestore."""
        if self.db is None:
            return None
        try:
            doc_ref = (
                self.db.collection(self.COLLECTION)
                .document(user_id)
                .collection("tasks")
                .document(task.task_id)
            )
            doc_ref.set(task.model_dump())
            logger.info(f"Created task '{task.title}' for user {user_id}")
            return task.task_id
        except Exception as e:
            logger.warning(f"Failed to create task for {user_id}: {e}")
            return None

    def get_tasks(
        self,
        user_id: str,
        status_filter: Optional[str] = None,
        limit: int = 20,
    ) -> List[dict]:
        """Get tasks for a user, optionally filtered by status."""
        if self.db is None:
            return []
        try:
            col_ref = (
                self.db.collection(self.COLLECTION)
                .document(user_id)
                .collection("tasks")
            )
            if status_filter:
                query = col_ref.where("status", "==", status_filter)
            else:
                query = col_ref

            query = query.order_by(
                "assigned_at", direction="DESCENDING"
            ).limit(limit)

            results = []
            for doc in query.stream():
                results.append(doc.to_dict())
            return results
        except Exception as e:
            logger.warning(f"Failed to get tasks for {user_id}: {e}")
            return []

    def get_due_tasks(self, user_id: str) -> List[dict]:
        """Get pending/in_progress tasks that are due or overdue."""
        if self.db is None:
            return []
        try:
            col_ref = (
                self.db.collection(self.COLLECTION)
                .document(user_id)
                .collection("tasks")
            )
            now_iso = datetime.now().isoformat()
            tomorrow_iso = (
                datetime.now() + timedelta(hours=24)
            ).isoformat()

            due_tasks = []
            for status_val in ("pending", "in_progress"):
                query = col_ref.where("status", "==", status_val)
                for doc in query.stream():
                    data = doc.to_dict()
                    due_date = data.get("due_date", "")
                    if not due_date:
                        continue
                    if due_date <= now_iso:
                        data["is_overdue"] = True
                        due_tasks.append(data)
                    elif due_date <= tomorrow_iso:
                        data["is_overdue"] = False
                        due_tasks.append(data)

            return due_tasks
        except Exception as e:
            logger.warning(f"Failed to get due tasks for {user_id}: {e}")
            return []

    def update_task(
        self,
        user_id: str,
        task_id: str,
        updates: dict,
    ) -> Optional[dict]:
        """Update a task (status, completed_count, notes)."""
        if self.db is None:
            return None
        try:
            doc_ref = (
                self.db.collection(self.COLLECTION)
                .document(user_id)
                .collection("tasks")
                .document(task_id)
            )
            doc = doc_ref.get()
            if not doc.exists:
                return None

            data = doc.to_dict()

            if "status" in updates:
                data["status"] = updates["status"]
                if updates["status"] == "completed":
                    data["completed_at"] = datetime.now().isoformat()

            if "completed_count" in updates:
                data["completed_count"] = updates["completed_count"]
                if data["completed_count"] >= data.get("target_count", 1):
                    data["status"] = "completed"
                    data["completed_at"] = datetime.now().isoformat()

            if "note" in updates and updates["note"]:
                notes = data.get("notes", [])
                notes.append(updates["note"])
                data["notes"] = notes

            doc_ref.update(data)
            return data
        except Exception as e:
            logger.warning(f"Failed to update task {task_id}: {e}")
            return None

    def format_due_tasks_for_prompt(self, due_tasks: List[dict]) -> str:
        """Format due/overdue tasks as text for system prompt injection."""
        if not due_tasks:
            return ""

        parts = []
        overdue = [t for t in due_tasks if t.get("is_overdue")]
        upcoming = [t for t in due_tasks if not t.get("is_overdue")]

        if overdue:
            parts.append("OVERDUE practice tasks:")
            for t in overdue:
                progress = f"{t.get('completed_count', 0)}/{t.get('target_count', 1)}"
                parts.append(f"  - {t['title']} (progress: {progress})")

        if upcoming:
            parts.append("Upcoming practice tasks (due soon):")
            for t in upcoming:
                progress = f"{t.get('completed_count', 0)}/{t.get('target_count', 1)}"
                parts.append(
                    f"  - {t['title']} (due: {t.get('due_date', 'unknown')}, "
                    f"progress: {progress})"
                )

        parts.append(
            "Mention these tasks naturally in conversation. "
            "For overdue tasks, gently check in. For upcoming tasks, encourage progress. "
            "Do NOT lecture or pressure the user."
        )

        return (
            "--- PRACTICE TASKS ---\n"
            + "\n".join(parts)
            + "\n--- END PRACTICE TASKS ---"
        )

    def _auto_assign_task(
        self, user_id: str, session: ExerciseSession
    ) -> Optional[str]:
        """Auto-create a practice task when an exercise is completed."""
        try:
            template = self.get_exercise(session.exercise_id)
            if template is None or not template.practice_prompt:
                return None

            due = (
                datetime.now() + timedelta(days=template.practice_days)
            ).isoformat()

            task = PracticeTask(
                user_id=user_id,
                source_exercise_id=session.exercise_id,
                source_session_id=session.session_id,
                modality=session.modality,
                title=f"Practice: {template.title}",
                description=template.practice_prompt,
                due_date=due,
                target_count=template.practice_target_count,
            )

            return self.create_task(user_id, task)
        except Exception as e:
            logger.warning(f"Auto-task assignment failed for {user_id}: {e}")
            return None

    # ── Prompt Formatting ────────────────────────────────

    def format_exercise_for_prompt(
        self,
        exercise_session: ExerciseSession,
    ) -> str:
        """
        Format active exercise context for system prompt injection.
        Returns the exercise instructions + current step guidance.
        """
        exercise = self.get_exercise(exercise_session.exercise_id)
        if exercise is None:
            return ""

        step_idx = exercise_session.current_step
        if step_idx >= len(exercise.steps):
            return (
                f"--- ACTIVE EXERCISE: {exercise.title} ({exercise.modality.value.upper()}) ---\n"
                f"The user has completed all steps. Congratulate them and "
                f"ask how they feel after the exercise.\n"
                f"--- END EXERCISE ---"
            )

        current_step_text = exercise.steps[step_idx]
        progress = f"Step {step_idx + 1} of {len(exercise.steps)}"

        parts = [
            f"--- ACTIVE EXERCISE: {exercise.title} ({exercise.modality.value.upper()}) ---",
            exercise.system_prompt_addon,
            f"Progress: {progress}",
            f"Current step to guide the user through:\n{current_step_text}",
            "--- END EXERCISE ---",
        ]

        return "\n".join(parts)


# ── Singleton ─────────────────────────────────────────────

_therapy_service: Optional[TherapyService] = None


def get_therapy_service() -> TherapyService:
    """Get or create TherapyService singleton."""
    global _therapy_service
    if _therapy_service is None:
        _therapy_service = TherapyService()
    return _therapy_service
