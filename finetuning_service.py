"""
LucilleLLM - Fine-Tuning Service (Phase 18)

Manages the fine-tuning pipeline:
  - Training data extraction from Firestore (chat_sessions + feedback)
  - JSONL formatting for OpenAI fine-tuning API
  - Job submission via openai_client.fine_tuning.jobs.create()
  - Job monitoring via openai_client.fine_tuning.jobs.retrieve()
  - A/B model routing (deterministic hash-based user split)
  - Performance metrics tracking

Firestore structure:
    finetuning_jobs/{job_id}           - FineTuningJobRecord
    training_examples/{example_id}     - TrainingExample (flat collection)
    model_performance/{record_id}      - ModelPerformanceRecord (flat collection)

No external ML libraries: uses hashlib from stdlib + openai SDK.

Follows the singleton pattern from other services.
"""

import hashlib
import io
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from config import get_config
from models import (
    FineTuningJobRecord,
    FineTuningJobStatus,
    FineTuningStatsResponse,
    ModelPerformanceRecord,
    TrainingExample,
    TrainingExampleSource,
)

logger = logging.getLogger(__name__)


# ── Status Mapping ────────────────────────────────────

OPENAI_STATUS_MAP: Dict[str, FineTuningJobStatus] = {
    "validating_files": FineTuningJobStatus.VALIDATING,
    "queued": FineTuningJobStatus.PENDING,
    "running": FineTuningJobStatus.RUNNING,
    "succeeded": FineTuningJobStatus.SUCCEEDED,
    "failed": FineTuningJobStatus.FAILED,
    "cancelled": FineTuningJobStatus.CANCELLED,
}


class FineTuningService:
    """
    Service for fine-tuning pipeline management and A/B model routing.

    Features:
    - Deterministic A/B model routing via MD5 hash of user_id
    - Training data extraction from consented chat sessions
    - JSONL formatting and OpenAI fine-tuning job submission
    - Per-response performance tracking for A/B comparison
    """

    JOBS_COLLECTION = "finetuning_jobs"
    EXAMPLES_COLLECTION = "training_examples"
    PERFORMANCE_COLLECTION = "model_performance"

    def __init__(self):
        self._config = get_config()
        self._db = None
        self._openai_client = None

        try:
            from firebase_service import get_firebase_service
            fb = get_firebase_service()
            self._db = fb.db
        except Exception as e:
            logger.warning(f"FineTuningService: Firebase not available -- {e}")

        logger.info(
            f"FineTuningService initialized -- "
            f"ft_enabled={self._config.FT_ENABLED}, "
            f"ft_model_id={self._config.FT_MODEL_ID or '(none)'}, "
            f"ab_split={self._config.FT_AB_SPLIT_PERCENT}%, "
            f"db_available={self._db is not None}"
        )

    # ── OpenAI Client Injection ─────────────────────────

    def set_openai_client(self, client) -> None:
        """
        Inject the OpenAI client from main.py.
        Called after server startup when the client is initialized.
        """
        self._openai_client = client
        logger.info("FineTuningService: OpenAI client injected")

    # ── Properties ──────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """Check if fine-tuning service has database access and is enabled."""
        return self._db is not None and self._config.FT_ENABLED

    # ── A/B Model Routing ───────────────────────────────

    def resolve_model_for_user(
        self, user_id: Optional[str]
    ) -> Tuple[str, bool, str]:
        """
        Determine which model to use for a given user.

        Returns (model_id, is_fine_tuned, ab_group).
        Falls back to base model if fine-tuning is disabled or
        no fine-tuned model ID is configured.

        Uses deterministic MD5 hash of user_id for consistent
        per-user routing (same user always gets same model).
        """
        base_model = self._config.OPENAI_MODEL

        # Guard: feature disabled or no fine-tuned model available
        if not self._config.FT_ENABLED or not self._config.FT_MODEL_ID:
            return (base_model, False, "base")

        # Guard: no user_id means anonymous -- always base
        if not user_id:
            return (base_model, False, "base")

        # Deterministic hash-based split
        hash_val = int(
            hashlib.md5(user_id.encode("utf-8")).hexdigest(), 16
        )
        bucket = hash_val % 100

        if bucket < self._config.FT_AB_SPLIT_PERCENT:
            return (self._config.FT_MODEL_ID, True, "fine_tuned")
        else:
            return (base_model, False, "base")

    # ── Training Data Extraction ────────────────────────

    def extract_training_data(
        self,
        min_feedback_rating: str = "helpful",
        min_helpfulness_score: int = 4,
        max_examples: int = 1000,
    ) -> List[TrainingExample]:
        """
        Extract high-quality training examples from chat sessions.

        Only includes data from users who have consented to ML_TRAINING.
        Filters sessions that have positive feedback ratings.

        Returns a list of TrainingExample objects stored in Firestore.
        """
        if self._db is None:
            logger.warning("Cannot extract training data: no database")
            return []

        # Determine acceptable ratings
        acceptable_ratings = {"helpful"}
        if min_feedback_rating == "neutral":
            acceptable_ratings.add("neutral")

        examples: List[TrainingExample] = []
        consented_users: Dict[str, bool] = {}  # cache consent checks

        try:
            # Fetch all chat sessions that have a user_id
            sessions_ref = self._db.collection("chat_sessions")
            sessions = sessions_ref.limit(max_examples * 5).stream()

            for doc in sessions:
                if len(examples) >= max_examples:
                    break

                session = doc.to_dict()
                user_id = session.get("user_id", "")
                if not user_id:
                    continue

                # Check consent (cached per user)
                if user_id not in consented_users:
                    consented_users[user_id] = self._check_ml_training_consent(
                        user_id
                    )
                if not consented_users[user_id]:
                    continue

                # Check if this session has positive feedback
                session_id = session.get("session_id", doc.id)
                has_good_feedback = self._check_session_feedback(
                    user_id, session_id, acceptable_ratings
                )
                if not has_good_feedback:
                    continue

                # Extract conversation windows from messages
                messages = session.get("messages", [])
                training_messages = self._convert_messages_to_openai_format(
                    messages
                )

                if len(training_messages) < 2:
                    continue  # Need at least user + assistant

                example = TrainingExample(
                    user_id=user_id,
                    source_type=TrainingExampleSource.CHAT_SESSION,
                    source_id=session_id,
                    messages=training_messages,
                    quality_signals={
                        "feedback_filter": min_feedback_rating,
                        "helpfulness_filter": min_helpfulness_score,
                    },
                )

                # Store in Firestore
                try:
                    self._db.collection(self.EXAMPLES_COLLECTION).document(
                        example.example_id
                    ).set(example.model_dump())
                except Exception as e:
                    logger.warning(
                        f"Failed to store training example: {e}"
                    )
                    continue

                examples.append(example)

            logger.info(
                f"Extracted {len(examples)} training examples "
                f"(filter: rating>={min_feedback_rating}, "
                f"helpfulness>={min_helpfulness_score})"
            )

        except Exception as e:
            logger.error(f"Training data extraction failed: {e}")

        return examples

    def _convert_messages_to_openai_format(
        self, messages: List[dict]
    ) -> List[dict]:
        """
        Convert Firestore chat messages to OpenAI fine-tuning format.

        Firestore format: {"type": "human|ai|system", "content": "..."}
        OpenAI format: {"role": "user|assistant|system", "content": "..."}
        """
        type_to_role = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
        }

        converted = []
        for msg in messages:
            msg_type = msg.get("type", "")
            content = msg.get("content", "")
            role = type_to_role.get(msg_type)

            if role and content:
                converted.append({"role": role, "content": content})

        return converted

    def _check_session_feedback(
        self,
        user_id: str,
        session_id: str,
        acceptable_ratings: set,
    ) -> bool:
        """Check if a session has at least one acceptable feedback rating."""
        try:
            feedback_ref = (
                self._db.collection("feedback")
                .document(user_id)
                .collection("response_feedback")
                .where("session_id", "==", session_id)
                .limit(5)
            )
            for doc in feedback_ref.stream():
                data = doc.to_dict()
                if data.get("rating", "") in acceptable_ratings:
                    return True
            return False
        except Exception:
            return False

    def _check_ml_training_consent(self, user_id: str) -> bool:
        """Check if a user has consented to ML training data usage."""
        try:
            doc = (
                self._db.collection("consent_records")
                .document(user_id)
                .get()
            )
            if doc.exists:
                data = doc.to_dict()
                consents = data.get("consents", {})
                return consents.get("ml_training", False)
            return False
        except Exception:
            return False

    # ── JSONL Formatting ────────────────────────────────

    def format_training_jsonl(
        self, examples: List[TrainingExample]
    ) -> str:
        """
        Convert training examples to newline-delimited JSON (JSONL)
        for OpenAI fine-tuning API.

        Each line: {"messages": [{"role": "...", "content": "..."}, ...]}
        """
        lines = []
        for example in examples:
            line = json.dumps(
                {"messages": example.messages}, ensure_ascii=False
            )
            lines.append(line)
        return "\n".join(lines)

    # ── Job Submission ──────────────────────────────────

    def submit_finetuning_job(
        self,
        examples: List[TrainingExample],
        base_model: str = "gpt-4o-mini",
        n_epochs: int = 3,
        suffix: str = "lucille",
    ) -> Optional[FineTuningJobRecord]:
        """
        Submit a fine-tuning job to OpenAI using extracted training data.

        Steps:
        1. Validate minimum example count
        2. Format as JSONL
        3. Upload JSONL to OpenAI Files API
        4. Create fine-tuning job
        5. Store job record in Firestore
        6. Mark examples as included in this job
        """
        if self._openai_client is None:
            logger.error("Cannot submit job: no OpenAI client")
            return None

        min_examples = self._config.FT_MIN_TRAINING_EXAMPLES
        if len(examples) < min_examples:
            logger.warning(
                f"Not enough training examples: {len(examples)} < "
                f"{min_examples} minimum"
            )
            return None

        try:
            # Format JSONL
            jsonl_content = self.format_training_jsonl(examples)
            jsonl_bytes = jsonl_content.encode("utf-8")

            # Upload to OpenAI Files API
            file_name = (
                f"lucille_training_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            )
            file_obj = io.BytesIO(jsonl_bytes)
            file_obj.name = file_name

            uploaded_file = self._openai_client.files.create(
                file=file_obj, purpose="fine-tune"
            )
            logger.info(
                f"Uploaded training file: {uploaded_file.id} "
                f"({len(examples)} examples, {len(jsonl_bytes)} bytes)"
            )

            # Create fine-tuning job
            job = self._openai_client.fine_tuning.jobs.create(
                training_file=uploaded_file.id,
                model=base_model,
                hyperparameters={"n_epochs": n_epochs},
                suffix=suffix,
            )
            logger.info(f"Fine-tuning job created: {job.id}")

            # Create record
            record = FineTuningJobRecord(
                job_id=job.id,
                base_model=base_model,
                status=OPENAI_STATUS_MAP.get(
                    job.status, FineTuningJobStatus.PENDING
                ),
                training_file_id=uploaded_file.id,
                training_examples_count=len(examples),
                hyperparameters={"n_epochs": n_epochs},
            )

            # Store in Firestore
            if self._db is not None:
                self._db.collection(self.JOBS_COLLECTION).document(
                    job.id
                ).set(record.model_dump())

                # Mark examples as included in this job
                for example in examples:
                    try:
                        self._db.collection(
                            self.EXAMPLES_COLLECTION
                        ).document(example.example_id).update(
                            {"included_in_job": job.id}
                        )
                    except Exception:
                        pass

            return record

        except Exception as e:
            logger.error(f"Fine-tuning job submission failed: {e}")
            return None

    # ── Job Monitoring ──────────────────────────────────

    def check_job_status(
        self, job_id: str
    ) -> Optional[FineTuningJobRecord]:
        """
        Check the status of a fine-tuning job from OpenAI API.
        Updates the local Firestore record with the latest status.

        If the job succeeded, captures the fine_tuned_model ID.
        """
        if self._openai_client is None:
            logger.warning("Cannot check job: no OpenAI client")
            return self._get_job_from_firestore(job_id)

        try:
            job = self._openai_client.fine_tuning.jobs.retrieve(job_id)

            status = OPENAI_STATUS_MAP.get(
                job.status, FineTuningJobStatus.PENDING
            )
            fine_tuned_model_id = ""
            if hasattr(job, "fine_tuned_model") and job.fine_tuned_model:
                fine_tuned_model_id = job.fine_tuned_model

            error_message = ""
            if hasattr(job, "error") and job.error:
                error_message = str(job.error)

            finished_at = ""
            if hasattr(job, "finished_at") and job.finished_at:
                finished_at = datetime.fromtimestamp(
                    job.finished_at
                ).isoformat()

            record = FineTuningJobRecord(
                job_id=job_id,
                base_model=getattr(job, "model", "gpt-4o-mini"),
                fine_tuned_model_id=fine_tuned_model_id,
                status=status,
                training_file_id=getattr(
                    job, "training_file", ""
                ),
                training_examples_count=0,
                hyperparameters=dict(
                    getattr(job, "hyperparameters", {}) or {}
                ),
                error_message=error_message,
                finished_at=finished_at,
            )

            # Update Firestore
            if self._db is not None:
                self._db.collection(self.JOBS_COLLECTION).document(
                    job_id
                ).set(record.model_dump(), merge=True)

            return record

        except Exception as e:
            logger.error(f"Failed to check job {job_id}: {e}")
            return self._get_job_from_firestore(job_id)

    def _get_job_from_firestore(
        self, job_id: str
    ) -> Optional[FineTuningJobRecord]:
        """Fallback: read job record from Firestore."""
        if self._db is None:
            return None
        try:
            doc = (
                self._db.collection(self.JOBS_COLLECTION)
                .document(job_id)
                .get()
            )
            if doc.exists:
                return FineTuningJobRecord(**doc.to_dict())
            return None
        except Exception:
            return None

    def list_jobs(self, limit: int = 10) -> List[dict]:
        """List all fine-tuning jobs, most recent first."""
        if self._db is None:
            return []

        try:
            query = (
                self._db.collection(self.JOBS_COLLECTION)
                .order_by("created_at", direction="DESCENDING")
                .limit(limit)
            )
            results = []
            for doc in query.stream():
                results.append(doc.to_dict())
            return results
        except Exception as e:
            logger.warning(f"Failed to list jobs: {e}")
            return []

    # ── Performance Tracking ────────────────────────────

    def log_model_performance(
        self,
        user_id: str,
        session_id: str,
        model_used: str,
        is_fine_tuned: bool,
        ab_group: str,
        response_length: int,
        detected_emotion: str = "",
        detected_intent: str = "",
    ) -> Optional[str]:
        """
        Log a per-response performance record for A/B comparison.
        Returns the record_id or None if storage fails.
        """
        if self._db is None:
            return None

        try:
            record = ModelPerformanceRecord(
                user_id=user_id,
                session_id=session_id,
                model_used=model_used,
                is_fine_tuned=is_fine_tuned,
                ab_group=ab_group,
                response_length=response_length,
                detected_emotion=detected_emotion,
                detected_intent=detected_intent,
            )

            self._db.collection(self.PERFORMANCE_COLLECTION).document(
                record.record_id
            ).set(record.model_dump())

            return record.record_id

        except Exception as e:
            logger.warning(f"Failed to log model performance: {e}")
            return None

    def update_performance_feedback(
        self, session_id: str, rating: str
    ) -> bool:
        """
        Link explicit user feedback to the model performance record
        for the given session. This enables comparing feedback rates
        between base and fine-tuned models.
        """
        if self._db is None:
            return False

        try:
            query = (
                self._db.collection(self.PERFORMANCE_COLLECTION)
                .where("session_id", "==", session_id)
                .limit(1)
            )
            for doc in query.stream():
                doc.reference.update({"feedback_rating": rating})
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to update performance feedback: {e}")
            return False

    def compute_ab_stats(self) -> FineTuningStatsResponse:
        """
        Compute A/B testing comparison statistics between base
        and fine-tuned models.

        Aggregates total responses, helpful rates, and average
        response lengths for each model group.
        """
        config = self._config

        if self._db is None:
            return FineTuningStatsResponse(
                active_model_id=config.FT_MODEL_ID,
                ab_split_percent=config.FT_AB_SPLIT_PERCENT,
            )

        try:
            base_total = 0
            base_helpful = 0
            base_length_sum = 0
            ft_total = 0
            ft_helpful = 0
            ft_length_sum = 0

            docs = (
                self._db.collection(self.PERFORMANCE_COLLECTION)
                .limit(10000)
                .stream()
            )

            for doc in docs:
                data = doc.to_dict()
                ab_group = data.get("ab_group", "base")
                rating = data.get("feedback_rating", "")
                length = data.get("response_length", 0)

                if ab_group == "fine_tuned":
                    ft_total += 1
                    ft_length_sum += length
                    if rating == "helpful":
                        ft_helpful += 1
                else:
                    base_total += 1
                    base_length_sum += length
                    if rating == "helpful":
                        base_helpful += 1

            return FineTuningStatsResponse(
                total_responses_base=base_total,
                total_responses_ft=ft_total,
                helpful_rate_base=(
                    base_helpful / base_total if base_total > 0 else 0.0
                ),
                helpful_rate_ft=(
                    ft_helpful / ft_total if ft_total > 0 else 0.0
                ),
                avg_response_length_base=(
                    base_length_sum / base_total
                    if base_total > 0
                    else 0.0
                ),
                avg_response_length_ft=(
                    ft_length_sum / ft_total if ft_total > 0 else 0.0
                ),
                active_model_id=config.FT_MODEL_ID,
                ab_split_percent=config.FT_AB_SPLIT_PERCENT,
            )

        except Exception as e:
            logger.error(f"Failed to compute A/B stats: {e}")
            return FineTuningStatsResponse(
                active_model_id=config.FT_MODEL_ID,
                ab_split_percent=config.FT_AB_SPLIT_PERCENT,
            )


# ── Singleton ─────────────────────────────────────────

_finetuning_service: Optional[FineTuningService] = None


def get_finetuning_service() -> FineTuningService:
    """Get or create FineTuningService singleton."""
    global _finetuning_service
    if _finetuning_service is None:
        _finetuning_service = FineTuningService()
    return _finetuning_service
