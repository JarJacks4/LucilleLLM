# LucilleLLM API Reference

> **Base URL (local):** `http://localhost:8080`
> **Base URL (production):** `https://lucillellm2-<hash>-ue.a.run.app`
> **Interactive docs:** `{BASE_URL}/docs` (Swagger UI) | `{BASE_URL}/redoc` (ReDoc)

---

## Table of Contents

1. [Core Chat](#1-core-chat)
2. [Session Management](#2-session-management)
3. [User Profiles & Onboarding](#3-user-profiles--onboarding)
4. [Memories](#4-memories)
5. [Therapy Exercises](#5-therapy-exercises)
6. [Practice Tasks](#6-practice-tasks)
7. [Feedback & Outcomes](#7-feedback--outcomes)
8. [Soundscapes](#8-soundscapes)
9. [Safety & Crisis](#9-safety--crisis)
10. [GDPR & Compliance](#10-gdpr--compliance)
11. [Voice I/O](#11-voice-io)
12. [Wearable Health](#12-wearable-health)
13. [Reinforcement Learning](#13-reinforcement-learning)
14. [Fine-Tuning](#14-fine-tuning)
15. [Admin Dashboard](#15-admin-dashboard)
16. [Admin Operations](#16-admin-operations)
17. [Annual Reviews](#17-annual-reviews)
18. [System Health](#18-system-health)

---

## 1. Core Chat

The main conversational endpoints. These run the full pipeline: emotion detection, safety screening, RAG retrieval, RL modality selection, prompt building, LLM response, and Firestore persistence.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Send a message and receive a complete response. Runs the full chat pipeline (emotion detection, safety check, dependency monitoring, cultural context, wearable health context, memory recall, RAG retrieval, RL modality selection, 5-layer prompt building, LangChain agent invocation, output validation, escalation check, A/B testing, Firestore persistence). Returns the full response once generation is complete. |
| `POST` | `/chat/stream` | Same full pipeline as `/chat`, but returns the response as **Server-Sent Events (SSE)** for real-time token-by-token streaming. Each SSE message contains `{content, done, session_id, response, message_count}`. Final signal: `data: [DONE]`. Designed for FlutterFlow compatibility. |
| `POST` | `/chat/voice` | Voice-enabled chat. Accepts optional base64-encoded audio input, transcribes it via STT, runs the full `/chat` pipeline internally, then optionally converts the response to speech via TTS. `response_format` controls output: `"text"`, `"audio"`, or `"both"` (default). |
| `GET` | `/chat-interface` | Returns an HTML test UI for interacting with the chatbot in a browser. Useful for development and demos. |

### Request Body (`POST /chat` and `/chat/stream`)

```json
{
  "message": "I've been feeling anxious about work lately",
  "session_id": "optional-uuid (auto-generated if omitted)",
  "user_id": "optional-user-id (enables personalization)"
}
```

### Response (`POST /chat`)

```json
{
  "response": "I hear you. Let's talk about what's causing that anxiety...",
  "session_id": "abc-123",
  "message_count": 5,
  "conversation_summary": "User discussed work-related anxiety..."
}
```

---

## 2. Session Management

Manage chat sessions stored in Firestore. Each session contains the full message history between a user and Lucille.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Generate a new unique session ID (UUID). Use this to start a fresh conversation. |
| `GET` | `/chat/{session_id}` | Retrieve the full chat history for a given session, including all messages and metadata. |
| `DELETE` | `/chat/{session_id}` | Permanently delete a chat session and all its messages from Firestore. |
| `GET` | `/sessions/` | List recent chat sessions. Accepts `?limit=100` query parameter. Returns session IDs, timestamps, and message counts. |
| `GET` | `/session/{session_id}/validate` | Check if a session ID exists and is valid. Returns session metadata if found. |

---

## 3. User Profiles & Onboarding

Manage user profiles built on a 5-layer behavioral model: Persona, Affective, Behavioral, Motivational, and Cognitive layers. These profiles drive personalized responses.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/users/onboard` | Onboard a new user. Accepts answers to onboarding questions and builds a full 5-layer profile (communication style, emotional triggers, habits, goals, beliefs). Stores in Firestore. |
| `GET` | `/users/{user_id}` | Retrieve a user's complete profile including all 5 behavioral layers. |
| `PUT` | `/users/{user_id}` | Update specific fields of a user's profile. Supports partial updates to any layer. Invalidates the profile cache. |
| `DELETE` | `/users/{user_id}` | Delete a user's profile from Firestore. |
| `POST` | `/users/{user_id}/mood` | Record a mood entry for the user. Accepts mood label, intensity (1-10), context, and detection method (manual/text_auto/image_auto). Appended to the Affective layer's mood history. |
| `GET` | `/users/{user_id}/sessions` | List all chat sessions belonging to a specific user. Accepts `?limit=50`. |

---

## 4. Memories

Lucille maintains three types of long-term memory per user: **Episodic** (personal experiences), **Semantic** (general knowledge/preferences), and **Factual** (specific facts). Memories are stored with embeddings for semantic search.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/users/{user_id}/memories` | List all memories for a user. Filterable by `?memory_type=episodic` and `?limit=50`. |
| `POST` | `/users/{user_id}/memories` | Create a new memory. Requires `content`, `memory_type` (episodic/semantic/factual), and optional `importance` (1-10). Generates an embedding for future semantic search. |
| `POST` | `/users/{user_id}/memories/search` | Semantic search across a user's memories. Accepts a `query` string, returns the most relevant memories ranked by embedding similarity. |
| `DELETE` | `/users/{user_id}/memories/{memory_id}` | Delete a specific memory by ID. |
| `POST` | `/users/{user_id}/memories/consolidate` | Trigger memory consolidation. Merges related memories, removes duplicates, and strengthens frequently accessed memories. |

---

## 5. Therapy Exercises

Lucille offers guided therapy exercises across 4 modalities: **CBT** (Cognitive Behavioral Therapy), **ACT** (Acceptance & Commitment Therapy), **DBT** (Dialectical Behavior Therapy), and **MI** (Motivational Interviewing). Each exercise has multiple steps.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/therapy/exercises` | List all available exercise templates. Filter by `?modality=cbt` (cbt/act/dbt/mi). Returns exercise ID, name, description, modality, steps, and estimated duration. |
| `GET` | `/therapy/exercises/{exercise_id}` | Get detailed information about a specific exercise template including all steps and instructions. |
| `GET` | `/therapy/recommend/{user_id}` | Get personalized exercise recommendations for a user based on their current emotional state, profile, and history. Uses RL (Thompson Sampling) when enabled. Accepts `?limit=3`. |
| `POST` | `/therapy/{user_id}/start` | Start an exercise session. Requires `exercise_id` in the body. Creates a session in Firestore, optionally auto-starts a matched soundscape. Returns session ID and first step. |
| `POST` | `/therapy/{user_id}/advance/{session_id}` | Advance to the next step in an active exercise session. Accepts optional `note` for the user's response to the current step. Returns the next step or marks the session complete. |
| `POST` | `/therapy/{user_id}/abandon/{session_id}` | Abandon (quit) an in-progress exercise session. Marks it as abandoned in Firestore. |
| `GET` | `/therapy/{user_id}/active` | Get the user's currently active (in-progress) exercise session, if any. Returns full session state including current step. |
| `GET` | `/therapy/{user_id}/history` | List the user's completed and abandoned exercise sessions. Accepts `?limit=20`. |

---

## 6. Practice Tasks

Between-session practice tasks assigned to users to reinforce therapy concepts. Tasks have due dates and completion tracking.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/therapy/{user_id}/tasks` | List all practice tasks for a user. Filter by `?status=pending` (pending/completed/skipped) and `?limit=20`. |
| `GET` | `/therapy/{user_id}/tasks/due` | Get tasks that are currently due (past their scheduled date and not yet completed). |
| `POST` | `/therapy/{user_id}/tasks` | Create a new practice task. Requires `title`, `description`, and optional `due_date`. |
| `PUT` | `/therapy/{user_id}/tasks/{task_id}` | Update a task (mark complete, change status, add notes). |
| `GET` | `/therapy/{user_id}/progress` | Get a comprehensive progress summary: exercises completed, tasks done, streaks, modality breakdown, and overall engagement score. |

---

## 7. Feedback & Outcomes

Collect user feedback on individual responses and exercise outcomes. This data drives the RL system and effectiveness tracking.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/feedback/{user_id}/response` | Submit feedback on a specific chat response. Accepts `session_id`, `message_index`, `helpfulness` (1-5), and optional `comment`. Updates the RL bandit state for the modality used. |
| `POST` | `/feedback/{user_id}/exercise-outcome` | Submit an outcome rating for a completed exercise. Accepts `exercise_session_id`, `effectiveness` (1-5), `mood_before`, `mood_after`, and optional `notes`. Feeds into modality effectiveness profiles. |
| `GET` | `/feedback/{user_id}/history` | List the user's feedback history (both response and exercise feedback). Accepts `?limit=20`. |
| `GET` | `/feedback/{user_id}/effectiveness` | Get the user's effectiveness profile: per-modality average scores, best-performing modality, total feedback count. Uses cached results (120s TTL). |

---

## 8. Soundscapes

Ambient audio sessions (rain, ocean, forest, etc.) that can play during therapy exercises or independently for relaxation. Audio files are stored in Google Cloud Storage with signed URLs.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/soundscapes` | List all available soundscape templates. Filter by `?category=nature` (nature/ambient/music/white_noise). |
| `GET` | `/soundscapes/categories` | List all soundscape categories with descriptions and counts. |
| `GET` | `/soundscapes/recommend/{user_id}` | Get personalized soundscape recommendations based on the user's current emotion and active exercise. Accepts optional `?emotion=anxious` and `?exercise_id=...`. |
| `GET` | `/soundscapes/{soundscape_id}` | Get details for a specific soundscape (name, description, duration, category). |
| `POST` | `/soundscapes/{user_id}/start` | Start a soundscape listening session. Requires `soundscape_id`. Records start time in Firestore. |
| `POST` | `/soundscapes/{user_id}/stop/{session_id}` | Stop an active soundscape session. Records duration and end time. |
| `GET` | `/soundscapes/{soundscape_id}/audio` | Get a time-limited signed URL to stream the audio file from GCS. URL expires after the configured period (default: 60 minutes). |
| `GET` | `/soundscapes/{user_id}/history` | List the user's soundscape listening history. Accepts `?limit=20`. |

---

## 9. Safety & Crisis

Safety systems for crisis detection, jailbreak prevention, and escalation. These run automatically during `/chat` but can also be triggered manually.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/safety/resources` | List crisis helplines and mental health resources (organized by country/region). Always accessible, no auth required. |
| `GET` | `/safety/{user_id}/audit` | Get the safety event audit log for a user. Shows all flagged events (crisis keywords detected, jailbreak attempts, high-risk outputs). Accepts `?limit=50`. |
| `POST` | `/safety/check` | Manually run a safety check on arbitrary text. Returns risk level (CRITICAL/HIGH/MEDIUM/LOW), detected concerns, and whether crisis resources should be shown. Useful for testing. |

---

## 10. GDPR & Compliance

Data privacy endpoints for GDPR compliance, consent management, data retention, and HIPAA audit logging.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/users/{user_id}/export` | **GDPR Data Portability.** Export all of a user's data (profile, chat sessions, memories, feedback, exercise history, health metrics, consent records) in a standardized JSON format. |
| `DELETE` | `/users/{user_id}/data` | **GDPR Right to Erasure.** Cascade-delete ALL user data across all 22 Firestore collections. Irreversible. Logs the deletion in the audit trail. |
| `POST` | `/users/{user_id}/consent` | Record a new consent entry. Accepts consent `type` (data_processing, analytics, health_data, etc.) and `granted` (true/false). |
| `GET` | `/users/{user_id}/consent` | Get current consent status for a user across all consent types. |
| `PUT` | `/users/{user_id}/consent` | Update an existing consent entry (e.g., withdraw consent for analytics). |
| `POST` | `/admin/retention/enforce` | **Admin.** Manually trigger the data retention enforcement job. Deletes records older than configured retention periods across all collections. |
| `GET` | `/admin/retention/policies` | **Admin.** View the current data retention policies (days until deletion per collection type). |
| `GET` | `/admin/audit-log` | **Admin.** View the HIPAA-compliant audit trail. Logs all data access, modifications, and deletions. 7-year retention. Accepts `?limit=100` and `?user_id=...` filters. |

---

## 11. Voice I/O

Text-to-Speech and Speech-to-Text endpoints. TTS uses `edge-tts` (free, no API key) by default with multiple voice options.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/tts` | Convert text to speech. Accepts `text` and optional `voice` (default: `en-US-AriaNeural`), `rate` (e.g., `+10%`). Returns base64-encoded audio. |
| `POST` | `/stt` | Convert speech to text. Accepts base64-encoded audio input. Returns transcribed text and confidence score. |

---

## 12. Wearable Health

Sync and analyze health data from wearable devices (sleep patterns, heart rate, activity levels). This context is injected into the chat pipeline for health-aware responses.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/wearables/{user_id}/sync` | Sync health data from a wearable device. Accepts sleep records (duration, quality, stages), activity records (steps, calories, active minutes), and heart rate data. Stores daily metrics in Firestore. |
| `GET` | `/wearables/{user_id}/metrics` | Get raw daily health metrics for a user. Accepts `?days=7` to control the lookback window. |
| `GET` | `/wearables/{user_id}/summary` | Get an aggregated health summary with averages, trends, and insights across sleep, activity, and heart rate. |
| `GET` | `/wearables/{user_id}/sleep-insights` | Get detailed sleep analysis: average duration, quality trends, sleep debt, and personalized recommendations. |

---

## 13. Reinforcement Learning

Lucille uses Thompson Sampling (a multi-armed bandit algorithm) to learn which therapy modality (CBT/ACT/DBT/MI) works best for each user based on their feedback.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/rl/{user_id}/bandit-state` | Get the current Thompson Sampling state for a user. Shows alpha/beta parameters per modality, success rates, exploration bonus, and which modality would be selected next. Useful for debugging and transparency. |

---

## 14. Fine-Tuning

Pipeline for extracting training data from high-quality conversations, submitting fine-tuning jobs to OpenAI, and A/B testing fine-tuned models against the base model.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/finetuning/status` | Get the current fine-tuning system status: whether it's enabled, active model, base model, A/B split percentage, and total training examples collected. |
| `POST` | `/finetuning/extract-training-data` | Extract high-quality conversation pairs from Firestore to build training datasets. Filters by minimum helpfulness score and formats into OpenAI fine-tuning JSONL format. |
| `POST` | `/finetuning/submit-job` | Submit a fine-tuning job to the OpenAI API. Uses extracted training data. Returns the OpenAI job ID for tracking. |
| `GET` | `/finetuning/jobs` | List all fine-tuning jobs (pending, running, completed, failed). |
| `GET` | `/finetuning/jobs/{job_id}` | Get detailed status of a specific fine-tuning job including progress, metrics, and result model ID. |
| `GET` | `/finetuning/ab-stats` | Get A/B testing statistics comparing the fine-tuned model vs. base model. Shows response counts, average helpfulness scores, and statistical significance. |

---

## 15. Admin Dashboard

Monitoring dashboard with real-time metrics. The HTML dashboard provides charts; JSON endpoints provide raw data for custom dashboards.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/admin/dashboard` | **HTML page.** Full admin dashboard with interactive charts showing system health, user engagement, therapy effectiveness, safety events, model performance, and RL metrics. Auto-refreshes every 30 seconds. |
| `GET` | `/admin/dashboard/system` | **JSON.** System metrics: request latency (p50/p95/p99), error rates, cache hit ratios, active sessions, memory usage. |
| `GET` | `/admin/dashboard/engagement` | **JSON.** User engagement metrics: daily/weekly/monthly active users, average session length, messages per session, retention rates. |
| `GET` | `/admin/dashboard/therapy` | **JSON.** Therapy metrics: exercises started/completed/abandoned, completion rates by modality, average effectiveness scores. |
| `GET` | `/admin/dashboard/safety` | **JSON.** Safety metrics: crisis events detected, jailbreak attempts blocked, escalation tickets created, risk level distribution. |
| `GET` | `/admin/dashboard/models` | **JSON.** Model performance: average response latency, token usage, cost tracking, A/B test results. |
| `GET` | `/admin/dashboard/rl` | **JSON.** RL metrics: modality selection distribution, exploration vs. exploitation ratio, average reward per modality. |
| `GET` | `/admin/dashboard/all` | **JSON.** Combined response with ALL metric categories in a single request. |

---

## 16. Admin Operations

Escalation queue management for cases that need human therapist review.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/admin/escalations` | List all escalation tickets. These are auto-created when a user triggers 3+ safety events within 7 days. Filterable by status. |
| `GET` | `/admin/escalations/stats` | Get queue statistics: total open tickets, average time to resolution, tickets by priority. |
| `GET` | `/admin/escalations/{escalation_id}` | Get full details of an escalation ticket including the triggering events, user context, and resolution history. |
| `PUT` | `/admin/escalations/{escalation_id}` | Update an escalation ticket (change status to reviewed/resolved, add notes, assign to a team member). |
| `GET` | `/admin/audio-status` | Check the status of audio files in Google Cloud Storage (which soundscapes have audio uploaded, file sizes, missing files). |

---

## 17. Annual Reviews

LLM-powered periodic review generation that summarizes a user's therapeutic journey over a time period.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reviews/{user_id}/generate` | Generate an annual/periodic review for a user. The LLM analyzes their chat history, exercise completion, mood trends, and progress to produce a comprehensive summary with insights and recommendations. Accepts optional `period_days` (default: 365). |
| `GET` | `/reviews/{user_id}` | List all generated reviews for a user. |
| `GET` | `/reviews/{user_id}/{review_id}` | Get a specific review by ID. |

---

## 18. System Health

Health check and monitoring endpoints for infrastructure monitoring and load balancers.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Basic health check. Returns `{"status": "healthy"}` with 200 OK. Used by Cloud Run health probes and load balancers. |
| `GET` | `/health/detailed` | Detailed health check. Verifies connectivity to Firestore, OpenAI API, FAISS index, and GCS. Returns status per dependency. |
| `GET` | `/metrics` | Prometheus-style metrics: request counts, latency histograms, error rates, cache stats, active connections. |

---

## Authentication (Not Yet Implemented)

> **Important:** All endpoints are currently **unauthenticated**. Before production deployment, add authentication middleware. Recommended approach:
>
> - **Firebase Authentication** for user-facing endpoints
> - **API key or JWT** for admin endpoints (`/admin/*`)
> - **Service account tokens** for internal/microservice calls

---

## Common Request Headers

| Header | Value | When Required |
|--------|-------|---------------|
| `Content-Type` | `application/json` | All POST/PUT requests |
| `Accept` | `text/event-stream` | `/chat/stream` (SSE) |
| `Authorization` | `Bearer <token>` | Future: when auth is added |

---

## Error Responses

All endpoints return errors in this format:

```json
{
  "detail": "Human-readable error message"
}
```

| Status Code | Meaning |
|-------------|---------|
| `400` | Bad request (invalid input, missing fields) |
| `404` | Resource not found (session, user, exercise, etc.) |
| `429` | Rate limited (10 req/min for /chat, 100 req/min global) |
| `500` | Internal server error (OpenAI API failure, Firestore error, etc.) |

---

## Rate Limits

| Scope | Limit |
|-------|-------|
| `/chat` and `/chat/stream` | 10 requests/minute per IP |
| All other endpoints | 100 requests/minute global |

---

## Architecture Summary

```
Client Request
    |
    v
[Middleware: Rate Limit -> Metrics -> Privacy]
    |
    v
[main.py - 92 endpoints]
    |
    +---> firebase_service.py    (Firestore: 22 collections)
    +---> emotion_service.py     (OpenAI emotion/intent detection)
    +---> safety_service.py      (Crisis detection, jailbreak blocking)
    +---> memory_service.py      (Episodic/semantic/factual recall)
    +---> prompt_engine.py       (5-layer system prompt builder)
    +---> chat_agent_service.py  (LangChain agent with tools)
    +---> therapy_service.py     (CBT/ACT/DBT/MI exercises)
    +---> progress_service.py    (Tasks, progress tracking)
    +---> feedback_service.py    (Ratings, effectiveness)
    +---> soundscape_service.py  (Audio sessions, GCS)
    +---> rl_service.py          (Thompson Sampling optimization)
    +---> voice_service.py       (TTS/STT)
    +---> wearable_service.py    (Health data integration)
    +---> cultural_service.py    (Country-aware adaptation)
    +---> dependency_service.py  (Usage pattern monitoring)
    +---> compliance_service.py  (GDPR export/deletion)
    +---> audit_service.py       (HIPAA audit trail)
    +---> escalation_service.py  (Auto-escalation queue)
    +---> finetuning_service.py  (Training data, A/B testing)
    +---> monitoring_service.py  (Dashboard metrics)
    +---> storage_service.py     (GCS signed URLs)
    +---> cache.py               (In-memory TTL cache)
    +---> config.py              (68 config fields)
    +---> models.py              (70+ Pydantic models)
    +---> middleware.py          (Rate limiting, metrics, privacy)
```
