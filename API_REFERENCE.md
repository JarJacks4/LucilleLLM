# LucilleLLM API Reference

> **Base URL (local):** `http://localhost:8080`
> **Base URL (production):** `https://lucille-861854898360.us-central1.run.app` (Cloud Run service `lucille`, us-central1)
> **Interactive docs:** `{BASE_URL}/docs` (Swagger UI) | `{BASE_URL}/redoc` (ReDoc)
> **Escape app endpoints (v1):** sections 20–27. OpenAPI for FlutterFlow import: [`docs/escape_v1_openapi.json`](docs/escape_v1_openapi.json). Costs and config: [`docs/API_V1.md`](docs/API_V1.md).
>
> _Updated October 2026: added the `/v1` Escape API, documented assessments, the self-care score and image mood analysis, and corrected the production URL and authentication sections. No existing endpoint was removed._

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
19. [Assessments & Wellness Scores](#19-assessments--wellness-scores)

**Escape v1 API** (`/v1`)

20. [v1 Basics: auth, time zone, consent](#20-v1-basics-auth-time-zone-consent)
21. [Me, Home & Suggestions](#21-me-home--suggestions)
22. [Mood Scan & Mood Stats](#22-mood-scan--mood-stats)
23. [Journal](#23-journal)
24. [Soundscapes AI](#24-soundscapes-ai)
25. [Self-Care Score & Plans](#25-self-care-score--plans)
26. [Privacy (GDPR) v1](#26-privacy-gdpr-v1)
27. [Config, Jobs & Webhooks](#27-config-jobs--webhooks)
28. [Render Service (lucille-render)](#28-render-service-lucille-render)

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
| `POST` | `/users/{user_id}/mood/analyze-image` | **Requires the user's own token.** Upload a face image (multipart `file`); OpenAI Vision detects the dominant emotion and, unless `?store=false`, records it as a mood entry with `detected_via=image_auto`. Replaces the legacy ViT classifier. |
| `GET` | `/users/{user_id}/selfcare-score` | **Requires the user's own token.** Legacy self-care engagement score (0–100) from mood stability, exercise engagement and effectiveness, task completion and consistency. Not a clinical measure. The Escape app's new score is `GET /v1/selfcare/score` ([section 25](#25-self-care-score--plans)). |
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

> These legacy routes stay because the current app and the iOS `LucilleSoundscapesClient` call them. The Soundscapes AI catalog (9 categories, modes, Mood Field, Compose) is in [section 24](#24-soundscapes-ai).

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
| `GET` | `/users/{user_id}/export` | **Requires the user's own Firebase token (or admin).** **GDPR Data Portability.** Export all of a user's data (profile, chat sessions, memories, feedback, exercise history, health metrics, consent records) in a standardized JSON format. |
| `DELETE` | `/users/{user_id}/data` | **Requires the user's own Firebase token (or admin).** **GDPR Right to Erasure.** Cascade-delete ALL user data across all 22 Firestore collections. Irreversible. Logs the deletion in the audit trail. |
| `POST` | `/users/{user_id}/consent` | **Requires the user's own token** (as do the GET and PUT below; these GDPR routes were unauthenticated before October 2026). Record a new consent entry. Accepts consent `type` (data_processing, analytics, health_data, etc.) and `granted` (true/false). |
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

## 19. Assessments & Wellness Scores

Validated instruments (PHQ-9, GAD-7, WHO-5). Scores come only from the instruments' published scoring.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/assessments/instruments` | List available instruments (PHQ-9, GAD-7, WHO-5) with item counts and score ranges. |
| `GET` | `/assessments/instruments/{assessment_type}` | One instrument's questions and answer scale. |
| `POST` | `/assessments/{user_id}/start` | Start an assessment session for the user. |
| `POST` | `/assessments/{user_id}/answer/{session_id}` | Submit one answer; returns the next question. |
| `POST` | `/assessments/{user_id}/complete/{session_id}` | Finish and score the assessment. Item 9 of the PHQ-9 triggers crisis resources. |
| `GET` | `/assessments/{user_id}/history` | Past assessment results. |
| `GET` | `/assessments/{user_id}/latest` | Most recent score for each instrument. |
| `GET` | `/assessments/{user_id}/wellness-score` | Composite wellness score: WHO-5 (0–100) with PHQ-9 and GAD-7 breakdowns. |

---

## 20. v1 Basics: auth, time zone, consent

The `/v1` endpoints power the reworked Escape app: Journal V2, Mood Scan and Mood Stats, Soundscapes AI, the Self-Care Score and plans, and the in-app privacy controls. They live in `escape_api/` and are mounted by `register(app)` in `main.py`.

**Every v1 call**

| Header | Value | Notes |
|--------|-------|-------|
| `Authorization` | `Bearer <Firebase ID token>` | Required except where a row below says **public**. FlutterFlow: `currentJwtToken`. iOS: `Auth.auth().currentUser?.getIDToken()`. |
| `X-Timezone` | `America/New_York` or `-04:00` | IANA name or UTC offset. Sets "today", streak days and night mode. Falls back to the user's saved time zone, then UTC. |
| `Content-Type` | `application/json` | POST / PUT / PATCH. |

**Rules**

* The user is always taken from the token. v1 paths never contain a user id, so one user cannot reach another's data.
* Mood and journal data are health data. Storing them needs the `wellbeing_data` consent; AI reflection needs `ai_reflection`; camera pulse needs `camera_scan`; voice transcripts need `voice`; weather needs `location_weather`. A missing consent returns:

```json
{ "detail": { "code": "consent_required", "purpose": "wellbeing_data",
              "label": "Store my mood check-ins and journal entries",
              "action": "PUT /v1/privacy/consents" } }
```

* Free text (journal entries, mood notes, compose prompts) runs through the existing keyword safety screen. On a crisis match the response includes `safety.crisis: true` and 988 resources, no AI call is made, and the event is logged to `safety_audit/{uid}/events`.
* AI-written fields carry `aiGenerated: true`; responses that include Lucille's writing include `disclosure`.
* Lists page with `?limit=` and `?cursor=` (pass back `nextCursor`).

**Data location:** all v1 data sits under `escape_users/{uid}` in Firestore (`state`, `consents`, `journal_entries`, `journal_letters`, `mood_checkins`, `mood_days`, `mood_months`, `listening_sessions`, `library`, `compositions`, `plans`, `scores`, `weekly`, `usage`). Clients may read their own tree; only the API writes.

---

## 21. Me, Home & Suggestions

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/me` | Streak, coins, lifetime coins, time zone, name, prefs, age band and energy level. |
| `PATCH` | `/v1/me` | Update `timezone`, `displayName`, `wakeTime` (`HH:MM`), `birthYear` (stored only as an age band + `isMinor`), `blend`, `lucilleWhisper`, `defaultBrainwave`. |
| `GET` | `/v1/me/progress` | One streak across mind, body, journal and sound (one grace day per 7 days), recent active days, coins and the coin rules with daily caps. |
| `GET` | `/v1/energy/centers` | The 7 energy centers, Grounding → Creativity → Power → Connection → Expression → Intuition → Purpose. The Energy Scan unlocks at 1,000 lifetime coins; each further 1,000 opens the next center. |
| `GET` | `/v1/home` | Home in one call: Mood Orb summary, Self-Care Score, streak, coins, energy, today's plan steps, one suggestion, AI disclosure. Query: `period=week\|month\|year`, optional `lat`, `lon`. |
| `GET` | `/v1/suggestions` | Lucille Suggestions. Query: `context=home\|mood_result\|journal_saved\|session_complete\|explore`, optional `word`, `limit` (1–6). Every item's `deeplink` has been checked against `escape_api/data/deeplinks.json`. |

**Coins** (server-side only, per local day): mood check-in 5 (×3), journal entry 15 (×3), reflection 5 (×3), gratitude 10 (×1), letter 10 (×1), 1 per 5 listening minutes (×24), plan step 10 (×6), energy scan 10 (×1).

---

## 22. Mood Scan & Mood Stats

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/mood/words` | **Public.** 36 mood words with Mood Field coordinates (valence, energy), families and orb tones. Cache 24 h. |
| `POST` | `/v1/mood/interpret` | Preview a reading from scan inputs without saving. No AI call. |
| `POST` | `/v1/mood/checkins` | Save a Mood Scan or check-in. Needs `wellbeing_data` (+ `camera_scan` when a scan sends pulse). |
| `GET` | `/v1/mood/checkins` | History, newest first. Query: `limit`, `cursor`, `day_from`, `day_to` (`YYYY-MM-DD`). |
| `GET` | `/v1/mood/latest` | The latest reading plus its orb (seeds "Save this moment"). |
| `DELETE` | `/v1/mood/checkins/{checkin_id}` | Delete one check-in; the rollups adjust. |
| `GET` | `/v1/mood/summary` | The single conglomerate Mood Orb. Query: `period=week\|month\|year`, optional `lat`, `lon`. |
| `GET` | `/v1/mood/stats` | Mood Stats page: the chosen period in full plus the other two in brief. |
| `GET` | `/v1/mood/suggestions` | Things to do in the app for a mood (`word`, default = latest). |
| `GET` | `/v1/mood/orb` | Which orb loop and background wash to show now (`tone`, optional `lat`, `lon`). |

**`POST /v1/mood/checkins`**: send any mix of the three inputs. They are blended with weights tap 0.5, voice 0.3 and pulse 0.2 (pulse affects energy only).

```json
{
  "source": "scan",
  "word": "Restless",
  "valence": 0.3, "energy": 0.75,
  "text": "too many tabs open in my head",
  "pulse": { "bpm": 88, "hrv": 32 },
  "tags": ["work"], "note": "optional"
}
```

`source` is one of `scan`, `tap`, `voice`, `text`, `journal`, `session_before`, `session_after`, `energy_scan`, `checkin`, `healthkit`.

Response (201):

```json
{
  "checkin": { "id": "mood_…", "word": "Restless", "family": "restless", "tone": "restless",
               "valence": 0.31, "energy": 0.77, "inputs": ["word","tap","voice","pulse"], "localDay": "2026-10-09" },
  "lucille": { "word": "Restless", "line": "Restless. Your system is running fast. We can slow it down together.", "aiGenerated": false },
  "orb": { "tone": "restless", "accent": "#EF7702", "orbHevc": "…/journal/journal_orb_restless_hevc.mp4",
           "orbH264": "…_h264.mp4", "orbPoster": "…_poster.webp", "bgHevc": "…/journal_bg_warmwash_hevc.mp4", "night": false },
  "suggestions": [ { "activityId": "breath_box", "title": "Box Breathing", "minutes": 4,
                     "deeplink": { "route": "BoxBreathingGoalPage", "path": "/boxBreathingGoalPage", "url": "escape://boxBreathingGoalPage" } } ],
  "journalSeed": { "checkinId": "mood_…", "word": "Restless", "deeplink": { "route": "HealthJournal", "…": "…" } },
  "safety": { "crisis": false, "riskLevel": "low" },
  "streakDays": 3, "coinsAwarded": 5
}
```

**`GET /v1/mood/summary?period=week`**: `word`, `tone`, `size` (80–124 px; grows with how many check-ins there are), `big` (stats-page size), `caption`, `sub`, `entries`, `activeDays`, `pleasantPct`, `mix` (top 4 words with `pct` and colours), `trend` (7 days, 15 two-day buckets for month, or 12 months for year; each `{label, h 0–100, empty, c}`) and `orb`. Orb tones: `bright`, `calm`, `tender`, `restless`, `heavy`, `mixed`.

---

## 23. Journal

Four modes: `free` (Free write), `guided` (Lucille guided), `gratitude`, `ritual` (Ritual Spark).

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/journal/home` | Mood Orb for `period`, the last scan if it is under 12 h old (with its orb), the weekly reflection if one exists, the 4 modes, streak, coins, night flag. |
| `GET` | `/v1/journal/modes` | **Public.** The 4 modes with label, subtitle, icon key and colour. |
| `POST` | `/v1/journal/prompt` | A prompt for `mode` + `word`. Comes from the prompt dataset at no cost. `personalize: true` in guided mode lets Lucille tailor it (counts toward `QUOTA_GUIDED_PROMPT_PER_DAY`, cached per mood + themes). |
| `POST` | `/v1/journal/entries` | Create an entry (`status: draft\|saved`). Needs `wellbeing_data` (+ `voice` with a transcript). Saving awards coins, extends the streak and, if the entry has no scan seed, adds a `journal` check-in so it feeds the Mood Orb. |
| `GET` | `/v1/journal/entries` | List with previews. Query: `limit`, `cursor`, `mode`, `theme`, `status=saved\|draft\|all`. |
| `GET` | `/v1/journal/entries/{entry_id}` | Full entry with its stored reflection. |
| `PATCH` | `/v1/journal/entries/{entry_id}` | Edit text, publish a draft, save `userReframe`, or record `moodAfter` (a mood word). Changing the text clears the old reflection. |
| `DELETE` | `/v1/journal/entries/{entry_id}` | Delete the entry and the check-in it created. |
| `POST` | `/v1/journal/entries/{entry_id}/reflect` | Lucille's reflection, a reframe for unpleasant moods, themes, a tuned soundscape and 2 suggestions. Stored on the entry, so repeat calls are free (`?regenerate=true` to redo). Needs `ai_reflection` for the AI version; otherwise, or over quota, a written fallback is returned. |
| `GET` | `/v1/journal/insights` | `days` (7–90, default 14): mood spark, top themes, energy centers fed. |
| `GET` | `/v1/journal/weekly` | Lucille's weekly letter. Written once per ISO week (needs ≥ 2 saved entries), then read back. |
| `GET` | `/v1/journal/resurface` | "On this day" entry from about 1 year, 3 months or 1 month ago. |
| `POST` | `/v1/journal/letters` | Seal a letter to future you: `body`, `deliverInDays` 7 \| 30 \| 90. |
| `GET` | `/v1/journal/letters` | Letters; `body` is withheld until `deliverAt`. |
| `DELETE` | `/v1/journal/letters/{letter_id}` | Delete a letter. |

**`POST /v1/journal/entries`**

```json
{
  "mode": "guided",
  "status": "saved",
  "body": "Too many tabs open in my head about work.",
  "prompt": "You came in restless. What's the loudest thing in your head this evening?",
  "moodSeed": { "checkinId": "mood_…" },
  "energyCenter": "power",
  "voice": { "transcript": "…" },
  "tags": ["work"]
}
```

Gratitude uses `"gratitude": ["…", "…", "…"]`; Ritual Spark uses `"intention": "Tonight I will…"`. Response: `{ entry, safety, coinsAwarded, streakDays, checkinId? }`.

**`POST /v1/journal/entries/{id}/reflect`** response:

```json
{
  "reflection": { "reflection": "It sounds like a lot was asking for your attention…",
                  "reframe": "Instead of 'I have to handle all of it', try…",
                  "themes": ["work"], "energyCenter": "power", "soundscapeCategory": "binaural_beats",
                  "aiGenerated": true, "createdAt": "2026-10-09T03:12:00Z" },
  "mood": { "word": "Restless", "tone": "restless" },
  "orb": { "…": "…" },
  "soundscape": { "id": "binaural_beats_02", "title": "Alpha Calm 10 Hz", "deeplink": { "…": "…" } },
  "suggestions": [ "…" ],
  "disclosure": "Lucille is an AI companion, not a person or a licensed therapist…"
}
```

For a flagged (crisis) entry, `reflection.crisis` is `true`, the text points to 988, and `suggestions` is empty.

---

## 24. Soundscapes AI

Nine saved categories: `music_meditations` (Music Meditations), `vaporwave`, `jazz`, `nature`, `binaural_beats`, `brainwave_music`, `raw_frequencies`, `sleep_ambient` (Sleep / Ambient Music), `depression_anxiety` (Music for Depression & Anxiety). Modes: `picks` (Lucille Picks), `focus`, `calm`, `sleep`, `move`. The catalog is a static file (`escape_api/data/soundscape_catalog.json`, 39 tracks), so browsing costs no database reads.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/soundscapes/categories` | **Public.** The 9 categories (label, description, icon, colour, modes, disclaimer, track count) and the modes with their bpm ranges. |
| `GET` | `/v1/soundscapes/home` | One call for the Sound tab: Play for right now, Inner Weather, Daily Drop, one row of 4 tracks per category (ordered for the latest mood), last 3 sessions, streak, night flag, `checkInDue`. Optional `lat`, `lon`. |
| `GET` | `/v1/soundscapes/catalog` | Tracks. Query: `category`, `mode`, `q` (search), `word` (sort for a mood), `limit`. |
| `GET` | `/v1/soundscapes/tracks/{track_id}` | One track plus `locked` (premium) and the category disclaimer. |
| `GET` | `/v1/soundscapes/visual` | Nearest of the 36 Mood Orb loops for `mode`, `energy`, `texture` (0–1), with `_night` twin after local sunset. |
| `GET` | `/v1/soundscapes/inputs-now` | Inner Weather: day phase, night flag, last mood; with `lat`/`lon` and `location_weather` consent, sunrise/sunset and weather (cached 15 min per ~10 km). |
| `POST` | `/v1/soundscapes/pick` | Lucille Picks (`mode: "picks"` chooses the mode from time of day and mood and says why) or the best match for a mode + `moodField`. |
| `GET` | `/v1/soundscapes/daily-drop` | Today's drop for your cohort (date × time zone × weather bucket), shared by everyone in that cohort. |
| `POST` | `/v1/soundscapes/compose` | Lucille Compose. Builds a deterministic sound recipe and queues a render. Daily limit 3 free / 20 premium (`429 daily_limit`). Returns `{compositionId, status, etaSec, composition}`. |
| `GET` | `/v1/soundscapes/compositions` | Your compositions. |
| `GET` | `/v1/soundscapes/compositions/{cid}` | Poll one (or listen to `escape_users/{uid}/compositions/{cid}` in Firestore). Status: `queued` → `rendering` → `ready` \| `failed`. |
| `POST` | `/v1/soundscapes/sessions` | Start a session: `trackId` or `compositionId`, `mode`, `startedFrom`, optional `timerMinutes`, `moodField`, `moodBefore` (a mood word from the 10-second check-in). |
| `POST` | `/v1/soundscapes/sessions/{sid}/complete` | `minutes`, `completed`, optional `moodAfter`. At ≥ 5 minutes: streak +, 1 coin per 5 minutes. Returns before/after mood and `showFeedbackCard` (false after Sleep). Idempotent. |
| `GET` | `/v1/soundscapes/sessions` | Listening history. |
| `GET` | `/v1/soundscapes/library` | Saved tracks and compositions. |
| `PUT` | `/v1/soundscapes/library/{item_id}` | Save a track id or a `cmp_…` composition. |
| `DELETE` | `/v1/soundscapes/library/{item_id}` | Remove from the Library. |

**Track object**

```json
{
  "id": "jazz_01", "title": "Rainy Window Trio", "category": "jazz", "mode": "focus",
  "moodField": { "energy": 0.4, "texture": 0.4 }, "bpm": 76, "key": "Bb",
  "brainwave": null, "frequencyHz": null, "durationSec": 1800, "premium": false,
  "aiLabel": "Composed by Lucille (AI)",
  "audio": { "intro": "…/soundscapes/jazz/jazz_01/intro.m4a",
             "bodies": ["…/body_1.m4a", "…/body_2.m4a", "…/body_3.m4a", "…/body_4.m4a"],
             "outro": "…/outro.m4a", "preview": "…/preview.m4a",
             "codec": "aac-lc", "container": "m4a", "crossfadeSec": 4 },
  "visual": { "name": "escape_focus_e50_t50_v3", "hevc": "…_hevc.mp4", "h264": "…_h264.mp4", "poster": "…" }
}
```

Play the intro, then the 4 bodies shuffled with 4-second crossfades (never the same body twice in a row), then the outro when the timer ends.

**Render flow:** Compose POSTs to the render service ([section 28](#28-render-service-lucille-render)) with `X-Api-Key`. The render service calls back `POST /v1/soundscapes/render-callback/{uid}/{compositionId}` with `X-Lucille-Secret` and `{status, segments:[{role,url,durationSec}], measured, failReason}`, first with `rendering`, then `ready` or `failed`. A full set (45 s intro, 4 × 150 s bodies, 60 s outro) takes about 2 minutes.

---

## 25. Self-Care Score & Plans

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/selfcare/score` | Self-Care Score 0–100, cached per day until something changes. Components: mood 25%, rhythm 20%, reflection 20%, practice 20%, plan 15% (missing components are dropped and the weights rebalanced). Returns `band` (Rebuilding, Finding rhythm, Steady, Thriving), `delta` vs yesterday, `insight` and one `nextStep` deep link. A habit score, not a clinical measure. |
| `GET` | `/v1/selfcare/score/history` | Daily snapshots (`days` 7–180). |
| `GET` | `/v1/selfcare/activities` | **Public.** The 29-item activity library; every item maps to a real in-app route. |
| `GET` | `/v1/selfcare/goals` | **Public.** `sleep_better`, `less_stress`, `focus`, `lift_mood`, `move_more`, `connect`, `know_myself`. |
| `POST` | `/v1/selfcare/plans` | `{goals: [1–3], minutesPerDay: 5–90, days: 3–14, provider?: "catalog"\|"claude"}`. Builds a plan of morning / day / evening steps from the library and makes it the active plan. `claude` needs `ANTHROPIC_API_KEY` and the `personalization` consent, is capped by `QUOTA_PLAN_LLM_PER_DAY`, and falls back to the catalog plan. |
| `GET` | `/v1/selfcare/plans/active` | The active plan and today's steps. |
| `POST` | `/v1/selfcare/plans/{plan_id}/items/{item_id}/complete` | Tick a step: 10 coins (×6/day) and the streak. |
| `DELETE` | `/v1/selfcare/plans/{plan_id}` | Delete a plan. |

---

## 26. Privacy (GDPR) v1

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/privacy/purposes` | **Public.** Consent purposes and labels: `wellbeing_data`, `ai_reflection`, `camera_scan`, `voice`, `health_data`, `location_weather`, `personalization`, `analytics`. |
| `GET` | `/v1/privacy/consents` | Current flags, `restricted`, policy version and the last 20 changes. |
| `PUT` | `/v1/privacy/consents` | `{flags: {purpose: bool}}`. Grant or withdraw; each change is logged with time, policy version and IP. (Art. 7, 9) |
| `POST` | `/v1/privacy/restrict` | `{restricted: true}` turns off AI reflection, personalization, analytics, location and health data while keeping stored data. (Art. 18) |
| `GET` | `/v1/privacy/disclosure` | **Public.** AI disclosure text, how often to repeat it (180 min; 60 for minors), "not therapy" notice, crisis resources, what is stored. (Art. 13; state AI-companion laws) |
| `GET` | `/v1/privacy/export` | Everything stored for the user as a JSON download: the v1 tree plus the legacy Lucille export (`includeLegacy=false` to skip). (Art. 15, 20) |
| `DELETE` | `/v1/privacy/data?scope=` | Erase `mood`, `journal`, `soundscapes`, `plans` or `all` v1 data. `all` keeps only the consent history as proof of withdrawal. (Art. 17) |
| `POST` | `/v1/privacy/delete-account` | `{"confirm": "DELETE", "alsoDeleteAuthAccount": true, "alsoDeleteLegacyData": true}`. Deletes v1 data, legacy Lucille data, the FlutterFlow `Users/{uid}` document and the Firebase Auth account; returns a receipt. (Art. 17) |

Rectification (Art. 16) is `PATCH /v1/journal/entries/{id}` and `DELETE /v1/mood/checkins/{id}`.

---

## 27. Config, Jobs & Webhooks

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/health` | **Public.** v1 liveness and whether the AI key is configured. |
| `GET` | `/v1/config` | **Public.** Feature flags, asset and audio base URLs, deep-link scheme, AI disclosure, daily limits. Read once at launch; cache 1 h. |
| `POST` | `/v1/jobs/letters-due` | **Admin** (`INTERNAL_SERVICE_KEY`). Hourly Cloud Scheduler job: delivers due letters as FlutterFlow push notifications. `{dryRun: true}` to count only. |
| `POST` | `/v1/webhooks/revenuecat` | RevenueCat webhook (`Authorization: Bearer $REVENUECAT_WEBHOOK_SECRET`); sets `isPremium`. Hidden from Swagger. |
| `POST` | `/v1/soundscapes/render-callback/{uid}/{cid}` | Lucille render callback (`X-Lucille-Secret`). Hidden from Swagger. |

**Cost and behaviour settings** (environment variables, see `escape_api/settings.py`): `LUCILLE_V1_MODEL` (default `gpt-4o-mini`), `LUCILLE_V1_MAX_TOKENS` (380), `LUCILLE_V1_LLM_ENABLED`, `QUOTA_REFLECT_PER_DAY` (6), `QUOTA_GUIDED_PROMPT_PER_DAY` (6), `QUOTA_COMPOSE_PER_DAY_FREE` (3), `QUOTA_COMPOSE_PER_DAY_PREMIUM` (20), `QUOTA_PLAN_LLM_PER_DAY` (2), `ENFORCE_CONSENT` (true), `DEEPLINKS_ALLOW_PLANNED` (false), `ASSET_BASE_URL`, `AUDIO_BASE_URL`, `LUCILLE_RENDER_URL`, `LUCILLE_RENDER_API_KEY`, `LUCILLE_RENDER_CALLBACK_SECRET`, `PUBLIC_BASE_URL`, `WEATHER_PROVIDER` (`open-meteo`, free tier is non-commercial; change before paid launch), `PLAN_PROVIDER` (`catalog`), `ANTHROPIC_API_KEY`, `ANTHROPIC_PLAN_MODEL`, `REVENUECAT_WEBHOOK_SECRET`, `ENERGY_LEVEL_STEP_COINS` (1000), `LEGACY_REQUIRE_AUTH` (false).

---

## 28. Render Service (`lucille-render`)

A separate Cloud Run service (`render_service/`) that turns a sound recipe into a segment set of AAC-LC `.m4a` files (48 kHz stereo, 160 kb/s, two-pass EBU R128 loudness: Focus −16, Calm −18, Sleep −20 LUFS, true peak −1 dBTP). It synthesises pads, bells and bowls, nature beds (rain, ocean, forest, stream, fire), pulses, binaural or isochronic brainwave layers and pure frequency drones, seeded by the composition id so a request always renders the same audio. Binaural beats are added after stereo widening and survive AAC-LC (tested).

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness + whether ffmpeg is present. |
| `POST` | `/v1/render` | `X-Api-Key: $RENDER_API_KEY`. Returns 202 immediately; renders in the background, uploads to `gs://<project>-escape-media/audio/compose/{compositionId}/`, then calls `callbackUrl`. |
| `POST` | `/v1/render/sync` | Same body; renders and returns the result in the response (tests, tools). |

Request body:

```json
{
  "compositionId": "cmp_7Qk2d9", "mode": "sleep", "prompt": "Rain on a cabin roof…",
  "recipe": { "tempoBpm": 52, "key": "D", "scale": "major_pentatonic", "brightness": 0.25, "reverb": 0.72,
              "stereoWidth": 0.8, "layers": { "ambience": "rain_roof", "melody": "soft_pads", "pulse": "slow_heartbeat" },
              "brainwave": { "type": "theta", "hz": 6 } },
  "segments": { "introSec": 45, "bodyCount": 4, "bodySec": 150, "outroSec": 60 },
  "output": { "bitrateKbps": 160, "loudnessLufs": -20, "truePeakDbtp": -1 },
  "energy": 0.22,
  "callbackUrl": "https://lucille-….run.app/v1/soundscapes/render-callback/{uid}/cmp_7Qk2d9"
}
```

Optional: `style` (`ambient` | `jazz` | `vaporwave` | `bowls` | `drone`), `droneHz`, `destPrefix` (GCS path override). Pre-render the catalog with `python -m render_service.catalog` (all 39 tracks, or `--only jazz_01`, `--free-only`, `--preview-only`, `--local out/`). Deployed by `cloudbuild-render.yaml` (2 CPU, 2 GiB, CPU always on, 2 renders per instance, max 5 instances). Secrets: `lucille-render-api-key`, `lucille-render-callback-secret`.

---

## Authentication

| Endpoints | Auth |
|-----------|------|
| `/v1/*` | Firebase ID token required (`Authorization: Bearer …`), except the rows marked **Public**. The user comes from the token. |
| `/chat`, `/chat/stream`, `/chat/voice` and other routes taking `user: get_current_user` | Firebase ID token or `INTERNAL_SERVICE_KEY`. |
| `/users/{user_id}/export`, `/users/{user_id}/data`, `/users/{user_id}/consent`, `/users/{user_id}/mood/analyze-image`, `/users/{user_id}/selfcare-score` | The same user's token, or an admin. |
| `/admin/*`, `/finetuning/*`, `/v1/jobs/*` | Admin: `INTERNAL_SERVICE_KEY`, or a Firebase user with the `admin: true` custom claim. |
| Other legacy user routes (`/users/{user_id}/…`, `/therapy/{user_id}/…`, `/soundscapes/{user_id}/…`, `/feedback`, `/wearables`, `/reviews`, `/assessments`, `/rl`, `/safety/{user_id}/audit`) | Open today so the current app keeps working. Set `LEGACY_REQUIRE_AUTH=true` once the FlutterFlow API groups send the token; then each needs that user's own token. |

---

## Common Request Headers

| Header | Value | When Required |
|--------|-------|---------------|
| `Content-Type` | `application/json` | All POST/PUT requests |
| `Accept` | `text/event-stream` | `/chat/stream` (SSE) |
| `Authorization` | `Bearer <Firebase ID token>` | All `/v1` calls (except public ones) and the protected routes above |
| `X-Timezone` | `America/New_York` or `-04:00` | All `/v1` calls (sets "today", streaks, night mode) |

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
| `401` | Missing, expired or revoked token (the app should refresh the token and retry once) |
| `403` | Not your data, admin only, or `{"detail":{"code":"consent_required","purpose":"…"}}` on `/v1` |
| `422` | Body or query failed validation (FastAPI lists the fields) |
| `404` | Resource not found (session, user, exercise, etc.) |
| `429` | Rate limited (10 req/min for /chat, 100 req/min global), or `/v1` daily limit `{"detail":{"code":"daily_limit","limit":3}}` |
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
[main.py - legacy endpoints]  +  [escape_api/ - /v1 Escape API, 67 operations]
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

escape_api/  (mounted by register(app) in main.py)
    +---> routers/               (me, mood, journal, soundscapes, selfcare, privacy, meta)
    +---> mood.py, journal.py, soundscapes.py, selfcare.py   (logic, no FastAPI)
    +---> core.py                (datasets, time zones, streak/coins, deep-link allowlist, safety)
    +---> llm.py                 (one small model, JSON mode, quotas, fallbacks)
    +---> repo.py                (Firestore + in-memory twin; data under escape_users/{uid})
    +---> legacy_guard.py        (auth on legacy GDPR routes; LEGACY_REQUIRE_AUTH)
    +---> data/*.json            (mood words, prompts, activities, deep links, soundscape catalog)
```
