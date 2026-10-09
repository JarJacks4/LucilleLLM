# Escape v1 API (Lucille)

New endpoints for the reworked Journal, Mood Scan / Mood Stats, Soundscapes AI, Self-Care Score + Plans, and GDPR.
They live in `escape_api/` and are mounted by two lines in `main.py`. **Every legacy endpoint in `main.py` is unchanged**
(the current app calls 47 of them), except that the GDPR routes the app never calls now require auth (see *Security*).

* Base URL: `https://lucille-861854898360.us-central1.run.app`
* Every call: `Authorization: Bearer <Firebase ID token>` (FlutterFlow: `currentJwtToken`) and `X-Timezone: America/New_York`
* The uid always comes from the token. There is no `{user_id}` in v1 paths, so one user can never read another's data.
* OpenAPI (import into FlutterFlow): `docs/escape_v1_openapi.json` · live docs: `/docs` on the service

## Endpoints (67 operations)

| Area | Method + path | What it's for |
|---|---|---|
| Config | `GET /v1/config` · `GET /v1/health` | Feature flags, asset base URLs, AI disclosure (cache 1 h) |
| Me | `GET/PATCH /v1/me` · `GET /v1/me/progress` · `GET /v1/energy/centers` | Time zone, name, prefs, age band; one streak; coins; 7 energy centers (1,000 coins each, Grounding → Purpose) |
| Home | `GET /v1/home` | One call: Mood Orb, Self-Care Score, streak, coins, today's plan, one suggestion |
| Suggestions | `GET /v1/suggestions?context=` | Lucille Suggestions; every link checked against `escape_api/data/deeplinks.json` |
| Mood | `POST /v1/mood/checkins` | Save a Mood Scan: fuses tap (Mood Field / word), voice text and pulse (bpm/hrv). Returns word, tone, orb video URLs, suggestions, a journal seed, safety |
| | `POST /v1/mood/interpret` | Same reading without saving (live preview while scanning) |
| | `GET /v1/mood/summary?period=week\|month\|year` | The single conglomerate Mood Orb: word, tone, size, caption, mix, trend |
| | `GET /v1/mood/stats` · `GET /v1/mood/checkins` · `GET /v1/mood/latest` · `DELETE /v1/mood/checkins/{id}` · `GET /v1/mood/words` · `GET /v1/mood/suggestions` · `GET /v1/mood/orb` | Mood Stats page, history, rectification |
| Journal | `GET /v1/journal/home` | Orb for the period, last scan (≤12 h) to "Save this moment", weekly reflection, 4 modes |
| | `POST /v1/journal/prompt` | Prompt for Free / Guided / Gratitude / Ritual Spark from the dataset; Guided can be personalised (quota) |
| | `POST/GET /v1/journal/entries` · `GET/PATCH/DELETE /v1/journal/entries/{id}` | Entries (draft or saved), voice transcript, energy-center tag, user reframe, mood after |
| | `POST /v1/journal/entries/{id}/reflect` | Lucille reflects + reframe + tuned soundscape + 2 suggestions. Stored on the entry; repeat calls are free |
| | `GET /v1/journal/insights` · `GET /v1/journal/weekly` · `GET /v1/journal/resurface` | 14-day spark, themes, energy feed; weekly letter (1 per week); "On this day" |
| | `POST/GET /v1/journal/letters` · `DELETE /v1/journal/letters/{id}` | Letter to future self; sealed until 7/30/90 days, then a push |
| Soundscapes | `GET /v1/soundscapes/categories` | The 9 categories: Music Meditations, Vaporwave, Jazz, Nature, Binaural Beats, Brainwave Music, Raw Frequencies, Sleep / Ambient, Music for Depression & Anxiety |
| | `GET /v1/soundscapes/home` | Play for right now, Inner Weather, Daily Drop, 9 category rows, recents |
| | `GET /v1/soundscapes/catalog` · `GET /v1/soundscapes/tracks/{id}` | Tracks with intro + 4 body + outro segment URLs and their Mood Orb loop |
| | `POST /v1/soundscapes/pick` | Lucille Picks or best match for a mode + Mood Field |
| | `GET /v1/soundscapes/inputs-now` · `GET /v1/soundscapes/daily-drop` · `GET /v1/soundscapes/visual` | Day phase, sun, night flag, weather; cohort Daily Drop; nearest of the 36 loops (+ `_night`) |
| | `POST /v1/soundscapes/compose` · `GET /v1/soundscapes/compositions[/{id}]` | Lucille Compose (deterministic recipe → render service, daily limit) |
| | `POST /v1/soundscapes/sessions` · `POST /v1/soundscapes/sessions/{id}/complete` · `GET /v1/soundscapes/sessions` | Session with before/after mood, streak, coins |
| | `GET /v1/soundscapes/library` · `PUT/DELETE /v1/soundscapes/library/{id}` | Saved tracks and compositions |
| Self-care | `GET /v1/selfcare/score` · `GET /v1/selfcare/score/history` | Self-Care Score 0–100: mood, rhythm, reflection, practice, plan; band; one next step |
| | `GET /v1/selfcare/activities` · `GET /v1/selfcare/goals` · `POST /v1/selfcare/plans` · `GET /v1/selfcare/plans/active` · `POST /v1/selfcare/plans/{id}/items/{item}/complete` · `DELETE /v1/selfcare/plans/{id}` | Plans built only from real in-app activities. `provider: "claude"` lets Claude order the plan later |
| Privacy | `GET /v1/privacy/purposes` · `GET/PUT /v1/privacy/consents` · `POST /v1/privacy/restrict` · `GET /v1/privacy/disclosure` · `GET /v1/privacy/export` · `DELETE /v1/privacy/data?scope=` · `POST /v1/privacy/delete-account` | GDPR Art. 7, 9, 13, 15, 16, 17, 18, 20, 21 + AI-companion disclosure |
| Jobs | `POST /v1/jobs/letters-due` (admin) · `POST /v1/webhooks/revenuecat` | Hourly Cloud Scheduler job; premium flag from RevenueCat |

## Data (Firestore)

All v1 data for a user sits under **one** document, `escape_users/{uid}`, so export and erasure are a single tree walk and
the FlutterFlow `Users` schema is never touched. Rules: owner may read, nobody may write (only the API writes).
Subcollections: `journal_entries`, `journal_letters`, `mood_checkins`, `mood_days`, `mood_months` (rollups),
`listening_sessions`, `library`, `compositions`, `plans`, `scores`, `weekly`, `usage`; docs `state`, `consents`.
Deploy `firestore.rules` and `firestore.indexes.json` (`firebase deploy --only firestore`).

## Cost controls

* **LLM only where it adds value**: entry reflection, the weekly letter, optional guided prompts. Everything else
  (mood reads, prompts, themes, suggestions, plans, score, sound recipes) is datasets + rules, so it is free.
* One small model (`LUCILLE_V1_MODEL`, default `gpt-4o-mini`), JSON mode, `max_tokens` 380, 20 s timeout, one retry.
* Results are stored where they belong (a reflection is generated once per entry; the weekly letter once per week).
  Guided prompts are cached per mood + themes cohort, not per user.
* Per-user daily quotas: `QUOTA_REFLECT_PER_DAY` (6), `QUOTA_GUIDED_PROMPT_PER_DAY` (6), `QUOTA_COMPOSE_PER_DAY_FREE` (3).
  Over quota → a good dataset reflection, never an error.
* Any LLM failure or missing key → deterministic fallback, so screens never break.
* Firestore: a check-in is 4 writes (doc + 2 rollups + state). Mood summaries read 1 range query (week/month) or
  12 docs (year) and are cached per instance until the next check-in. The catalog, prompts and activities are
  static files (0 reads). Daily Drops are per cohort (date × time zone × weather), not per user.
* Rough worst case at 1,000 daily users each hitting the reflection cap: 6,000 calls/day × ~1k tokens ≈ $1.50/day on
  gpt-4o-mini list prices (check current pricing). Realistic usage is a small fraction of that.
* Cloud Run stays at `min-instances=0`, `concurrency=80`.

## Safety + compliance built in

* Every journal entry, mood note and compose prompt runs the existing keyword safety screen (no API call). Crisis →
  entry flagged, no LLM call, 988 resources returned, event logged to `safety_audit`.
* Mood + journal storage needs explicit `wellbeing_data` consent (`ENFORCE_CONSENT=true`); AI reflection needs
  `ai_reflection`; camera pulse needs `camera_scan`; voice needs `voice`. 403 `consent_required` tells the app which.
* AI output carries `aiGenerated` and `/v1/privacy/disclosure` returns the AI disclosure text and how often to repeat it
  (every 3 h, every 1 h for minors). Lucille's system prompt forbids diagnosis or treatment claims.
* `PATCH /v1/me {birthYear}` stores an age band and `isMinor` for minor protections.

## Security fix included

The legacy `GET /users/{user_id}/export`, `DELETE /users/{user_id}/data` and `/users/{user_id}/consent` routes had no
auth, so anyone who knew a uid could export or erase that user's data. `escape_api/legacy_guard.py` now requires the
caller's own token on those (the app does not call them). All other legacy user routes keep working as today; set
`LEGACY_REQUIRE_AUTH=true` once the FlutterFlow calls send `Authorization: Bearer [currentJwtToken]`.

## Render service

`render_service/` is a second Cloud Run service (`lucille-render`) that renders Compose requests and the catalog into
AAC-LC segment sets and calls the API back. See `API_REFERENCE.md` section 28. Media (orbs, loops, audio) lives in the
public bucket `<project>-escape-media`, created by `scripts/gcp/setup_deploy_access.sh`, so the Firebase bucket
(which holds user uploads) stays private.

## Lucille's prompts

The reflection and weekly-letter prompts (`escape_api/llm.py`, `escape_api/journal.py`) wrap the person's writing in
`<entry>` tags and treat it as content only, never as instructions. The code, not the model, decides when a reframe is
offered: unpleasant moods or negative self-talk, in Free and Guided modes only. Every model reply is checked before it
is stored (length, at most one question, no diagnosis or medication language, no "you should", no claims of being
human, no dependence language); a reply that fails is replaced by the written fallback.

## Config (env vars)

`LUCILLE_V1_MODEL`, `LUCILLE_V1_MAX_TOKENS`, `LUCILLE_V1_LLM_ENABLED`, `QUOTA_*`, `ENFORCE_CONSENT`,
`DEEPLINKS_ALLOW_PLANNED` (true once Journal V2 / Mood Stats V2 pages ship), `ASSET_BASE_URL`, `AUDIO_BASE_URL`,
`LUCILLE_RENDER_URL`, `LUCILLE_RENDER_API_KEY`, `LUCILLE_RENDER_CALLBACK_SECRET`, `PUBLIC_BASE_URL`,
`WEATHER_PROVIDER` (`open-meteo` free tier is non-commercial: switch before paid launch), `PLAN_PROVIDER`
(`catalog` | `claude`), `ANTHROPIC_API_KEY`, `ANTHROPIC_PLAN_MODEL`, `REVENUECAT_WEBHOOK_SECRET`, `LEGACY_REQUIRE_AUTH`.

## Tests

`python -m pytest tests -q` (170 tests: `tests/v1` covers every v1 flow with in-memory storage and no network; `tests/render` renders short segment sets and checks loudness and the binaural beat). `scripts/eval_reflections.py` runs Lucille's reflection prompt against the live model on 8 tricky entries.
