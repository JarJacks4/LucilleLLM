"""Environment-driven settings for the v1 API. Every cost knob lives here."""

import os


def _b(key: str, default: bool) -> bool:
    v = os.getenv(key)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _i(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except ValueError:
        return default


class Settings:
    """Read on construction so tests can change env and call reload_settings()."""

    def __init__(self) -> None:
        # LLM: one small model, short outputs, JSON mode. gpt-4o-mini is the repo default.
        self.llm_model = os.getenv("LUCILLE_V1_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        self.llm_max_tokens = _i("LUCILLE_V1_MAX_TOKENS", 380)
        self.llm_timeout_s = _i("LUCILLE_V1_TIMEOUT_S", 20)
        self.llm_enabled = _b("LUCILLE_V1_LLM_ENABLED", True)

        # Per-user daily quotas on anything that costs money (LLM calls, renders).
        self.quota_reflect_per_day = _i("QUOTA_REFLECT_PER_DAY", 6)
        self.quota_guided_prompt_per_day = _i("QUOTA_GUIDED_PROMPT_PER_DAY", 6)
        self.quota_compose_per_day_free = _i("QUOTA_COMPOSE_PER_DAY_FREE", 3)
        self.quota_compose_per_day_premium = _i("QUOTA_COMPOSE_PER_DAY_PREMIUM", 20)
        self.quota_plan_llm_per_day = _i("QUOTA_PLAN_LLM_PER_DAY", 2)

        # GDPR: mood + journal are health data (Art. 9) -> explicit consent before storing.
        self.enforce_consent = _b("ENFORCE_CONSENT", True)
        # Deep links to pages that are still being built.
        self.deeplinks_allow_planned = _b("DEEPLINKS_ALLOW_PLANNED", False)

        # Assets: GCS / CDN base for orbs, loops and audio (public-read bucket or CDN).
        self.asset_base_url = os.getenv("ASSET_BASE_URL", "https://storage.googleapis.com/escape-self-care-505618-escape-media")
        self.audio_base_url = os.getenv("AUDIO_BASE_URL", os.getenv("ASSET_BASE_URL", "https://storage.googleapis.com/escape-self-care-505618-escape-media"))

        # Lucille render service (soundscape Compose). Empty = queue only.
        self.render_url = os.getenv("LUCILLE_RENDER_URL", "")
        self.render_api_key = os.getenv("LUCILLE_RENDER_API_KEY", "")
        self.render_callback_secret = os.getenv("LUCILLE_RENDER_CALLBACK_SECRET", "")
        self.public_base_url = os.getenv("PUBLIC_BASE_URL", "https://lucille-861854898360.us-central1.run.app")

        # Weather for Inner Weather. Open-Meteo needs no key but its free tier is non-commercial:
        # switch WEATHER_PROVIDER to 'none' or a paid key before paid launch.
        self.weather_provider = os.getenv("WEATHER_PROVIDER", "open-meteo")

        # Self-Care plan provider: 'catalog' today; 'claude' later (ANTHROPIC_API_KEY).
        self.plan_provider = os.getenv("PLAN_PROVIDER", "catalog")
        self.anthropic_model = os.getenv("ANTHROPIC_PLAN_MODEL", "claude-haiku-4-5")

        # ── Lucille edition ────────────────────────────────────────────────
        # 'openai' keeps today's app exactly as it is. 'claude' additionally mounts
        # /v2/lucille/* and /v2/plans/* (the Claude edition). Nothing is ever mixed.
        self.lucille_edition = (os.getenv("LUCILLE_EDITION", "openai") or "openai").strip().lower()

        # Claude models (platform.claude.com, Oct 2026). 1M context, 128K max output.
        self.claude_model = os.getenv("LUCILLE_CLAUDE_MODEL", "claude-opus-5-5")
        self.claude_fast_model = os.getenv("LUCILLE_CLAUDE_FAST_MODEL", "claude-haiku-5-5")
        self.claude_cheap_model = os.getenv("LUCILLE_CLAUDE_CHEAP_MODEL", "claude-sonnet-5-5")
        self.claude_long_model = os.getenv("LUCILLE_CLAUDE_LONG_MODEL", "claude-fable-5-1")
        self.claude_long_reasoning = _b("LUCILLE_CLAUDE_LONG_REASONING", False)  # opt-in, pricier

        self.claude_max_tokens = _i("LUCILLE_CLAUDE_MAX_TOKENS", 1600)
        self.claude_timeout_s = _i("LUCILLE_CLAUDE_TIMEOUT_S", 60)
        self.claude_connect_timeout_s = _i("LUCILLE_CLAUDE_CONNECT_TIMEOUT_S", 10)
        self.claude_max_retries = _i("LUCILLE_CLAUDE_MAX_RETRIES", 2)
        self.claude_base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        self.claude_api_version = os.getenv("ANTHROPIC_API_VERSION", "2023-06-01")

        # Memory since inception: verbatim window + fold-down threshold.
        self.claude_verbatim_turns = _i("LUCILLE_CLAUDE_VERBATIM_TURNS", 40)
        self.claude_memory_top_k = _i("LUCILLE_CLAUDE_MEMORY_TOP_K", 20)
        self.claude_context_budget_tokens = _i("LUCILLE_CLAUDE_CONTEXT_BUDGET", 400_000)
        self.claude_tool_rounds = _i("LUCILLE_CLAUDE_TOOL_ROUNDS", 4)
        self.claude_memory_write = _b("LUCILLE_CLAUDE_MEMORY_WRITE", True)

        # Per-user daily quotas for the Claude edition.
        self.quota_claude_turn_per_day = _i("QUOTA_CLAUDE_TURN_PER_DAY", 60)
        self.quota_claude_plan_per_day = _i("QUOTA_CLAUDE_PLAN_PER_DAY", 4)
        self.quota_claude_doc_per_day = _i("QUOTA_CLAUDE_DOC_PER_DAY", 10)

        # Plan documents: GCS bucket + signed-URL lifetime. Empty bucket = served
        # from this API behind the caller's token instead (no public object).
        self.plan_docs_bucket = os.getenv("LUCILLE_PLAN_DOCS_BUCKET", "")
        self.plan_docs_ttl_hours = _i("LUCILLE_PLAN_DOCS_TTL_HOURS", 24)

        # AI disclosure cadence (contract): adults 3h / 12 turns, teens 1h / 5 turns.
        self.disclosure_hours_adult = _i("DISCLOSURE_HOURS_ADULT", 3)
        self.disclosure_turns_adult = _i("DISCLOSURE_TURNS_ADULT", 12)
        self.disclosure_hours_teen = _i("DISCLOSURE_HOURS_TEEN", 1)
        self.disclosure_turns_teen = _i("DISCLOSURE_TURNS_TEEN", 5)

        # Coins
        self.energy_level_step = _i("ENERGY_LEVEL_STEP_COINS", 1000)


settings = Settings()


def cfg() -> Settings:
    """Always read settings through this so reload_settings() takes effect everywhere."""
    return settings


def reload_settings() -> Settings:
    """Tests change env vars and call this."""
    global settings
    settings = Settings()
    return settings
