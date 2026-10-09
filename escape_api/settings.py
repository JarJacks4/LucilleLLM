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
        self.asset_base_url = os.getenv("ASSET_BASE_URL", "https://storage.googleapis.com/escape-self-care-ai.appspot.com")
        self.audio_base_url = os.getenv("AUDIO_BASE_URL", os.getenv("ASSET_BASE_URL", "https://storage.googleapis.com/escape-self-care-ai.appspot.com"))

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
