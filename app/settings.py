"""Application settings, read from environment / .env."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # GCP / Vertex AI
    # Optional: populated by .env locally; empty in Cloud Run (uses ADC via runtime SA).
    google_application_credentials: str = ""
    gcp_project_id: str
    gcp_location: str = "us-central1"

    # Models
    gemini_model_pro: str = "gemini-2.5-pro"
    gemini_model_flash: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"

    # World_Monitor intel layer
    world_monitor_base: str = "https://finance.worldmonitor.app"
    world_monitor_timeout_s: float = 3.0
    intel_mode: str = "auto"  # auto | live | snapshot


settings = Settings()
