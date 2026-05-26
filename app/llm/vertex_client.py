"""Thin wrapper around the google-genai client, pointed at Vertex AI.

All LLM calls in the POC go through `generate()`. Structured output is enabled
by passing a Pydantic model as `response_schema`.
"""
import base64
import os
import tempfile
from pathlib import Path

from google import genai
from google.genai import types

from app.settings import settings


def _resolve_credentials() -> None:
    """Point Application Default Credentials at a usable service-account key.

    Resolution order (first that works wins):
      1. Local key FILE named by `google_application_credentials` (dev on a laptop).
      2. Key supplied as an ENV-VAR SECRET — `GCP_SA_KEY_B64` (base64) or
         `GCP_SA_KEY_JSON` (raw JSON). This is how container hosts that only expose
         secrets as env vars (e.g. Hugging Face Spaces) deliver the key: we write it
         to a temp file and point ADC at it.
      3. Neither set → leave ADC ambient (Cloud Run runtime service account, or a
         `gcloud`-authenticated environment). Do NOT set the var to a bogus path.
    """
    setting = (settings.google_application_credentials or "").strip()
    if setting:
        p = Path(setting)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[2] / p
        if p.is_file():  # a real key file, not a directory / empty string
            os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(p))
            return

    key_b64 = os.environ.get("GCP_SA_KEY_B64", "").strip()
    key_json = os.environ.get("GCP_SA_KEY_JSON", "").strip()
    if key_b64 or key_json:
        content = base64.b64decode(key_b64).decode("utf-8") if key_b64 else key_json
        tmp = Path(tempfile.gettempdir()) / "gcp_sa_key.json"
        tmp.write_text(content, encoding="utf-8")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(tmp)
        return
    # else: rely on ambient ADC (Cloud Run runtime SA / gcloud login).


_resolve_credentials()

client = genai.Client(
    vertexai=True,
    project=settings.gcp_project_id,
    location=settings.gcp_location,
)


def generate(
    model: str,
    contents,
    *,
    response_schema=None,
    system_instruction=None,
    tools=None,
):
    """Call a Gemini model on Vertex AI.

    When `response_schema` is provided the response is forced to JSON matching
    that schema; access the parsed object via `response.parsed`.
    """
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json" if response_schema else None,
        response_schema=response_schema,
        system_instruction=system_instruction,
        tools=tools,
    )
    return client.models.generate_content(model=model, contents=contents, config=cfg)
