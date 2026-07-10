from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GeeAuthConfig:
    """Non-secret metadata required for Earth Engine service-account auth."""

    key_path: Path
    project_id: str | None
    service_account_email: str | None


def load_service_account_metadata(key_path: Path) -> GeeAuthConfig:
    """Read only non-secret metadata; never print the key contents."""
    if not key_path.exists():
        raise FileNotFoundError(f"Missing GEE service account key at {key_path}")
    with key_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    return GeeAuthConfig(
        key_path=key_path,
        project_id=metadata.get("project_id"),
        service_account_email=metadata.get("client_email"),
    )


def initialize_earth_engine(auth: GeeAuthConfig) -> Any:
    """Authenticate with the service account key without exposing secrets."""
    import ee

    if not auth.project_id:
        raise ValueError("gee-key.json does not contain a project_id.")
    if not auth.service_account_email:
        raise ValueError("gee-key.json does not contain a client_email.")

    credentials = ee.ServiceAccountCredentials(auth.service_account_email, str(auth.key_path))
    try:
        # This project works with legacy service-account initialization. Passing
        # an explicit project can require extra Service Usage permissions.
        ee.Initialize(credentials)
    except Exception as exc:  # noqa: BLE001 - Earth Engine wraps Google API errors.
        message = str(exc)
        if "serviceusage.services.use" in message or "serviceUsageConsumer" in message:
            raise RuntimeError(
                "Earth Engine reached Google Cloud, but the service account lacks permission "
                "to use the configured project. Grant roles/serviceusage.serviceUsageConsumer "
                "or use legacy service-account initialization."
            ) from exc
        if "Earth Engine API" in message or "earthengine.googleapis.com" in message:
            raise RuntimeError(
                "Earth Engine initialization failed. Check that the project is registered for "
                "Earth Engine and that the Earth Engine API is enabled."
            ) from exc
        raise
    return ee

