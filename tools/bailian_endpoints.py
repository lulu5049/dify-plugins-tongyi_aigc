from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

_COMPATIBLE_SUFFIX = "/compatible-mode/v1"
_NATIVE_SUFFIX = "/api/v1"


def normalize_compatible_base_url(base_url: str) -> str:
    """Normalize a Bailian URL to the OpenAI-compatible /compatible-mode/v1 base."""
    value = (base_url or "").strip().rstrip("/")
    if not value:
        raise ValueError("Bailian Base URL is required")

    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "Bailian Base URL must be a valid http(s) URL, for example "
            "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
        )

    path = parsed.path.rstrip("/")
    origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip("/")

    if path.endswith(_COMPATIBLE_SUFFIX):
        prefix = path[:-len(_COMPATIBLE_SUFFIX)].rstrip("/")
        return f"{origin}{prefix}{_COMPATIBLE_SUFFIX}"

    if path.endswith(_NATIVE_SUFFIX):
        prefix = path[:-len(_NATIVE_SUFFIX)].rstrip("/")
        return f"{origin}{prefix}{_COMPATIBLE_SUFFIX}"

    if path in {"", "/"}:
        return f"{origin}{_COMPATIBLE_SUFFIX}"

    raise ValueError(
        "Unsupported Bailian Base URL path. Paste the URL shown by Bailian, "
        "ending in /compatible-mode/v1 (or /api/v1)."
    )


def compatible_base_url(credentials: Mapping[str, Any]) -> str:
    return normalize_compatible_base_url(str(credentials.get("base_url") or ""))


def native_base_url(credentials: Mapping[str, Any]) -> str:
    compatible = compatible_base_url(credentials)
    return compatible[:-len(_COMPATIBLE_SUFFIX)] + _NATIVE_SUFFIX
