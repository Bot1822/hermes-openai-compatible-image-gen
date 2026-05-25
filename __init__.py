"""OpenAI-compatible image generation provider for Hermes Agent.

This plugin registers an ``ImageGenProvider`` that talks to any OpenAI-compatible
``/v1/images/generations`` endpoint, including NewAPI/one-api/LiteLLM style
proxies that expose ``gpt-image-2``.

Credential/base URL selection precedence:

1. ``OPENAI_COMPAT_IMAGE_API_KEY`` / ``OPENAI_COMPAT_IMAGE_BASE_URL`` env vars
2. ``image_gen.openai_compatible.api_key`` / ``base_url`` in config.yaml
3. ``image_gen.openai_compatible.provider`` referencing ``providers.<name>``
   in config.yaml (uses provider ``api``/``base_url`` and ``api_key``)

Model selection precedence:

1. ``OPENAI_COMPAT_IMAGE_MODEL`` env var
2. ``image_gen.openai_compatible.model`` in config.yaml
3. ``image_gen.model`` in config.yaml
4. ``gpt-image-2`` default

Quality selection precedence:

1. Explicit tool/provider ``quality`` kwarg
2. ``OPENAI_COMPAT_IMAGE_QUALITY`` env var
3. ``image_gen.openai_compatible.quality`` in config.yaml
4. legacy tier model suffix (``gpt-image-2-low`` / ``-medium`` / ``-high``)
5. ``low`` default
"""

from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    save_url_image,
    success_response,
)

logger = logging.getLogger(__name__)

PROVIDER_NAME = "openai-compatible"
CONFIG_KEY = "openai_compatible"
DEFAULT_API_MODEL = "gpt-image-2"
DEFAULT_QUALITY = "low"
DEFAULT_TIMEOUT = 300.0
VALID_QUALITIES = {"low", "medium", "high", "auto"}

_MODELS: Dict[str, Dict[str, Any]] = {
    "gpt-image-2": {
        "display": "GPT Image 2",
        "speed": "quality-dependent",
        "strengths": "OpenAI-compatible GPT Image 2 endpoint; quality configured separately",
        "api_model": "gpt-image-2",
    },
}

# Backward compatibility for old configs. These IDs are accepted if present in
# image_gen.model / image_gen.openai_compatible.model, but are intentionally not
# advertised in list_models() because quality belongs in a separate field.
_LEGACY_TIER_MODELS: Dict[str, Dict[str, str]] = {
    "gpt-image-2-low": {"api_model": "gpt-image-2", "quality": "low"},
    "gpt-image-2-medium": {"api_model": "gpt-image-2", "quality": "medium"},
    "gpt-image-2-high": {"api_model": "gpt-image-2", "quality": "high"},
}

_SIZES = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}


def _load_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        return cfg if isinstance(cfg, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not load Hermes config: %s", exc)
        return {}


def _image_gen_config() -> Dict[str, Any]:
    cfg = _load_config()
    section = cfg.get("image_gen")
    return section if isinstance(section, dict) else {}


def _plugin_config() -> Dict[str, Any]:
    section = _image_gen_config().get(CONFIG_KEY)
    return section if isinstance(section, dict) else {}


def _provider_config(provider_name: str) -> Dict[str, Any]:
    cfg = _load_config()
    providers = cfg.get("providers")
    if not isinstance(providers, dict):
        return {}
    provider = providers.get(provider_name)
    return provider if isinstance(provider, dict) else {}


def _first_str(*values: Any) -> Optional[str]:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _resolve_endpoint() -> Tuple[Optional[str], Optional[str], str]:
    """Return (base_url, api_key, source_label)."""
    cfg = _plugin_config()

    env_base = os.environ.get("OPENAI_COMPAT_IMAGE_BASE_URL")
    env_key = os.environ.get("OPENAI_COMPAT_IMAGE_API_KEY")
    if env_base or env_key:
        return _first_str(env_base), _first_str(env_key), "env"

    cfg_base = _first_str(cfg.get("base_url"), cfg.get("api"))
    cfg_key = _first_str(cfg.get("api_key"), cfg.get("key"))
    if cfg_base or cfg_key:
        return cfg_base, cfg_key, "image_gen.openai_compatible"

    provider_name = cfg.get("provider")
    if isinstance(provider_name, str) and provider_name.strip():
        pcfg = _provider_config(provider_name.strip())
        return (
            _first_str(pcfg.get("api"), pcfg.get("base_url"), pcfg.get("api_base")),
            _first_str(pcfg.get("api_key"), pcfg.get("key")),
            f"providers.{provider_name.strip()}",
        )

    return None, None, "unset"


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    env_model = os.environ.get("OPENAI_COMPAT_IMAGE_MODEL")
    if env_model:
        return _model_meta(env_model.strip())

    cfg = _plugin_config()
    candidate = cfg.get("model")
    if isinstance(candidate, str) and candidate.strip():
        return _model_meta(candidate.strip())

    top = _image_gen_config().get("model")
    if isinstance(top, str) and top.strip():
        return _model_meta(top.strip())

    return DEFAULT_API_MODEL, _MODELS[DEFAULT_API_MODEL]


def _model_meta(model_id: str) -> Tuple[str, Dict[str, Any]]:
    if model_id in _MODELS:
        return model_id, _MODELS[model_id]
    if model_id in _LEGACY_TIER_MODELS:
        legacy = _LEGACY_TIER_MODELS[model_id]
        return legacy["api_model"], {
            "display": legacy["api_model"],
            "speed": "quality-dependent",
            "strengths": "Legacy quality-tier model id; prefer image_gen.openai_compatible.quality",
            "api_model": legacy["api_model"],
            "legacy_quality": legacy["quality"],
            "legacy_model_id": model_id,
        }
    # Allow arbitrary compatible model ids configured by the user. Treat them
    # as direct API model ids.
    return model_id, {
        "display": model_id,
        "speed": "varies",
        "strengths": "Custom OpenAI-compatible image model",
        "api_model": model_id,
    }


def _normalize_quality(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    quality = value.strip().lower()
    return quality if quality in VALID_QUALITIES else None


def _resolve_quality(meta: Dict[str, Any], explicit: Any = None) -> str:
    cfg = _plugin_config()
    return (
        _normalize_quality(explicit)
        or _normalize_quality(os.environ.get("OPENAI_COMPAT_IMAGE_QUALITY"))
        or _normalize_quality(cfg.get("quality"))
        or _normalize_quality(meta.get("legacy_quality"))
        or DEFAULT_QUALITY
    )


def _normalize_base_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if base.endswith("/images/generations"):
        return base[: -len("/images/generations")]
    return base


def _request_json(base_url: str, api_key: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    url = _normalize_base_url(base_url) + "/images/generations"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310 - user-configured API endpoint
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code}: {raw[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error: {exc}") from exc

    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Invalid JSON response: {raw[:500]!r}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected JSON object response, got {type(parsed).__name__}")
    return parsed


def _extract_image_payload(response: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (b64_json, url, revised_prompt)."""
    data = response.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            return (
                _first_str(first.get("b64_json"), first.get("base64")),
                _first_str(first.get("url")),
                _first_str(first.get("revised_prompt")),
            )
    # Be forgiving for proxy variants that return top-level fields.
    return (
        _first_str(response.get("b64_json"), response.get("result"), response.get("image_base64")),
        _first_str(response.get("url"), response.get("image_url")),
        _first_str(response.get("revised_prompt")),
    )


class OpenAICompatibleImageGenProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def display_name(self) -> str:
        return "OpenAI Compatible"

    def is_available(self) -> bool:
        base_url, api_key, _source = _resolve_endpoint()
        return bool(base_url and api_key)

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": "provider-dependent",
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_API_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "OpenAI Compatible",
            "badge": "paid",
            "tag": "OpenAI-compatible /v1/images/generations endpoint (gpt-image-2)",
            "env_vars": [
                {
                    "key": "OPENAI_COMPAT_IMAGE_BASE_URL",
                    "prompt": "OpenAI-compatible image base URL (e.g. https://host/v1)",
                    "secret": False,
                },
                {
                    "key": "OPENAI_COMPAT_IMAGE_API_KEY",
                    "prompt": "OpenAI-compatible image API key",
                    "secret": True,
                },
            ],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider=PROVIDER_NAME,
                aspect_ratio=aspect,
            )

        base_url, api_key, source = _resolve_endpoint()
        if not base_url or not api_key:
            return error_response(
                error=(
                    "OpenAI-compatible image endpoint is not configured. Set "
                    "OPENAI_COMPAT_IMAGE_BASE_URL and OPENAI_COMPAT_IMAGE_API_KEY, "
                    "or configure image_gen.openai_compatible.{base_url,api_key}, "
                    "or set image_gen.openai_compatible.provider to a providers.<name> entry."
                ),
                error_type="auth_required",
                provider=PROVIDER_NAME,
                aspect_ratio=aspect,
            )

        model_id, meta = _resolve_model()
        size = str(kwargs.get("size") or _SIZES.get(aspect, _SIZES["square"]))
        quality = _resolve_quality(meta, kwargs.get("quality"))
        api_model = str(meta.get("api_model") or model_id or DEFAULT_API_MODEL)
        timeout_raw = _plugin_config().get("timeout_seconds", DEFAULT_TIMEOUT)
        try:
            timeout = float(timeout_raw)
        except (TypeError, ValueError):
            timeout = DEFAULT_TIMEOUT

        payload: Dict[str, Any] = {
            "model": api_model,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "n": 1,
        }
        # Optional knobs accepted by gpt-image-2/OpenAI-compatible proxies.
        for key in ("background", "output_format", "output_compression", "moderation"):
            value = kwargs.get(key)
            if value is not None:
                payload[key] = value

        try:
            response = _request_json(base_url, api_key, payload, timeout)
        except Exception as exc:  # noqa: BLE001
            logger.debug("OpenAI-compatible image generation failed", exc_info=True)
            return error_response(
                error=f"OpenAI-compatible image generation failed: {exc}",
                error_type="api_error",
                provider=PROVIDER_NAME,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        b64, url, revised_prompt = _extract_image_payload(response)
        prefix_model = api_model.replace("/", "_")
        if b64:
            try:
                # Validate enough to fail early on JSON text accidentally stored in result.
                base64.b64decode(b64, validate=False)
                saved_path = save_b64_image(b64, prefix=f"openai_compat_{prefix_model}_{quality}")
                image_ref = str(saved_path)
            except Exception as exc:  # noqa: BLE001
                return error_response(
                    error=f"Could not save base64 image to cache: {exc}",
                    error_type="io_error",
                    provider=PROVIDER_NAME,
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
        elif url:
            try:
                saved_path = save_url_image(url, prefix=f"openai_compat_{prefix_model}_{quality}")
                image_ref = str(saved_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not cache image URL %s (%s); returning URL", url, exc)
                image_ref = url
        else:
            return error_response(
                error="Response contained neither data[0].b64_json nor data[0].url",
                error_type="empty_response",
                provider=PROVIDER_NAME,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        extra: Dict[str, Any] = {
            "api_model": api_model,
            "size": size,
            "quality": quality,
            "endpoint_source": source,
        }
        if meta.get("legacy_model_id"):
            extra["legacy_model_id"] = meta["legacy_model_id"]
        if revised_prompt:
            extra["revised_prompt"] = revised_prompt
        usage = response.get("usage")
        if isinstance(usage, dict):
            extra["usage"] = usage

        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider=PROVIDER_NAME,
            extra=extra,
        )


def register(ctx: Any) -> None:
    ctx.register_image_gen_provider(OpenAICompatibleImageGenProvider())
