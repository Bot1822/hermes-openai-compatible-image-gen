#!/usr/bin/env python3
"""Smoke-test the plugin outside a full Hermes chat session.

Run from the repository root with Hermes source on PYTHONPATH, for example:

    PYTHONPATH=$HOME/.hermes/hermes-agent python3 scripts/smoke_test.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "__init__.py"

spec = importlib.util.spec_from_file_location("openai_compatible_image_gen", PLUGIN)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

provider = module.OpenAICompatibleImageGenProvider()
print("provider:", provider.name)
print("available:", provider.is_available())
print("models:", [m["id"] for m in provider.list_models()])

if provider.is_available():
    result = provider.generate(
        "生成一张简单测试图：白色背景上一颗小红苹果，扁平插画风格。",
        aspect_ratio="square",
    )
    summary = dict(result)
    if isinstance(summary.get("image"), str):
        summary["image"] = summary["image"][:200]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
else:
    print("Set OPENAI_COMPAT_IMAGE_BASE_URL/API_KEY or image_gen.openai_compatible.provider first.")
