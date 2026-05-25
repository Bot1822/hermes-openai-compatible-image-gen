# OpenAI-compatible Image Generation for Hermes Agent

Hermes Agent `image_gen` provider plugin for OpenAI-compatible image endpoints such as NewAPI, one-api, LiteLLM, or custom proxy servers that expose OpenAI-style image generation.

It calls:

```http
POST {base_url}/images/generations
```

with payloads like:

```json
{
  "model": "gpt-image-2",
  "prompt": "...",
  "size": "1024x1024",
  "quality": "low",
  "n": 1
}
```

The returned image (`data[0].b64_json` or `data[0].url`) is saved under `$HERMES_HOME/cache/images/`, and Hermes receives a local file path.

## Install

```bash
hermes plugins install Bot1822/hermes-openai-compatible-image-gen --enable
```

Then restart the CLI/gateway session so plugins reload:

```bash
# CLI: exit and reopen Hermes, or start a new session
# Gateway:
hermes gateway restart
```

## Configure

Choose either Option A or Option B.

### Option A: dedicated env vars

Add to `~/.hermes/.env`:

```bash
OPENAI_COMPAT_IMAGE_BASE_URL=https://your-openai-compatible-endpoint.example/v1
OPENAI_COMPAT_IMAGE_API_KEY=your-api-key
# Optional overrides:
# OPENAI_COMPAT_IMAGE_MODEL=gpt-image-2
# OPENAI_COMPAT_IMAGE_QUALITY=low
```

Then set Hermes image generation provider:

```bash
hermes config set image_gen.provider openai-compatible
hermes config set image_gen.model gpt-image-2
```

### Option B: reuse a Hermes LLM provider from `providers.<name>`

If your `~/.hermes/config.yaml` already has an OpenAI-compatible provider:

```yaml
providers:
  proxy:
    api: https://your-openai-compatible-endpoint.example/v1
    api_key: sk-...
```

Then configure image generation to reuse it:

```yaml
image_gen:
  provider: openai-compatible
  model: gpt-image-2
  openai_compatible:
    provider: proxy
    quality: low
```

This avoids duplicating the key in `.env`.

## Usage

After installation and configuration, enable the Hermes `image_gen` toolset for the platform/session where you want image generation. Then ask Hermes normally, for example:

```text
生成一张白色背景上的红苹果，扁平插画风格
```

Hermes will call its native `image_generate` tool, route the request to this provider, save the generated PNG/WebP/JPEG under `$HERMES_HOME/cache/images/`, and return/deliver the resulting image path or media attachment depending on platform support.

For a direct smoke test outside a full chat session:

```bash
cd ~/.hermes/plugins/openai-compatible-image-gen
PYTHONPATH=~/.hermes/hermes-agent python3 scripts/smoke_test.py
```

## Model and quality configuration

The clean configuration is: **model id stays the actual API model**, and **quality is a separate field**.

Recommended default:

```yaml
image_gen:
  provider: openai-compatible
  model: gpt-image-2
  openai_compatible:
    provider: proxy
    quality: low
```

Switch quality without changing model id:

```yaml
image_gen:
  provider: openai-compatible
  model: gpt-image-2
  openai_compatible:
    provider: proxy
    quality: medium  # low | medium | high | auto
```

Or with env vars:

```bash
OPENAI_COMPAT_IMAGE_MODEL=gpt-image-2
OPENAI_COMPAT_IMAGE_QUALITY=medium
```

Quality selection precedence:

1. explicit tool/provider `quality` kwarg;
2. `OPENAI_COMPAT_IMAGE_QUALITY` env var;
3. `image_gen.openai_compatible.quality` in `config.yaml`;
4. legacy tier model suffix, if configured;
5. `low` default.

### Backward compatibility

Older configs like these still work:

- `gpt-image-2-low`
- `gpt-image-2-medium`
- `gpt-image-2-high`

They are treated as compatibility aliases only. New configs should prefer:

```yaml
model: gpt-image-2
openai_compatible:
  quality: medium
```

## Why the default quality is `low`

`low` is the default because it is the safest operational choice for Hermes:

- lower cost/token usage for accidental or exploratory tool calls;
- lower latency for chat/gateway workflows;
- enough quality to verify that the provider and delivery path work;
- easy to upgrade explicitly to `medium` or `high` when fidelity matters.

If you prefer quality over cost/latency, set `image_gen.openai_compatible.quality` to `medium` or `high`.

## Why direct Images API, not Responses image_generation tool?

Hermes already provides a native `image_generate` tool. For normal image generation, the direct Images API is simpler, cheaper, more deterministic, and maps cleanly to Hermes' `ImageGenProvider` abstraction.

Responses API with `tools: [{type: "image_generation"}]` is better reserved for future multi-turn image editing flows.
