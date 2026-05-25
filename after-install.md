# Installed

The OpenAI-compatible image generation plugin is installed.

Recommended configuration:

```bash
hermes config set image_gen.provider openai-compatible
hermes config set image_gen.model gpt-image-2
```

Keep the model id as the real API model (`gpt-image-2`) and configure quality separately:

```yaml
image_gen:
  provider: openai-compatible
  model: gpt-image-2
  openai_compatible:
    provider: proxy
    quality: low
```

Use `quality: low` for the cheapest/fastest default, or switch to `medium` / `high` when image fidelity matters more than cost and latency.

Either set these in `~/.hermes/.env`:

```bash
OPENAI_COMPAT_IMAGE_BASE_URL=https://your-openai-compatible-endpoint.example/v1
OPENAI_COMPAT_IMAGE_API_KEY=***
```

or point the plugin at an existing Hermes provider in `~/.hermes/config.yaml` as shown above.

Restart Hermes CLI/gateway after enabling the plugin.
