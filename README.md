# hermes-murf-plugin

Murf TTS for [Hermes](https://github.com/NousResearch/hermes-agent). Streams audio over Murf's Falcon WebSocket API with sub-100ms time-to-first-audio, plus a persistent session mode for real-time, sentence-by-sentence voice agents.

[![PyPI](https://img.shields.io/pypi/v/hermes-plugin-murf-tts.svg)](https://pypi.org/project/hermes-plugin-murf-tts/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hermes Plugin](https://img.shields.io/badge/Hermes-plugin-blue)](https://github.com/NousResearch/hermes-agent)

## Why this exists

Hermes ships with a `TTSProvider` interface but no first-party Murf backend. This plugin fills that gap and adds one thing most TTS integrations skip: a **persistent connection mode**. The naive approach opens a new WebSocket per sentence, paying a full handshake every time. For a voice agent speaking sentence-by-sentence as an LLM generates text, that's the difference between a natural reply and a choppy one. `open_live_session()` keeps one connection open for the whole turn instead.

## Quickstart

```bash
hermes plugins install sanchitasunil/hermes-murf-plugin
pip install 'websockets>=12'
```

That's it. `hermes plugins install` clones this repo and drops it into `~/.hermes/plugins/murf-tts` for you, and prompts for `MURF_API_KEY` on the spot (get one at [murf.ai/api](https://murf.ai/api)) if it isn't already set.

**Alternative: install via pip**

```bash
pip install hermes-plugin-murf-tts
```

Hermes auto-discovers it on next startup through the `hermes_agent.plugins` entry point, no manual enable step needed. Use this if you're managing Hermes plugins as regular Python dependencies (a requirements file, a Docker image, etc.) instead of through `hermes plugins install`.

Then in `~/.hermes/config.yaml`:

```yaml
tts:
  provider: murf
```

Pick a voice with `list_voices()`, or just set a default:

```bash
export MURF_DEFAULT_VOICE=en-US-terrell
```

## Features

- **Streaming synthesis** `stream()`, a one-shot generator matching the `TTSProvider` ABC contract.
- **Persistent live sessions** `open_live_session()` / `MurfLiveSession`, one connection for an entire multi-sentence reply instead of one per sentence.
- **Correct voice catalog lookups** `list_voices()` filters by model. Falcon and Gen2 have separate catalogs on Murf's side; skip the filter and you silently lose real voices.
- **Batch fallback** `synthesize()` writes a complete file for callers that don't stream.

## Usage

For normal use you don't touch these directly. Set `tts: provider: murf` in your config and Hermes instantiates the provider and calls it for you. The examples below show the API surface for anyone building on top of it, e.g. `from hermes_murf_plugin import MurfTTSProvider`. (`play()` here is a stand-in for however you send audio to your output.)

**One-shot streaming:**

```python
from hermes_murf_plugin import MurfTTSProvider

provider = MurfTTSProvider()
for chunk in provider.stream("Hello there!", voice="en-US-terrell"):
    play(chunk)
```

**Persistent session** (use this when text arrives incrementally, e.g. sentence-by-sentence from an LLM):

```python
session = provider.open_live_session(voice="en-US-terrell")
session.send_text("Hey, thanks for waiting.")
session.send_text("Here's what I found.", end=True)  # end=True closes the context, not the connection
for chunk in session.iter_audio():
    play(chunk)
session.close()
```

One thread calls `send_text()`, one thread iterates `iter_audio()`. Same one-sender/one-receiver rule `websockets.sync` itself enforces; don't call either method from two threads at once.

## Configuration

| Env var | Required | Purpose |
|---|---|---|
| `MURF_API_KEY` | Yes | Your Murf API key ([get one here](https://murf.ai/api)) |
| `MURF_DEFAULT_VOICE` | No | Fallback voice ID when none is passed per call |
| `MURF_TTS_MODEL` | No | `falcon-2` (default) or `gen2` |
| `MURF_API_BASE_URL` | No | Override the REST base URL (default `https://api.murf.ai`) |

## Verified in production

This has been running behind a live Discord voice bot doing sentence-by-sentence streaming replies, live voice switching, and multi-speaker input. These are confirmed from real usage, not just read off the docs:

| Claim | What we saw |
|---|---|
| `GET /v1/speech/voices?model=FALCON` (or `GEN2`) | Required. Without the `model` param, the endpoint returns an incomplete catalog. Confirmed missing real voices (`en-IN-palak`, `en-IN-anisha`) without it. |
| Context-id multiplexing | Several `{"text": ..., "context_id": same_id}` messages, `end: true` only on the last, closes that *context*. The connection itself stays open. Confirmed safe to reuse across a whole multi-sentence reply. |
| `websockets.sync.client` under a threaded caller | Works cleanly with a dedicated sender thread and reader thread. |
| `stream()` vs `open_live_session()` | `stream()` reconnects every call, fine for one utterance. For a multi-sentence reply, that's a full handshake per sentence. The persistent session removes that cost entirely, and it's the difference that made real-time replies sound natural instead of choppy. |

## References

- [Stream input reference](https://murf.ai/api/docs/api-reference/text-to-speech/stream-input)
- [WebSockets overview](https://murf.ai/api/docs/text-to-speech/web-sockets)
- [Context ID](https://murf.ai/api/docs/text-to-speech/web-sockets/context-id)
- [Falcon 2 model docs](https://murf.ai/api/docs/text-to-speech-models/falcon-2)

## License

MIT. See [LICENSE](LICENSE).
