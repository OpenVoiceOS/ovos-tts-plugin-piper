> **This repository is archived. No further updates will be made.**
>
> Piper support has been absorbed into [phoonnx](https://github.com/TigreGotico/phoonnx), a unified ONNX TTS plugin that supports Piper, Matcha, GlowTTS, MMS, and more. Migrate using the guide below.

# ovos-tts-plugin-piper → phoonnx migration

## Install

```bash
pip install phoonnx
```

`espeak-ng` must be available in your base OS (install via your distro package manager) — same requirement as before.

## Configuration mapping

Replace the plugin module and key. Voice IDs use the `piper/<lang_code>-<name>` format:

**Before:**
```json
"tts": {
  "module": "ovos-tts-plugin-piper",
  "ovos-tts-plugin-piper": {
    "voice": "alan-low"
  }
}
```

**After:**
```json
"tts": {
  "module": "phoonnx",
  "phoonnx": {
    "voice": "piper/en_GB-alan-low"
  }
}
```

### Local model path

```json
"tts": {
  "module": "phoonnx",
  "phoonnx": {
    "model": "/path/to/model.onnx",
    "model_config": "/path/to/model.onnx.json"
  }
}
```

### Remote URL

```json
"tts": {
  "module": "phoonnx",
  "phoonnx": {
    "model": "https://huggingface.co/poisson-fish/piper-vasco/resolve/main/onnx/vasco.onnx",
    "model_config": "https://huggingface.co/poisson-fish/piper-vasco/resolve/main/onnx/vasco.onnx.json"
  }
}
```

### Auto-select by language

Leave `voice` unset and phoonnx selects the best available voice for the configured language, including all Piper voices.

## Voice catalogue

All available Piper voices (and every other supported voice) are listed in [VOICES.md](https://github.com/TigreGotico/phoonnx/blob/dev/VOICES.md) — that is the canonical reference for voice IDs to use in your config.

## Credits

Original plugin by the OpenVoiceOS community. Piper TTS engine by [rhasspy](https://github.com/rhasspy/piper).
