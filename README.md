> **This repository is archived. No further updates will be made.**
>
> Piper support has been absorbed into [phoonnx](https://github.com/TigreGotico/phoonnx), a unified ONNX TTS plugin that supports Piper, Matcha, GlowTTS, MMS, and more — all from a single install. Migrate using the guide below.

# ovos-tts-plugin-piper → phoonnx migration

## Install

```bash
pip install phoonnx
```

`espeak-ng` must be available in your base OS (install via your distro package manager) — same requirement as before.

## Configuration mapping

Every Piper voice is available in phoonnx under the same name. Replace the plugin module and key:

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

The voice id format is `piper/<lang_code>-<name>` — matching the [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices/blob/main/voices.json) index. Short aliases without lang code also work: `"piper/alan-low"`.

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

Full list of available Piper voices: [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices/blob/main/voices.json)

phoonnx also includes Matcha, GlowTTS, MMS, and community voices — see [TigreGotico/phoonnx](https://github.com/TigreGotico/phoonnx) for the complete catalogue.

## Credits

Original plugin by the OpenVoiceOS community. Piper TTS engine by [rhasspy](https://github.com/rhasspy/piper).

> This plugin was funded by the Ministerio para la Transformación Digital y de la Función Pública and Plan de Recuperación, Transformación y Resiliencia - Funded by EU – NextGenerationEU within the framework of the project ILENIA with reference 2022/TL22/00215337
