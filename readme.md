## Description

OVOS TTS plugin for [piper](https://github.com/rhasspy/piper)

## Install

`pip install ovos-tts-plugin-piper`

## Configuration

download models from https://github.com/rhasspy/piper/releases/tag/v0.0.2

```json
  "tts": {
    "module": "ovos-tts-plugin-piper",
    "ovos-tts-plugin-piper": {
      "model": "path/model.onnx",
    }
  }
```