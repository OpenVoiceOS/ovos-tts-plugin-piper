## Description

[OpenVoiceOS (OVOS)](https://openvoiceos.org) TTS plugin for [piper](https://github.com/rhasspy/piper)

## Install

`pip install ovos-tts-plugin-piper`

## Configuration

download models from https://github.com/rhasspy/piper/releases/tag/v0.0.2

you can also pass an url for a .tar.gz model, and it will be auto downloaded

if no model is passed it will be auto selected based on language

you can pass a model name alias, eg "alan-low"

```json
  "tts": {
    "module": "ovos-tts-plugin-piper",
    "ovos-tts-plugin-piper": {
      "voice": "alan-low"
    }
  }
```
