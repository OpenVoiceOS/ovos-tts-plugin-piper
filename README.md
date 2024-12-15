## Description

[OpenVoiceOS (OVOS)](https://openvoiceos.org) TTS plugin for [piper](https://github.com/rhasspy/piper)

## Install

`pip install ovos-tts-plugin-piper`

## Configuration

voice models are automatically downloaded from https://huggingface.co/rhasspy/piper-voices into `~/.local/share/piper_tts`

full list of voices can be found [here](https://huggingface.co/rhasspy/piper-voices/blob/main/voices.json)

you can also pass a short name alias without lang code, eg `"alan-low"` instead of `"en_GB-alan-low"`

```json
  "tts": {
    "module": "ovos-tts-plugin-piper",
    "ovos-tts-plugin-piper": {
      "voice": "alan-low"
    }
  }
```
if no voice is set it will be auto selected based on language
