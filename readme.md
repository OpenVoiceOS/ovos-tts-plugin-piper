## Description

OVOS TTS plugin for [piper](https://github.com/rhasspy/piper)

## Install

`pip install ovos-tts-plugin-piper`

## Configuration

Models can be loaded from the ones built-in to Piper, a list of pre-defined downloadable voices, or from the devices local storage.
Models are stored in `$XDG_HOME/piper_tts/$model_name"` and consist of a .onnx and .json file, ex:
```bash
ls -1 $XDG_HOME/piper_tts/example/
example.onnx
example.onnx.json
```

Available aliases can be found in [this list](https://github.com/OpenVoiceOS/ovos-tts-plugin-piper/blob/dev/ovos_tts_plugin_piper/__init__.py#L154)
A list of downloadable models can be found [here](https://github.com/rhasspy/piper/releases/tag/v0.0.2) or [here](https://huggingface.co/rhasspy/piper-voices/tree/main), to use one just link to the .onnx file in the `voice` parameter of the configuration

Passed URLs can be to a .onnx file which contains an appropriately named .json definition file in the same location, or to a .tar.gz archive containing the files

if no model is passed it will be auto selected based on language

```json
  "tts": {
    "module": "ovos-tts-plugin-piper",
    "ovos-tts-plugin-piper": {
      "voice": "alan-low"
    }
  }
```


