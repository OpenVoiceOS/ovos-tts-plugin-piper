## Description

OVOS TTS plugin for [piper](https://github.com/MycroftAI/piper1)

## Install

`pip install ovos-tts-plugin-piper`

you can either [compile](https://github.com/MycroftAI/mycroft-core/blob/dev/scripts/install-piper.sh) [piper](https://github.com/MycroftAI/piper1) or use [forslund's repo](https://forslund.github.io/mycroft-desktop-repo/)

## Configuration

if piper is available system wide you just need to specify a voice

```json
  "tts": {
    "module": "ovos-tts-plugin-piper",
    "ovos-tts-plugin-piper": {
      "voice": "ap",
    }
  }
```


### Advanced config

voices can be urls, file paths, or included voices, you can find compatible festvox voices [here](http://www.festvox.org/flite/packed/flite-2.0/voices/)

You can also specify the piper binary location

```json
  "tts": {
    "module": "ovos-tts-plugin-piper",
    "ovos-tts-plugin-piper": {
      "voice": "http://www.festvox.org/flite/packed/flite-2.0/voices/cmu_us_fem.flitevox",
      "binary": "~/piper1/piper"
    }
  }
        
```

Mycroft premium subscribers have access to a female voice called trinity
This voice is actually a pre compiled piper binary with the voice included

The plugin should automatically detect this binary and use it, but you can 
also specify the binary location directly

If you are a subscriber the voice should have been downloaded to `/opt/mycroft/voices/piper_tn`

```json
  "tts": {
    "module": "ovos-tts-plugin-piper",
    "ovos-tts-plugin-piper": {
      "voice": "trinity",
      "binary": "/opt/mycroft/voices/piper_tn"
    }
  }
        
```