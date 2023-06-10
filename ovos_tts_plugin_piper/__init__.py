# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import tarfile

import requests
from ovos_plugin_manager.templates.g2p import Grapheme2PhonemePlugin
from ovos_plugin_manager.templates.tts import TTS
from ovos_utils.log import LOG
from ovos_utils.xdg_utils import xdg_data_home

from ovos_tts_plugin_piper.engine import Piper


class PiperG2P(Grapheme2PhonemePlugin):
    piper = None

    def __init__(self, piper_engine, *args, **kwargs):
        self.piper = piper_engine
        super().__init__(*args, **kwargs)

    def get_ipa(self, word, lang=None, ignore_oov=False):
        return self.piper.phonemizer.phonemize(word, phoneme_separator=" ").split(" ")


class PiperTTSPlugin(TTS):
    """Interface to Piper TTS."""
    lang2voices = {
        'ca': ['upc_ona-x-low', 'upc_pau-x-low'],
        'da': ['nst_talesyntese-medium'],
        'de': ['eva_k-x-low',
               'karlsson-low',
               'kerstin-low',
               'pavoque-low',
               'ramona-low',
               'thorsten-low'],
        'el-gr': ['rapunzelina-low'],
        'en-gb': ['alan-low', 'southern_english_female-low'],
        'en-us': ['amy-low',
                  'danny-low',
                  'kathleen-low',
                  'lessac-low',
                  'lessac-medium',
                  'libritts-high',
                  'ryan-high',
                  'ryan-low',
                  'ryan-medium',
                  'lessac'],
        'es': ['carlfm-x-low', 'mls_10246-low', 'mls_9972-low'],
        'fi': ['harri-low'],
        'fr': ['gilles-low', 'mls_1840-low', 'siwis-low', 'siwis-medium'],
        'it': ['riccardo_fasol-x-low'],
        'kk': ['iseke-x-low', 'issai-high', 'raya-x-low'],
        'ne': ['google-medium', 'google-x-low'],
        'nl': ['mls_5809-low',
               'mls_7432-low',
               'nathalie-x-low',
               'rdh-medium',
               'rdh-x-low'],
        'no': ['talesyntese-medium'],
        'pl': ['mls_6892-low'],
        'pt-br': ['edresson-low'],
        'uk': ['lada-x-low'],
        'vi': ['25hours-single-low', 'vos-x-low'],
        'zh-cn': ['huayan-x-low']}
    voice2url = {
        '25hours-single-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-vi-25hours-single-low.tar.gz',
        'alan-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-gb-alan-low.tar.gz',
        'amy-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-amy-low.tar.gz',
        'carlfm-x-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-es-carlfm-x-low.tar.gz',
        'danny-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-danny-low.tar.gz',
        'edresson-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-pt-br-edresson-low.tar.gz',
        'eva_k-x-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-de-eva_k-x-low.tar.gz',
        'gilles-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-fr-gilles-low.tar.gz',
        'google-medium': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-ne-google-medium.tar.gz',
        'google-x-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-ne-google-x-low.tar.gz',
        'harri-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-fi-harri-low.tar.gz',
        'huayan-x-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-zh-cn-huayan-x-low.tar.gz',
        'iseke-x-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-kk-iseke-x-low.tar.gz',
        'issai-high': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-kk-issai-high.tar.gz',
        'karlsson-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-de-karlsson-low.tar.gz',
        'kathleen-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-kathleen-low.tar.gz',
        'kerstin-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-de-kerstin-low.tar.gz',
        'lada-x-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-uk-lada-x-low.tar.gz',
        'lessac': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us_lessac.tar.gz',
        'lessac-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-lessac-low.tar.gz',
        'lessac-medium': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-lessac-medium.tar.gz',
        'libritts-high': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-libritts-high.tar.gz',
        'mls_10246-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-es-mls_10246-low.tar.gz',
        'mls_1840-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-fr-mls_1840-low.tar.gz',
        'mls_5809-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-nl-mls_5809-low.tar.gz',
        'mls_6892-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-pl-mls_6892-low.tar.gz',
        'mls_7432-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-nl-mls_7432-low.tar.gz',
        'mls_9972-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-es-mls_9972-low.tar.gz',
        'nathalie-x-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-nl-nathalie-x-low.tar.gz',
        'nst_talesyntese-medium': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-da-nst_talesyntese-medium.tar.gz',
        'pavoque-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-de-pavoque-low.tar.gz',
        'ramona-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-de-ramona-low.tar.gz',
        'rapunzelina-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-el-gr-rapunzelina-low.tar.gz',
        'raya-x-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-kk-raya-x-low.tar.gz',
        'rdh-medium': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-nl-rdh-medium.tar.gz',
        'rdh-x-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-nl-rdh-x-low.tar.gz',
        'riccardo_fasol-x-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-it-riccardo_fasol-x-low.tar.gz',
        'ryan-high': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-ryan-high.tar.gz',
        'ryan-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-ryan-low.tar.gz',
        'ryan-medium': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-ryan-medium.tar.gz',
        'siwis-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-fr-siwis-low.tar.gz',
        'siwis-medium': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-fr-siwis-medium.tar.gz',
        'southern_english_female-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-gb-southern_english_female-low.tar.gz',
        'talesyntese-medium': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-no-talesyntese-medium.tar.gz',
        'thorsten-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-de-thorsten-low.tar.gz',
        'upc_ona-x-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-ca-upc_ona-x-low.tar.gz',
        'upc_pau-x-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-ca-upc_pau-x-low.tar.gz',
        'vos-x-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-vi-vivos-x-low.tar.gz'
    }

    def __init__(self, lang="en-us", config=None):
        super(PiperTTSPlugin, self).__init__(lang, config)
        model = self.config.get("model") or \
                self.lang2voices.get(lang) or \
                self.lang2voices.get(lang.split("-"[0]))
        if isinstance(model, list):
            model = model[0]

        if not os.path.isfile(model) and model in self.voice2url:
            xdg_p = f"{xdg_data_home()}/piper_tts/{model}"
            if not os.path.isdir(xdg_p):
                model = self.voice2url[model]

        if model.startswith("http"):
            m = model.split("/")[-1]
            xdg_p = f"{xdg_data_home()}/piper_tts/{m.split('.')[0]}"

            model_tar = f"{xdg_p}/{m}"
            if not os.path.isfile(model_tar):
                LOG.info(f"downloading piper model: {model}")
                os.makedirs(xdg_p, exist_ok=True)
                data = requests.get(model)
                with open(model_tar, "wb") as f:
                    f.write(data.content)

                with tarfile.open(model_tar) as file:
                    file.extractall(xdg_p)
            for f in os.listdir(xdg_p):
                if f.endswith(".onnx"):
                    model = f"{xdg_p}/{f}"
                    LOG.debug(f"selected model: {model}")
                    break
            else:
                raise FileNotFoundError("onnx model not found")

        self.model = model
        self.model_json = self.config.get("model_json") or self.model + ".json"
        self.use_cuda = self.config.get("use_cuda", False)
        self.engine = Piper(self.model, config_path=self.model_json, use_cuda=self.use_cuda)

        self.speaker = self.config.get("speaker")  # "Id of speaker (default: 0)"
        self.noise_scale = self.config.get("noise-scale")  # generator noise
        self.length_scale = self.config.get("length-scale")  # Phoneme length
        self.noise_w = self.config.get("noise-w")  # Phoneme width noise
        self.g2p = PiperG2P(self.engine)

    def get_tts(self, sentence, wav_file, lang=None, speaker=None):
        """Generate WAV and phonemes.

        Arguments:
            sentence (str): sentence to generate audio for
            wav_file (str): output file
            lang (str): optional lang override
            speaker (int): optional speaker override

        Returns:
            tuple ((str) file location, (str) generated phonemes)
        """
        lang = lang or self.lang
        # HACK: bug in some neon-core versions - neon_audio.tts.neon:_get_tts:198 - INFO - Legacy Neon TTS signature found 
        if isinstance(speaker, dict):
            LOG.warning("Legacy Neon TTS signature found, pass speaker as a str")
            speaker = None
        wav_bytes = self.engine.synthesize(sentence,
                                           speaker_id=speaker or self.speaker,
                                           length_scale=self.length_scale,
                                           noise_scale=self.noise_scale,
                                           noise_w=self.noise_w)
        with open(wav_file, "wb") as f:
            f.write(wav_bytes)

        phonemes = self.g2p.utterance2arpa(sentence, lang, ignore_oov=True)
        phonemes = " ".join([f"{p}:0.4" for p in phonemes])

        return wav_file, phonemes

    def available_languages(self) -> set:
        return set(self.lang2voices.keys())


PiperTTSPluginConfig = {
    lang: [{v: {"model": PiperTTSPlugin.voice2url[v], "offline": True}}
           for v in voices]
    for lang, voices in PiperTTSPlugin.lang2voices.items()
}

if __name__ == "__main__":
    config = {}
    config["model"] = "alan-low"
    e = PiperTTSPlugin(config=config)
    e.get_tts( "one oh clock", "hello.wav")
