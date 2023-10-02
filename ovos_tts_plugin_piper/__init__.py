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
import io
import os
import re
import tarfile
import wave

import requests
from ovos_plugin_manager.templates.g2p import Grapheme2PhonemePlugin
from ovos_plugin_manager.templates.tts import TTS
from ovos_tts_plugin_piper.engine import Piper
from ovos_utils.log import LOG
from ovos_utils.xdg_utils import xdg_data_home


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
        'ru': ['irina-medium',
               'denis-medium',
               'dmitri-medium',
               'ruslan-medium'],
        'uk': ['lada-x-low'],
        'vi': ['25hours-single-low', 'vos-x-low'],
        'zh-cn': ['huayan-x-low']}
    voice2url = {
        '25hours-single-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-vi-25hours-single-low.tar.gz',
        'alan-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-gb-alan-low.tar.gz',
        'amy-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-amy-low.tar.gz',
        'carlfm-x-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-es-carlfm-x-low.tar.gz',
        'danny-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-danny-low.tar.gz',
        'denis-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/denis/medium/ru_RU-denis-medium.onnx',
        'dmitri-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx',
        'edresson-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-pt-br-edresson-low.tar.gz',
        'eva_k-x-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-de-eva_k-x-low.tar.gz',
        'gilles-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-fr-gilles-low.tar.gz',
        'google-medium': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-ne-google-medium.tar.gz',
        'google-x-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-ne-google-x-low.tar.gz',
        'harri-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-fi-harri-low.tar.gz',
        'huayan-x-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-zh-cn-huayan-x-low.tar.gz',
        'irina-medium': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-ru-irinia-medium.tar.gz',
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
        'ruslan-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/ruslan/medium/ru_RU-ruslan-medium.onnx',
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
    engines = {}

    def __init__(self, lang="en-us", config=None):
        super(PiperTTSPlugin, self).__init__(lang, config)
        if self.voice == "default":
            if self.lang.startswith("en"):
                # alan pope is the default english voice of mycroft/OVOS
                self.voice = "alan-low"
            else:
                self.voice = self.lang2voices.get(lang) or \
                             self.lang2voices.get(lang.split("-")[0]) or \
                             "alan-low"

        if isinstance(self.voice, list):
            self.voice = self.voice[0]

        self.use_cuda = self.config.get("use_cuda", False)
        self.noise_scale = self.config.get("noise-scale")  # generator noise
        self.length_scale = self.config.get("length-scale")  # Phoneme length
        self.noise_w = self.config.get("noise-w")  # Phoneme width noise

        # pre-load models
        preload_voices = self.config.get("preload_voices") or []
        preload_langs = self.config.get("preload_langs") or [self.lang]

        for lang in preload_langs:
            if lang not in self.lang2voices:
                lang = lang.split("-")[0]
            voice = self.lang2voices.get(lang)
            if voice and voice not in preload_voices:
                preload_voices.append(voice)

        for voice in preload_voices:
            self.get_model(voice=voice)

    def get_model(self, lang=None, voice=None, speaker=None):

        # find default voice  (should be called model not voice....)
        if voice is None and lang is not None:
            if lang.startswith("en"):
                # alan pope is the default english voice of mycroft/OVOS
                voice = "alan-low"
            else:
                voice = self.lang2voices.get(lang) or \
                        self.lang2voices.get(lang.split("-")[0])

        voice = voice or self.voice

        if isinstance(voice, list):
            voice = voice[0]

        # find speaker  (should be called voice not speaker...)
        if "#" in voice:
            voice, speaker2 = voice.split("#")
            try:
                speaker2 = int(speaker2)
                if speaker is not None:
                    LOG.warning("requested voice and speaker args conflict, "
                                "ignoring requested speaker in favor of speaker defined in voice string")
                    speaker = speaker2
            except:
                LOG.warning("invalid speaker requested in voice string, ignoring it")

        speaker = speaker or 0  # default for this model

        # pre-loaded models
        if voice in PiperTTSPlugin.engines:
            return PiperTTSPlugin.engines[voice], speaker

        # find requested voice
        if voice in self.voice2url:
            xdg_p = f"{xdg_data_home()}/piper_tts/{voice}"
            if not os.path.isdir(xdg_p):
                url = self.voice2url[voice]

                m = url.split("/")[-1]
                xdg_p = f"{xdg_data_home()}/piper_tts/{m.split('.')[0]}"

                model_file = f"{xdg_p}/{m}"
                if not os.path.isfile(model_file):
                    LOG.info(f"downloading piper model: {url}")
                    os.makedirs(xdg_p, exist_ok=True)
                    # TODO - streaming download
                    data = requests.get(url)
                    with open(model_file, "wb") as f:
                        f.write(data.content)

                    if url.endswith(".onnx"):
                        json_data = requests.get(url + '.json')
                        with open(model_file + '.json', "wb") as f:
                            f.write(json_data.content)
                    else:
                        with tarfile.open(model_file) as file:
                            file.extractall(xdg_p)

                for f in os.listdir(xdg_p):
                    if f.endswith(".onnx"):
                        model = f"{xdg_p}/{f}"
                        engine = Piper(model, config_path=model + ".json", use_cuda=self.use_cuda)
                        LOG.debug(f"loaded model: {model}")
                        PiperTTSPlugin.engines[voice] = engine
                        return engine, speaker
                else:
                    raise FileNotFoundError("onnx model not found")
        else:
            raise ValueError(f"invalid voice: {voice}")

    def _piper_synth(self, text: str, lang: str, voice: str, speaker: int) -> bytes:
        """Synthesize audio from text and return WAV bytes"""

        engine, speaker = self.get_model(lang, voice, speaker)

        sents = re.split('(?<=[.!?]) +', text)

        results = [engine.synthesize(sentence,
                                     speaker_id=speaker,
                                     length_scale=self.length_scale,
                                     noise_scale=self.noise_scale,
                                     noise_w=self.noise_w)
                   for sentence in [text]]

        with io.BytesIO() as wav_io:
            wav_file: wave.Wave_write = wave.open(wav_io, "wb")
            wav_file.setframerate(engine.config.sample_rate)
            wav_file.setsampwidth(2)
            wav_file.setnchannels(1)

            with wav_file:
                for audio_bytes in results:
                    # Add audio to existing WAV file
                    wav_file.writeframes(audio_bytes)
            wav_bytes = wav_io.getvalue()

        return wav_bytes

    def get_tts(self, sentence, wav_file, lang=None, voice=None, speaker=None):
        """Generate WAV and phonemes.

        Arguments:
            sentence (str): sentence to generate audio for
            wav_file (str): output file
            lang (str): optional lang override
            voice (str): optional voice override
            speaker (int): optional speaker override

        Returns:
            tuple ((str) file location, (str) generated phonemes)
        """
        lang = lang or self.lang
        # HACK: bug in some neon-core versions - neon_audio.tts.neon:_get_tts:198 - INFO - Legacy Neon TTS signature found
        if isinstance(speaker, dict):
            LOG.warning("Legacy Neon TTS signature found, pass speaker as a str")
            speaker = None

        wav_bytes = self._piper_synth(sentence, lang, voice, speaker)
        with open(wav_file, "wb") as f:
            f.write(wav_bytes)

        return wav_file, None

    def available_languages(self) -> set:
        return set(self.lang2voices.keys())


PiperTTSPluginConfig = {
    lang: [{v: {"model": PiperTTSPlugin.voice2url[v], "offline": True}}
           for v in voices]
    for lang, voices in PiperTTSPlugin.lang2voices.items()
}

if __name__ == "__main__":
    config = {}
    config["lang"] = "en-us"
    e = PiperTTSPlugin(config=config)
    e.get_tts("one oh clock? yes! it is one on the clock", "hello.wav")
    e.get_tts("one oh clock", "libritts-high.wav", voice="libritts-high")
