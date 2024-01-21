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
import json
import os
import tarfile
import wave
from typing import List, Optional

import numpy as np
import onnxruntime
import requests
from ovos_plugin_manager.templates.tts import TTS
from ovos_utils.log import LOG
from ovos_utils.xdg_utils import xdg_data_home
from piper.config import PiperConfig
from piper.voice import PiperVoice as _PV
from piper.util import audio_float_to_int16


class PiperVoice(_PV):

    # working around upstream bug in single speaker models
    def synthesize_ids_to_raw(
            self,
            phoneme_ids: List[int],
            speaker_id: Optional[int] = None,
            length_scale: Optional[float] = None,
            noise_scale: Optional[float] = None,
            noise_w: Optional[float] = None,
    ) -> bytes:
        """Synthesize raw audio from phoneme ids."""
        if length_scale is None:
            length_scale = self.config.length_scale

        if noise_scale is None:
            noise_scale = self.config.noise_scale

        if noise_w is None:
            noise_w = self.config.noise_w

        phoneme_ids_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        phoneme_ids_lengths = np.array([phoneme_ids_array.shape[1]], dtype=np.int64)
        scales = np.array(
            [noise_scale, length_scale, noise_w],
            dtype=np.float32,
        )

        args = {
            "input": phoneme_ids_array,
            "input_lengths": phoneme_ids_lengths,
            "scales": scales
        }

        if self.config.num_speakers <= 1:
            speaker_id = None

        if (self.config.num_speakers > 1) and (speaker_id is None):
            # Default speaker
            speaker_id = 0

        if speaker_id is not None:
            sid = np.array([speaker_id], dtype=np.int64)
            args["sid"] = sid  # <- this is the bug fix, upstream passes "sid": None to args
            # which crashes single speaker models

        # Synthesize through Onnx
        audio = self.session.run(None, args, )[0].squeeze((0, 1))
        audio = audio_float_to_int16(audio.squeeze())
        return audio.tobytes()


class PiperTTSPlugin(TTS):
    """Interface to Piper TTS."""
    lang2voices = {
        'ar': ['kareem-medium'],
        'ca': ['upc_ona-x-low', 'upc_pau-x-low'],
        'cs': ['jirka-low', 'jirka-medium'],
        'da': ['nst_talesyntese-medium'],
        'de': ['eva_k-x-low',
               'karlsson-low',
               'kerstin-low',
               'pavoque-low',
               'ramona-low',
               'thorsten-high',
               'thorsten-low',
               'thorsten-medium',
               'thorsten_emotional-medium'],
        'el': ['rapunzelina-low'],
        'en-gb': ['alan-low', 'alba-medium', 'southern_english_female-low'],
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
        'es': ['ald-medium',
               'carlfm-x-low',
               'davefx-medium',
               'mls_10246-low',
               'mls_9972-low',
               'sharvard-medium'],
        'fi': ['harri-low', 'harri-medium'],
        'fr': ['gilles-low',
               'mls_1840-low',
               'siwis-low',
               'siwis-medium',
               'upmc-medium'],
        'hu': ['anna-medium', 'berta-medium', 'imre-medium'],
        'is': ['bui-medium', 'salka-medium', 'steinn-medium', 'ugla-medium'],
        'it': ['riccardo_fasol-x-low'],
        'ka': ['natia-medium'],
        'kk': ['iseke-x-low', 'issai-high', 'raya-x-low'],
        'lb': ['marylux-medium'],
        'ne': ['google-medium', 'google-x-low'],
        'nl': ['mls_5809-low',
               'mls_7432-low',
               'nathalie-x-low',
               'rdh-medium',
               'rdh-x-low'],
        'no': ['talesyntese-medium'],
        'pl': ['darkman-medium',
               'gosia-medium',
               'mc_speech-medium',
               'mls_6892-low'],
        'pt-br': ['edresson-low', 'faber-medium'],
        'pt': ['tugao-medium'],
        'ro': ['mihai-medium'],
        'ru': ['irina-medium',
               'denis-medium',
               'dmitri-medium',
               'ruslan-medium'],
        'sk': ['lili-medium'],
        'sr': ['serbski_institut-medium'],
        'sv': ['nst-medium'],
        'sw': ['lanfrica-medium'],
        'tr': ['dfki-medium', 'fahrettin-medium'],
        'uk': ['lada-x-low', 'ukrainian_tts-medium'],
        'vi': ['25hours-single-low', 'vais1000-medium', 'vos-x-low'],
        'zh-cn': ['huayan-x-low', 'huayan-medium']}
    voice2url = {
        '25hours-single-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-vi-25hours-single-low.tar.gz',
        'alan-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-gb-alan-low.tar.gz',
        'alba-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alba/medium/en_GB-alba-medium.onnx',
        'ald-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/ald/medium/es_MX-ald-medium.onnx',
        'amy-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-amy-low.tar.gz',
        'anna-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/hu/hu_HU/anna/medium/hu_HU-anna-medium.onnx',
        'berta-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/hu/hu_HU/berta/medium/hu_HU-berta-medium.onnx',
        'bui-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/is/is_IS/bui/medium/is_IS-bui-medium.onnx',
        'carlfm-x-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-es-carlfm-x-low.tar.gz',
        'danny-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-danny-low.tar.gz',
        'darkman-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/darkman/medium/pl_PL-darkman-medium.onnx',
        'davefx-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx',
        'denis-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/denis/medium/ru_RU-denis-medium.onnx',
        'dfki-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/tr/tr_TR/dfki/medium/tr_TR-dfki-medium.onnx',
        'dmitri-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx',
        'edresson-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-pt-br-edresson-low.tar.gz',
        'eva_k-x-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-de-eva_k-x-low.tar.gz',
        'faber-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx',
        'fahrettin-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/tr/tr_TR/fahrettin/medium/tr_TR-fahrettin-medium.onnx',
        'gilles-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-fr-gilles-low.tar.gz',
        'google-medium': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-ne-google-medium.tar.gz',
        'google-x-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-ne-google-x-low.tar.gz',
        'gosia-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx',
        'harri-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-fi-harri-low.tar.gz',
        'harri-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/fi/fi_FI/harri/medium/fi_FI-harri-medium.onnx',
        'huayan-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx',
        'huayan-x-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-zh-cn-huayan-x-low.tar.gz',
        'imre-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/hu/hu_HU/imre/medium/hu_HU-imre-medium.onnx',
        'irina-medium': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-ru-irinia-medium.tar.gz',
        'iseke-x-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-kk-iseke-x-low.tar.gz',
        'issai-high': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-kk-issai-high.tar.gz',
        'jirka-low': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/cs/cs_CZ/jirka/low/cs_CZ-jirka-low.onnx',
        'jirka-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/cs/cs_CZ/jirka/medium/cs_CZ-jirka-medium.onnx',
        'kareem-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/ar/ar_JO/kareem/medium/ar_JO-kareem-medium.onnx',
        'karlsson-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-de-karlsson-low.tar.gz',
        'kathleen-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-kathleen-low.tar.gz',
        'kerstin-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-de-kerstin-low.tar.gz',
        'lada-x-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-uk-lada-x-low.tar.gz',
        'lanfrica-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/sw/sw_CD/lanfrica/medium/sw_CD-lanfrica-medium.onnx',
        'lessac': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us_lessac.tar.gz',
        'lessac-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-lessac-low.tar.gz',
        'lessac-medium': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-lessac-medium.tar.gz',
        'libritts-high': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-libritts-high.tar.gz',
        'lili-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/sk/sk_SK/lili/medium/sk_SK-lili-medium.onnx',
        'marylux-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/lb/lb_LU/marylux/medium/lb_LU-marylux-medium.onnx',
        'mc_speech-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/mc_speech/medium/pl_PL-mc_speech-medium.onnx',
        'mihai-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/ro/ro_RO/mihai/medium/ro_RO-mihai-medium.onnx',
        'mls_10246-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-es-mls_10246-low.tar.gz',
        'mls_1840-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-fr-mls_1840-low.tar.gz',
        'mls_5809-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-nl-mls_5809-low.tar.gz',
        'mls_6892-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-pl-mls_6892-low.tar.gz',
        'mls_7432-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-nl-mls_7432-low.tar.gz',
        'mls_9972-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-es-mls_9972-low.tar.gz',
        'nathalie-x-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-nl-nathalie-x-low.tar.gz',
        'natia-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/ka/ka_GE/natia/medium/ka_GE-natia-medium.onnx',
        'nst-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/sv/sv_SE/nst/medium/sv_SE-nst-medium.onnx',
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
        'salka-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/is/is_IS/salka/medium/is_IS-salka-medium.onnx',
        'serbski_institut-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/sr/sr_RS/serbski_institut/medium/sr_RS-serbski_institut-medium.onnx',
        'sharvard-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx',
        'siwis-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-fr-siwis-low.tar.gz',
        'siwis-medium': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-fr-siwis-medium.tar.gz',
        'steinn-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/is/is_IS/steinn/medium/is_IS-steinn-medium.onnx',
        'southern_english_female-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-gb-southern_english_female-low.tar.gz',
        'talesyntese-medium': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-no-talesyntese-medium.tar.gz',
        'thorsten-high': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx',
        'thorsten-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-de-thorsten-low.tar.gz',
        'thorsten-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx',
        'thorsten_emotional-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten_emotional/medium/de_DE-thorsten_emotional-medium.onnx',
        'tugao-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_PT/tug%C3%A3o/medium/pt_PT-tug%C3%A3o-medium.onnx',
        'ugla-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/is/is_IS/ugla/medium/is_IS-ugla-medium.onnx',
        'ukrainian_tts-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/uk/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium.onnx',
        'upc_ona-x-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-ca-upc_ona-x-low.tar.gz',
        'upc_pau-x-low': 'https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-ca-upc_pau-x-low.tar.gz',
        'upmc-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx',
        'vais1000-medium': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx',
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

                        with open(model + ".json", "r", encoding="utf-8") as config_file:
                            config_dict = json.load(config_file)

                        engine = PiperVoice(
                            config=PiperConfig.from_dict(config_dict),
                            session=onnxruntime.InferenceSession(
                                str(model),
                                sess_options=onnxruntime.SessionOptions(),
                                providers=["CPUExecutionProvider"]
                                if not self.use_cuda
                                else ["CUDAExecutionProvider"],
                            ),
                        )

                        LOG.debug(f"loaded model: {model}")
                        PiperTTSPlugin.engines[voice] = engine
                        return engine, speaker
                else:
                    raise FileNotFoundError("onnx model not found")
        else:
            raise ValueError(f"invalid voice: {voice}")

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

        engine, speaker = self.get_model(lang, voice, speaker)
        with wave.open(wav_file, "wb") as f:
            engine.synthesize(sentence, f,
                              speaker_id=speaker,
                              length_scale=self.length_scale,
                              noise_scale=self.noise_scale,
                              noise_w=self.noise_w)

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
