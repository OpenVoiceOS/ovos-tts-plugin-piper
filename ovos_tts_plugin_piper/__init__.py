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

from .lang2voices import LANG2VOICES
from .voice2url import VOICE2URL

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
    
    lang2voices = {}
    voice2url = {}
    engines = {}

    def __init__(self, lang="en-us", config=None):
        super(PiperTTSPlugin, self).__init__(lang, config)

        self.lang2voices = LANG2VOICES
        self.voice2url = VOICE2URL

        self.voice = self.config.get("model", self.voice)

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
        self.model_path = self.config.get("model-path", f"{xdg_data_home()}/piper_tts")

        self.get_model(voice=self.voice)

    def get_model_name(self, src: str) -> str:
        if src in self.voice2url:
            return self.voice2url[src].rsplit('/', 1)[-1].split('.')[0]
        if src.startswith("http"):
            return src.rsplit('/', 1)[-1].split('.')[0]
        raise ValueError("Must be predefined voice or url to onnx / gz model")
    
    def get_speaker_from_model_name(self, name: str) -> int:
        speaker = 0 # default for this model
        if "#" in name:
            speaker2 = name.split("#")[1]
            try:
                speaker2 = int(speaker2)
                if speaker is not None:
                    LOG.warning("requested voice and speaker args conflict, "
                                "ignoring requested speaker in favor of speaker defined in voice string")
                    speaker = speaker2
            except:
                LOG.warning("invalid speaker requested in voice string, ignoring it")

        return speaker

    def get_model(self, voice=None, speaker=None):
        voice = voice or self.voice

        if isinstance(voice, list):
            voice = voice[0]

        # find speaker  (should be called voice not speaker...)
        speaker = self.get_speaker_from_model_name(voice)

        # pre-loaded models
        if voice in PiperTTSPlugin.engines:
            return PiperTTSPlugin.engines[voice], speaker

        xdg_p = f"{self.model_path}/{self.get_model_name(voice)}"
        if not os.path.isdir(xdg_p):
            url = voice
            if voice in self.voice2url:
                url = self.voice2url[voice]
            self.download_model(xdg_p, url)
        
        engine = self.load_model_directory(xdg_p)
        LOG.debug(f"loaded model: {xdg_p}")
        PiperTTSPlugin.engines[voice] = engine
        return engine, speaker
        
    def download_model(self, model_directory: str, url: str):
        """Download the file for a model into a model directory
            Model name will be determined from the file name
        
        Arguments:
            model_directory (str): the directory to store models in
            url (str): the url to the .onnx or .tar.gz
        """
        if not url.startswith("http"):
            raise ValueError("model url must start with http")
        
        file_name = url.rsplit("/", -1)[-1]
        os.makedirs(model_directory, exist_ok=True)
        LOG.info(f"downloading piper model: {url}")
        data = requests.get(url, timeout=120)
        with open(f"{model_directory}/{file_name}", "wb") as f:
            f.write(data.content)

        if url.endswith(".onnx"):
            json_data = requests.get(url + '.json', timeout=120)
            with open(f"{model_directory}/{file_name}.json", "wb") as f:
                f.write(json_data.content)
        else:
            with tarfile.open(f"{model_directory}/{file_name}") as file:
                file.extractall(model_directory)

    def load_model_directory(self, model_dir: str) -> PiperVoice:
        """Create an instance of a PiperVoice from a directory containing an .onnx file and its .json definition"""
        for f in os.listdir(model_dir):
            if f.endswith("onnx"):
                model = f"{model_dir}/{f}"

                with open(model + ".json", "r", encoding="utf-8") as config_file:
                    config_dict = json.load(config_file)

                return PiperVoice(
                    config = PiperConfig.from_dict(config_dict),
                    session=onnxruntime.InferenceSession(
                        str(model),
                        sess_options=onnxruntime.SessionOptions(),
                        providers=["CPUExecutionProvider"]
                        if not self.use_cuda
                        else ["CUDAExecutionProvider"],
                    ),
                )
        raise FileNotFoundError("onnx model not found")


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
        voice = voice or self.voice
        # HACK: bug in some neon-core versions - neon_audio.tts.neon:_get_tts:198 - INFO - Legacy Neon TTS signature found
        if isinstance(speaker, dict):
            LOG.warning("Legacy Neon TTS signature found, pass speaker as a str")
            speaker = None

        engine, speaker = self.get_model(voice, speaker)
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
