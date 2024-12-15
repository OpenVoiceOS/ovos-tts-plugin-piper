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
import wave

import onnxruntime
from ovos_plugin_manager.templates.tts import TTS
from ovos_utils.lang import standardize_lang_tag
from ovos_utils.log import LOG

from ovos_tts_plugin_piper.download import LANG2VOICES, SHORTNAMES, VoiceNotFoundError, get_voice_files, get_lang_voices, get_default_voice
from ovos_tts_plugin_piper.piper import PiperVoice, PiperConfig


class PiperTTSPlugin(TTS):
    """Interface to Piper TTS."""
    engines = {}

    def __init__(self, config=None):
        super().__init__(config=config)
        if self.voice == "default":
            if self.lang.startswith("en"):
                # alan pope is the default english voice of mycroft/OVOS
                self.voice = "alan-low"
            else:
                self.voice = get_default_voice(self.lang)

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
            lang = standardize_lang_tag(lang)
            voice = LANG2VOICES.get(lang)
            if voice and isinstance(voice, list):
                voice = voice[0]
            if voice and voice not in preload_voices:
                preload_voices.append(voice)

        for voice in preload_voices:
            self.get_model(voice=voice)

    def get_model(self, lang=None, voice=None, speaker=None):

        # find default voice  (should be called model not voice....)
        if voice is None and lang is not None:
            voice = get_default_voice(lang)

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

        voice = SHORTNAMES.get(voice) or voice  # normalize aliases

        # pre-loaded models
        if voice in PiperTTSPlugin.engines:
            return PiperTTSPlugin.engines[voice], speaker

        try:
            model, model_config = get_voice_files(voice)
        except VoiceNotFoundError as e:
            LOG.error(f"Voice files for '{voice}' not found: {e}")
            raise

        with open(model_config, "r", encoding="utf-8") as config_file:
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
        # HACK: bug in some neon-core versions
        # neon_audio.tts.neon:_get_tts:198 - INFO - Legacy Neon TTS signature found
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
        return set(LANG2VOICES.keys())


PiperTTSPluginConfig = {
    lang: [{v: {"voice": v, "offline": True}}
           for v in voices]
    for lang, voices in LANG2VOICES.items()
}

if __name__ == "__main__":
    config = {}
    config["lang"] = "pt-pt"
    e = PiperTTSPlugin(config=config)
    e.get_tts("olá mundo", "hello.wav")
    print(PiperTTSPluginConfig)
