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
import onnxruntime
import wave
from langcodes import closest_supported_match
from ovos_plugin_manager.templates.tts import TTS
from ovos_tts_plugin_piper.piper import PiperVoice, PiperConfig
from ovos_tts_plugin_piper.voice_models import add_local_model, LOCALMODELS, LANG2VOICES, SHORTNAMES, \
    VoiceNotFoundError, get_voice_files, get_default_voice
from ovos_utils import classproperty
from ovos_utils.lang import standardize_lang_tag
from ovos_utils.log import LOG


def get_espeak_voice(lang: str) -> str:
    _ESPEAK_VOICES = {'es-419', 'ca', 'qya', 'ga', 'en-us-nyc', 'et', 'ky', 'io', 'fa-latn', 'en-gb', 'fo', 'haw', 'kl',
                      'ta', 'ml', 'gd', 'sd', 'es', 'hy', 'ur', 'ro', 'hi', 'or', 'ti', 'ca-va', 'om', 'tr', 'pa',
                      'smj', 'mk', 'bg', 'cv', "fr", 'fi', 'en-gb-x-rp', 'ru', 'mt', 'an', 'mr', 'pap', 'vi', 'id',
                      'chr-US-Qaaa-x-west', 'fr-be', 'ltg', 'my', 'nl', 'shn', 'ba', 'az', 'cmn', 'da', 'as', 'sw',
                      'piqd', 'en-us', 'hr', 'it', 'ug', 'th', 'mi', 'cy', 'ru-lv', 'ia', 'tt', 'hu', 'xex', 'te', 'ne',
                      'eu', 'ja', 'bpy', 'hak', 'cs', 'en-gb-scotland', 'hyw', 'uk', 'pt', 'bn', 'mto', 'yue',
                      'be', 'gu', 'sv', 'sl', 'cmn-latn-pinyin', 'lfn', 'lv', 'fa', 'sjn', 'nog', 'ms',
                      'vi-vn-x-central', 'lt', 'kn', 'he', 'qu', 'ca-ba', 'quc', 'nb', 'sk', 'tn', 'py', 'si', 'de',
                      'ar', 'en-gb-x-gbcwmd', 'bs', 'qdb', 'sq', 'sr', 'tk', 'en-029', 'ht', 'ru-cl', 'af', 'pt-br',
                      'fr-ch', 'ka', 'en-gb-x-gbclan', 'ko', 'is', 'ca-nw', 'gn', 'kok', 'la', 'lb', 'am', 'kk', 'ku',
                      'kaa', 'jbo', 'eo', 'uz', 'nci', 'vi-vn-x-south', 'el', 'pl', 'grc'}
    _INVALID = ['chr-US-Qaaa-x-west', 'en-us-nyc', 'fr-fr']  # fails to normalize

    if lang.lower() == "en-gb":
        return "en-gb-x-rp"
    if lang in _ESPEAK_VOICES or lang in _INVALID:
        return lang
    if lang.lower().split("-")[0] in _ESPEAK_VOICES:
        return lang.lower().split("-")[0]
    voices = [v for v in _ESPEAK_VOICES if v not in _INVALID]
    return closest_supported_match(lang, voices, 10)


class PiperTTSPlugin(TTS):
    """Interface to Piper TTS."""
    engines = {}

    def __init__(self, config=None):
        super().__init__(config=config)
        if self.config.get("model"):
            model = self.config["model"]
            model_config = self.config.get("model_config") or model + ".json"
            add_local_model(voice=self.voice, model_path=model,
                            model_cfg=model_config, lang=self.lang)

        elif self.voice == "default":
            if self.lang.startswith("en"):
                # alan pope is the default english voice of mycroft/OVOS
                self.voice = "alan-low"
            else:
                self.voice = get_default_voice(self.lang)

        if isinstance(self.voice, list):
            self.voice = self.voice[0]

        self.accent = self.config.get("accent", None)
        self.use_cuda = self.config.get("use_cuda", False)
        self.noise_scale = self.config.get("noise-scale")  # generator noise
        self.length_scale = self.config.get("length-scale")  # Phoneme length
        self.noise_w = self.config.get("noise-w")  # Phoneme width noise

        # pre-load models
        preload_voices = self.config.get("preload_voices") or [self.voice]
        preload_langs = self.config.get("preload_langs") or []

        for lang in preload_langs:
            lang = standardize_lang_tag(lang)
            voice = LANG2VOICES.get(lang)
            if voice and isinstance(voice, list):
                voice = voice[0]
            if voice and voice not in preload_voices:
                preload_voices.append(voice)

        for voice in preload_voices:
            self.lang2model(voice=voice)

    def lang2model(self, lang=None, voice=None, speaker=None):
        # find default voice  (should be called model not voice....)
        if voice is None and lang is not None:
            lang = standardize_lang_tag(lang)
            if lang == self.lang and LOCALMODELS:
                voice = self.voice
            else:
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
            return PiperTTSPlugin.engines[voice], speaker, voice

        try:
            model, model_config = get_voice_files(voice)
        except VoiceNotFoundError as e:
            LOG.error(f"Voice files for '{voice}' not found: {e}")
            raise

        return self.get_model(str(model), str(model_config), voice, speaker)

    def get_model(self, model: str, model_config: str,
                  voice: str = None, speaker=0):
        voice = voice or self.voice
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
        return engine, speaker, voice

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
        # HACK: bug in some neon-core versions
        # neon_audio.tts.neon:_get_tts:198 - INFO - Legacy Neon TTS signature found
        if isinstance(speaker, dict):
            LOG.warning("Legacy Neon TTS signature found, pass speaker as a str")
            speaker = None

        phonemizer_lang = None  # if None use model's default accent
        if self.accent:
            phonemizer_lang = get_espeak_voice(self.accent)

        if voice:
            # user requested a specific voice model to be used
            lang = lang or self.lang
            engine, speaker, voice = self.lang2model(lang, voice, speaker)
        elif lang:
            # requested a language but not a voice
            # - try to use default voice, but force language via phonemizer
            # - change to a different TTS model completely
            try:
                # force a specific espeak phonemizer to match lang pronunciation
                # this allows a voice to speak a different language a bit better
                phonemizer_lang = get_espeak_voice(lang)
                engine, speaker, voice = self.lang2model(self.lang, self.voice)
            except:  # change to a voice that supports the lang
                LOG.debug("Switching TTS model for one that supports target language")
                engine, speaker, voice = self.lang2model(lang, voice, speaker)
        else:
            # default case, no specific voice or lang requested
            engine, speaker, voice = self.lang2model(self.lang, self.voice)

        if phonemizer_lang:
            LOG.debug(f"Forcing Piper accent: {phonemizer_lang}")

        with wave.open(wav_file, "wb") as f:
            engine.synthesize(sentence, f,
                              speaker_id=speaker,
                              length_scale=self.length_scale,
                              noise_scale=self.noise_scale,
                              noise_w=self.noise_w,
                              phonemizer_lang=phonemizer_lang)

        return wav_file, None

    @classproperty
    def available_languages(cls) -> set:
        return set(LANG2VOICES.keys())


PiperTTSPluginConfig = {
    lang: [{v: {"voice": v, "offline": True}}
           for v in voices]
    for lang, voices in LANG2VOICES.items()
}

if __name__ == "__main__":
    for lang, sentence in [
        ("pt-PT", "olá mundo"),
        ("en-GB", "hello world"),
        ("en-US", "hello world"),
        ("en-AU", "hello world"),
        ("es-ES", "hola mundo"),
        ("ca-ES", "hola món"),
        ("fr-FR", "bonjour le monde"),
        ("it-IT", "ciao mondo"),
        ("nl-NL", "hallo wereld"),
        ("de-DE", "hallo welt")
    ]:
        e = PiperTTSPlugin()
        e.get_tts(sentence, f"ap_{lang}.wav", lang=lang)
