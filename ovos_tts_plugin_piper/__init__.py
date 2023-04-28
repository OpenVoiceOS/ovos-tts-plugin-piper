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

from ovos_plugin_manager.templates.tts import TTS

from ovos_tts_plugin_piper.engine import Piper
from ovos_plugin_manager.templates.g2p import Grapheme2PhonemePlugin


class PiperG2P(Grapheme2PhonemePlugin):
    piper = None

    def __init__(self, piper_engine, *args, **kwargs):
        self.piper = piper_engine
        super().__init__(*args, **kwargs)

    def get_ipa(self, word, lang=None, ignore_oov=False):
        return self.piper.phonemizer.phonemize(word, phoneme_separator=" ").split(" ")


class PiperTTSPlugin(TTS):
    """Interface to Piper TTS."""

    def __init__(self, lang="en-us", config=None):
        super(PiperTTSPlugin, self).__init__(lang, config)
        self.model = self.config["model"] # TODO auto-download
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
        wav_bytes = self.engine.synthesize(sentence,
                                           speaker_id=speaker or self.speaker,
                                           length_scale=self.length_scale,
                                           noise_scale=self.noise_scale,
                                           noise_w=self.noise_w)
        with open(wav_file, "wb") as f:
            f.write(wav_bytes)

        phonemes = self.g2p.utterance2arpa(sentence, lang)
        phonemes = " ".join([f"{p}:0.4" for p in phonemes])

        return wav_file, phonemes


PiperTTSPluginConfig = {
# TODO - list of all models + url + auto download if needed to XDG path
}


if __name__ == "__main__":
    config = {}
    config["model"] = "~/ovos-tts-plugin-piper/en-gb-southern_english_female-low.onnx"
    e = PiperTTSPlugin(config=config)
    e.get_tts("hello world", "hello.wav")
    e.execute("hello world")