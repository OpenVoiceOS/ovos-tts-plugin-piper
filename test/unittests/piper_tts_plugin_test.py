import unittest
import os
import shutil
from piper.voice import PiperVoice

from ovos_tts_plugin_piper import PiperTTSPlugin

class TestPiperTTSPluginMethods(unittest.TestCase):
    def tearDown(self) -> None:
        # try:
        #     shutil.rmtree("./test/models")
        #     os.remove("./test/hello.wav")
        # except:
        #     pass

        return super().tearDown()

    def test_class_setup(self):
        plugin = PiperTTSPlugin()
        self.assertEqual(plugin.voice, "alan-low")
        self.assertEqual(plugin.use_cuda, False)

    def test_class_setup_with_config(self):
        plugin = PiperTTSPlugin(config={
            "model": "alba-medium",
            "use_cuda": True,
            "noise-scale": 1,
            "length-scale": 1,
            "noise-w": 1,
            "model-path": "./test/models"
        })
        self.assertEqual(plugin.voice, "alba-medium")
        self.assertEqual(plugin.use_cuda, True)
        self.assertEqual(plugin.noise_scale, 1)
        self.assertEqual(plugin.length_scale, 1)
        self.assertEqual(plugin.noise_w, 1)
        self.assertEqual(plugin.model_path, "./test/models")

    def test_get_model_name(self):
        plugin = PiperTTSPlugin(config={
            "model-path": "./test/models"
        })
        self.assertEqual(plugin.get_model_name("alba-medium"), "en_GB-alba-medium")
        self.assertEqual(plugin.get_model_name("https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx"), "pt_BR-faber-medium")
        with self.assertRaises(ValueError):
            plugin.get_model_name("something-invalid")

    def test_download_model(self):
        plugin = PiperTTSPlugin(config={
            "model-path": "./test/models"
        })
        plugin.download_model("./test/models/pt_BR-faber-medium", "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx")
        self.assertTrue(os.path.isfile("./test/models/pt_BR-faber-medium/pt_BR-faber-medium.onnx"))
        self.assertTrue(os.path.isfile("./test/models/pt_BR-faber-medium/pt_BR-faber-medium.onnx.json"))

        plugin.download_model("./test/models/voice-ru-irinia-medium", "https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-ru-irinia-medium.tar.gz")
        self.assertTrue(os.path.isfile("./test/models/voice-ru-irinia-medium/ru-irinia-medium.onnx"))
        self.assertTrue(os.path.isfile("./test/models/voice-ru-irinia-medium/ru-irinia-medium.onnx.json"))

    def test_load_model_directory(self):
        plugin = PiperTTSPlugin(config={
            "model-path": "./test/models",
            "model": "alba-medium"
        })
        self.assertIsInstance(plugin.load_model_directory("./test/models/en_GB-alba-medium"), PiperVoice)

    def test_get_voice_subtype_from_model_name(self):
        plugin = PiperTTSPlugin()
        self.assertEqual(plugin.get_speaker_from_model_name("alan-low"), 0)
        self.assertEqual(plugin.get_speaker_from_model_name("alan-low#"), 0)
        self.assertEqual(plugin.get_speaker_from_model_name("alan-low#1"), 1)

    def test_get_tts(self):
        plugin = PiperTTSPlugin(config={
            "model-path": "./test/models"
        })
        plugin.get_tts("one oh clock? yes! it is one on the clock", "./test/hello.wav")
        self.assertTrue(os.path.isfile("./test/hello.wav"))
        file_stats = os.stat("./test/hello.wav")
        self.assertTrue(file_stats.st_size > 0)


if __name__ == '__main__':
    unittest.main()

