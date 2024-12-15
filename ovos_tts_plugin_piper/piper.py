import json
import wave
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Iterable, List, Optional, Tuple, Union

import numpy as np
import onnxruntime
from ovos_utils.log import LOG
from piper_phonemize import phonemize_codepoints, phonemize_espeak, tashkeel_run

PAD = "_"  # padding (0)
BOS = "^"  # beginning of sentence
EOS = "$"  # end of sentence


class PhonemeType(str, Enum):
    ESPEAK = "espeak"
    TEXT = "text"


@dataclass
class PiperConfig:
    """Piper configuration"""

    num_symbols: int
    """Number of phonemes"""

    num_speakers: int
    """Number of speakers"""

    sample_rate: int
    """Sample rate of output audio"""

    espeak_voice: str
    """Name of espeak-ng voice or alphabet"""

    length_scale: float
    noise_scale: float
    noise_w: float

    phoneme_id_map: Mapping[str, Sequence[int]]
    """Phoneme -> [id,]"""

    phoneme_type: PhonemeType
    """espeak or text"""

    @staticmethod
    def from_dict(config: Dict[str, Any]) -> "PiperConfig":
        inference = config.get("inference", {})

        return PiperConfig(
            num_symbols=config["num_symbols"],
            num_speakers=config["num_speakers"],
            sample_rate=config["audio"]["sample_rate"],
            noise_scale=inference.get("noise_scale", 0.667),
            length_scale=inference.get("length_scale", 1.0),
            noise_w=inference.get("noise_w", 0.8),
            #
            espeak_voice=config["espeak"]["voice"],
            phoneme_id_map=config["phoneme_id_map"],
            phoneme_type=PhonemeType(config.get("phoneme_type", PhonemeType.ESPEAK)),
        )


def audio_float_to_int16(
        audio: np.ndarray, max_wav_value: float = 32767.0
) -> np.ndarray:
    """Normalize audio and convert to int16 range"""
    audio_norm = audio * (max_wav_value / max(0.01, np.max(np.abs(audio))))
    audio_norm = np.clip(audio_norm, -max_wav_value, max_wav_value)
    audio_norm = audio_norm.astype("int16")
    return audio_norm


@dataclass
class PiperVoice:
    session: onnxruntime.InferenceSession
    config: PiperConfig

    @staticmethod
    def load(
            model_path: Union[str, Path],
            config_path: Optional[Union[str, Path]] = None,
            use_cuda: bool = False,
    ) -> "PiperVoice":
        """Load an ONNX model and config."""
        if config_path is None:
            config_path = f"{model_path}.json"

        with open(config_path, "r", encoding="utf-8") as config_file:
            config_dict = json.load(config_file)

        providers: List[Union[str, Tuple[str, Dict[str, Any]]]]
        if use_cuda:
            providers = [
                (
                    "CUDAExecutionProvider",
                    {"cudnn_conv_algo_search": "HEURISTIC"},
                )
            ]
        else:
            providers = ["CPUExecutionProvider"]

        return PiperVoice(
            config=PiperConfig.from_dict(config_dict),
            session=onnxruntime.InferenceSession(
                str(model_path),
                sess_options=onnxruntime.SessionOptions(),
                providers=providers,
            ),
        )

    def phonemize(self, text: str) -> List[List[str]]:
        """Text to phonemes grouped by sentence."""
        if self.config.phoneme_type == PhonemeType.ESPEAK:
            if self.config.espeak_voice == "ar":
                # Arabic diacritization
                # https://github.com/mush42/libtashkeel/
                text = tashkeel_run(text)

            return phonemize_espeak(text, self.config.espeak_voice)

        if self.config.phoneme_type == PhonemeType.TEXT:
            return phonemize_codepoints(text)

        raise ValueError(f"Unexpected phoneme type: {self.config.phoneme_type}")

    def phonemes_to_ids(self, phonemes: List[str]) -> List[int]:
        """Phonemes to ids."""
        id_map = self.config.phoneme_id_map
        ids: List[int] = list(id_map[BOS])

        for phoneme in phonemes:
            if phoneme not in id_map:
                LOG.warning("Missing phoneme from id map: %s", phoneme)
                continue

            ids.extend(id_map[phoneme])
            ids.extend(id_map[PAD])

        ids.extend(id_map[EOS])

        return ids

    def synthesize(
            self,
            text: str,
            wav_file: wave.Wave_write,
            speaker_id: Optional[int] = None,
            length_scale: Optional[float] = None,
            noise_scale: Optional[float] = None,
            noise_w: Optional[float] = None,
            sentence_silence: float = 0.0,
    ):
        """Synthesize WAV audio from text."""
        wav_file.setframerate(self.config.sample_rate)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setnchannels(1)  # mono

        for audio_bytes in self.synthesize_stream_raw(
                text,
                speaker_id=speaker_id,
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w=noise_w,
                sentence_silence=sentence_silence,
        ):
            wav_file.writeframes(audio_bytes)

    def synthesize_stream_raw(
            self,
            text: str,
            speaker_id: Optional[int] = None,
            length_scale: Optional[float] = None,
            noise_scale: Optional[float] = None,
            noise_w: Optional[float] = None,
            sentence_silence: float = 0.0,
    ) -> Iterable[bytes]:
        """Synthesize raw audio per sentence from text."""
        sentence_phonemes = self.phonemize(text)

        # 16-bit mono
        num_silence_samples = int(sentence_silence * self.config.sample_rate)
        silence_bytes = bytes(num_silence_samples * 2)

        for phonemes in sentence_phonemes:
            phoneme_ids = self.phonemes_to_ids(phonemes)
            yield self.synthesize_ids_to_raw(
                phoneme_ids,
                speaker_id=speaker_id,
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w=noise_w,
            ) + silence_bytes

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
