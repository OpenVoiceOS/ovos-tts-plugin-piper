"""Utility for downloading Piper voices."""
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Set, Tuple, Union, List
from urllib.parse import quote
from urllib.request import urlopen

from langcodes import closest_match, tag_distance
from ovos_utils.lang import standardize_lang_tag
from ovos_utils.log import LOG
from ovos_utils.xdg_utils import xdg_data_home

VOICES_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/{file}"
DATA_DIR = f"{xdg_data_home()}/piper_tts"
_DIR = Path(__file__).parent
_SKIP_FILES = {"MODEL_CARD"}

LANG2VOICES = defaultdict(list)
SHORTNAMES = {}
LOCALMODELS = {}


class VoiceNotFoundError(FileNotFoundError):
    pass


def get_file_hash(path: Union[str, Path], bytes_per_chunk: int = 8192) -> str:
    """Hash a file in chunks using md5."""
    path_hash = hashlib.md5()
    with open(path, "rb") as path_file:
        chunk = path_file.read(bytes_per_chunk)
        while chunk:
            path_hash.update(chunk)
            chunk = path_file.read(bytes_per_chunk)

    return path_hash.hexdigest()


def get_available_voices(update_voices: bool = False) -> Dict[str, Any]:
    """Loads available voices from downloaded or embedded JSON file."""
    download_dir = Path(DATA_DIR)
    voices_download = download_dir / "voices.json"

    if update_voices:
        # Download latest voices.json
        voices_url = VOICES_URL.format(file="voices.json")
        LOG.debug("Downloading %s to %s", voices_url, voices_download)
        try:
            with urlopen(voices_url) as response, open(
                    voices_download, "wb"
            ) as download_file:
                shutil.copyfileobj(response, download_file)
        except Exception as e:
            LOG.error(f"Failed to download {voices_url}: {e}")

    # Prefer downloaded file to embedded
    voices_embedded = _DIR / "voices.json"
    voices_path = voices_download if voices_download.exists() else voices_embedded

    LOG.debug("Loading %s", voices_path)
    with open(voices_path, "r", encoding="utf-8") as voices_file:
        return json.load(voices_file)


def get_voice_files(name: str) -> Tuple[Path, Path]:
    voices_info = get_available_voices()
    name = SHORTNAMES.get(name) or name
    if name in LOCALMODELS:
        model_path, model_cfg = LOCALMODELS[name]
        return Path(model_path), Path(model_cfg)

    if name not in voices_info:
        raise VoiceNotFoundError(name)

    voice_info = voices_info[name]
    voice_files = voice_info["files"]
    files_to_download: Set[str] = set()

    data_dir = Path(DATA_DIR)

    # Check sizes/hashes
    for file_path, file_info in voice_files.items():
        if file_path in files_to_download:
            # Already planning to download
            continue

        file_name = Path(file_path).name
        if file_name in _SKIP_FILES:
            continue

        data_file_path = data_dir / file_name
        LOG.debug("Checking %s", data_file_path)
        if not data_file_path.exists():
            LOG.debug("Missing %s", data_file_path)
            files_to_download.add(file_path)
            continue

        expected_size = file_info["size_bytes"]
        actual_size = data_file_path.stat().st_size
        if expected_size != actual_size:
            LOG.warning(
                "Wrong size (expected=%s, actual=%s) for %s",
                expected_size,
                actual_size,
                data_file_path,
            )
            files_to_download.add(file_path)
            continue

        expected_hash = file_info["md5_digest"]
        actual_hash = get_file_hash(data_file_path)
        if expected_hash != actual_hash:
            LOG.warning(
                "Wrong hash (expected=%s, actual=%s) for %s",
                expected_hash,
                actual_hash,
                data_file_path,
            )
            files_to_download.add(file_path)
            continue

    if (not voice_files) and (not files_to_download):
        raise VoiceNotFoundError(f"Unable to find or download voice: {name}")

    # Download missing files
    for file_path in files_to_download:
        file_name = Path(file_path).name
        if file_name in _SKIP_FILES:
            continue
        file_url = VOICES_URL.format(file=file_path)
        _download_file(file_url, data_dir / file_name, file_name)
    return find_voice(name)


def _download_file(file_url, download_file_path: Path, file_name=None):
    download_file_path.parent.mkdir(parents=True, exist_ok=True)
    file_name = file_name or download_file_path.name
    LOG.debug("Downloading %s to %s", file_url, download_file_path)
    try:
        with urlopen(quote(file_url, safe=":/")) as response, open(
                download_file_path, "wb"
        ) as download_file:
            shutil.copyfileobj(response, download_file)
        LOG.info("Downloaded %s (%s)", download_file_path, file_url)
    except Exception as e:
        LOG.error(f"Failed to download {file_url}: {e}")
        raise VoiceNotFoundError(f"Could not download file {file_name}") from e


def find_voice(name: str) -> Tuple[Path, Path]:
    if name in LOCALMODELS:
        model_path, model_cfg = LOCALMODELS[name]
        return Path(model_path), Path(model_cfg)
    data_dir = Path(DATA_DIR)
    name = SHORTNAMES.get(name) or name
    onnx_path = data_dir / f"{name}.onnx"
    config_path = data_dir / f"{name}.onnx.json"

    if onnx_path.exists() and config_path.exists():
        return onnx_path, config_path

    raise VoiceNotFoundError(f"Missing files for voice {name}")


for voice, data in get_available_voices().items():
    lang = standardize_lang_tag(data["language"]["code"])
    name = voice.replace(data["language"]["code"] + "-", "")
    LANG2VOICES[lang].append(name)
    SHORTNAMES[name] = voice


def get_best_lang_code(desired_lang):
    desired_lang = standardize_lang_tag(desired_lang)
    lang, dist = closest_match(desired_lang, LANG2VOICES)
    if dist < 10:
        return lang
    raise VoiceNotFoundError("Unsupported language")


def get_lang_voices(lang: str) -> List[Tuple[str, int]]:
    lang = standardize_lang_tag(lang)
    voices = [(SHORTNAMES[v2], tag_distance(lang, k))
              for k, v in LANG2VOICES.items()
              for v2 in v
              if tag_distance(lang, k) < 10]
    if not voices:
        raise VoiceNotFoundError("Unsupported language")
    return sorted(voices, key=lambda k: k[1])


def get_default_voice(lang: str) -> str:
    return get_lang_voices(lang)[0][0]


def add_local_model(voice: str, model_path: str, model_cfg: str, lang: str):
    global LOCALMODELS, LANG2VOICES
    LOG.info(f"Adding local piperTTS model: {model_path}")
    if model_path.startswith("http"):
        data_dir = Path(DATA_DIR)
        file_name = model_path.split("/")[-1]
        download_file_path = data_dir / file_name
        if not download_file_path.exists():
            _download_file(model_path, download_file_path)
        model_path = download_file_path
    if model_cfg.startswith("http"):
        data_dir = Path(DATA_DIR)
        file_name = model_cfg.split("/")[-1]
        download_file_path = data_dir / file_name
        if not download_file_path.exists():
            _download_file(model_cfg, download_file_path)
        model_cfg = download_file_path

    LOCALMODELS[voice] = [model_path, model_cfg]
    LANG2VOICES[lang].append(voice)
