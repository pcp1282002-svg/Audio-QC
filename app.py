"""
Interactive Streamlit app for conversation audio and transcript QC.

Input:
- Google Drive folder or file link
- One mixed WAV plus separate speaker-channel WAV files
- AssemblyAI JSON transcript with word-level timestamps
- Riverside TXT transcript for speaker-name mapping only

Output:
- SNR and silence floor for every WAV, using Silero VAD speech/non-speech masks
- Speaker density from AssemblyAI speaker segments, matching the reference analyzer
- Overlap ratio from AssemblyAI word-level timestamps
- Count and details of monologues over 60 seconds from word-level turns

Install:
  pip install -r requirements.txt

Run:
  streamlit run app.py
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import wave
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import gdown
except ImportError:
    gdown = None


SUPPORTED_EXTENSIONS = {".wav", ".json", ".txt"}
MONOLOGUE_MIN_SECONDS = 60.0
DEFAULT_TURN_GAP_SECONDS = 0.2
CLIENT_SNR_MIN_DB = 20.0
CLIENT_SILENCE_MAX_DBFS = -40.0
SAFE_PASS_SNR_MIN_DB = 23.0
SAFE_PASS_SILENCE_MAX_DBFS = -43.0
DRIVE_REQUEST_TIMEOUT_SECONDS = 60
RIVERSIDE_WORDS_PER_SECOND = 2.7
RIVERSIDE_MAX_INFERRED_OVERLAP_TAIL_SECONDS = 0.75
NATURALNESS_PASS_SCORE = 75
FRONTEND_HIDDEN_COLUMNS = {
    "conversation_key",
    "_speech_intervals_vad",
    "_audio_path",
    "timing_source",
    "raw_speaker_label",
    "speaking_time_sec",
    "primary_turn_source",
    "riverside_timed_turn_count",
    "riverside_names_found",
    "speaker_name_mapping_fallback_used",
}
OUTSIZED_SILENCE_SECONDS = 5.0

BACKCHANNEL_WORDS = {
    "uh",
    "um",
    "hm",
    "hmm",
    "mhm",
    "mm",
    "mmm",
    "yeah",
    "yep",
    "yes",
    "right",
    "okay",
    "ok",
    "sure",
    "gotcha",
    "exactly",
    "true",
    "alright",
    "correct",
    "yup",
    "aha",
    "ah",
    "oh",
}

PARALINGUISTIC_PATTERNS = [
    r"\b(laughs?|laughter|laughing|haha|hehe)\b",
    r"\b(sighs?|gasps?|coughs?|breathes?|breathing|throat clearing|clears throat)\b",
    r"\[(laugh|laughter|sigh|gasp|cough|breath|noise)\]",
    r"\((laugh|laughter|sigh|gasp|cough|breath|noise)\)",
]

DISFLUENCY_PATTERNS = [
    r"\b(um+|uh+|hmm+|erm|er)\b",
    r"\b(i mean|you know|sort of|kind of)\b",
    r"\b(\w+)\s+\1\b",
    r"[-/]\s*$",
]


# -----------------------------
# General helpers
# -----------------------------


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def clean_word(text: str) -> str:
    text = str(text or "").strip().lower()
    return re.sub(r"^[^\w]+|[^\w]+$", "", text)


def normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def merge_intervals(intervals: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    cleaned = []

    for start, end in intervals:
        start = safe_float(start)
        end = safe_float(end)

        if end > start:
            cleaned.append((start, end))

    if not cleaned:
        return []

    cleaned.sort(key=lambda item: (item[0], item[1]))
    merged = [cleaned[0]]

    for start, end in cleaned[1:]:
        prev_start, prev_end = merged[-1]

        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def interval_duration(intervals: Iterable[Tuple[float, float]]) -> float:
    return sum(max(0.0, end - start) for start, end in intervals)


def speaker_intervals_union_duration(
    speaker_intervals: Dict[str, List[Tuple[float, float]]]
) -> float:
    all_intervals = []

    for intervals in speaker_intervals.values():
        all_intervals.extend(intervals)

    return interval_duration(merge_intervals(all_intervals))


def seconds_to_minutes(seconds: float) -> float:
    return round(safe_float(seconds) / 60.0, 3)


def format_mmss(seconds: float) -> str:
    total_seconds = int(math.floor(max(0.0, safe_float(seconds)) + 0.5))
    minutes, sec = divmod(total_seconds, 60)

    return f"{minutes:02d}:{sec:02d}"


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, safe_float(seconds))
    minutes, sec = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)

    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:05.2f}"

    return f"{minutes:02d}:{sec:05.2f}"


def frontend_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=list(FRONTEND_HIDDEN_COLUMNS), errors="ignore")


def apply_app_styles() -> None:
    st.markdown(
        """
<style>
div[data-testid="stSidebar"] {display: none;}
div[data-testid="stAppViewContainer"] {
  background: #0b1120;
  color: #e5edf7;
}
div[data-testid="stHeader"] {background: rgba(11, 17, 32, 0.86);}
.block-container {
  max-width: 1240px;
  padding-top: 2.2rem;
  padding-bottom: 3rem;
}
h1, h2, h3, h4, h5, h6, p, label, span {color: #e5edf7;}
div[data-testid="stMetric"] {
  background: #111827;
  border: 1px solid #263244;
  border-radius: 8px;
  padding: 14px 16px;
}
div[data-testid="stMetric"] label, div[data-testid="stMetricValue"] {
  color: #f8fafc;
}
div[data-testid="stDataFrame"] {
  border: 1px solid #263244;
  border-radius: 8px;
}
div.stButton > button, div.stDownloadButton > button {
  border-radius: 8px;
  border: 1px solid #60a5fa;
  background: #2563eb;
  color: white;
  font-weight: 700;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
  border-color: #93c5fd;
  background: #1d4ed8;
  color: white;
}
div[data-testid="stTabs"] button {
  color: #cbd5e1;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
  color: #ffffff;
}
div[data-testid="stAlert"] {
  border-radius: 8px;
}
</style>
""",
        unsafe_allow_html=True,
    )


def parse_timestamp_to_seconds(value: str) -> Optional[float]:
    value = str(value or "").strip().strip("[]()")

    if not value:
        return None

    parts = value.split(":")

    try:
        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60.0 + seconds

        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600.0 + minutes * 60.0 + seconds
    except ValueError:
        return None

    return None


def count_text_words(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", str(text or "")))


def text_tokens(text: str) -> set:
    tokens = set()

    for token in re.findall(r"\b[\w']+\b", str(text or "").lower()):
        cleaned = clean_word(token)

        if len(cleaned) >= 2:
            tokens.add(cleaned)

    return tokens


def token_bag(text: str) -> Counter:
    bag = Counter()

    for token in re.findall(r"\b[\w']+\b", str(text or "").lower()):
        cleaned = clean_word(token)

        if len(cleaned) > 2:
            bag[cleaned] += 1

    return bag


def cosine_similarity(bag_a: Counter, bag_b: Counter) -> float:
    if not bag_a or not bag_b:
        return 0.0

    dot = sum(count * bag_b.get(token, 0) for token, count in bag_a.items())
    norm_a = math.sqrt(sum(count * count for count in bag_a.values()))
    norm_b = math.sqrt(sum(count * count for count in bag_b.values()))

    if not norm_a or not norm_b:
        return 0.0

    return dot / (norm_a * norm_b)


def is_generic_speaker_label(value: str) -> bool:
    value = str(value or "").strip()
    normalized = normalize_key(value)

    return bool(
        re.fullmatch(r"[a-z]", value, re.I)
        or re.fullmatch(r"\d+", value)
        or re.fullmatch(r"(speaker|spk|participant|channel|ch|track)\d*[a-z]?", normalized)
    )


def title_case_speaker_name(value: str) -> str:
    parts = []

    for part in re.split(r"\s+", str(value or "").strip()):
        if not part:
            continue

        if part.isupper() and len(part) <= 3:
            parts.append(part)
        else:
            parts.append(part[:1].upper() + part[1:].lower())

    return " ".join(parts)


def name_from_email_or_raw(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value or "")).strip(" -:")

    if not value:
        return ""

    if "@" not in value:
        return value

    local_part = value.split("@", 1)[0]
    raw_parts = re.split(r"[._\-\s]+", local_part)
    name_parts = []

    for part in raw_parts:
        part = re.sub(r"\d+$", "", part).strip()

        if part:
            name_parts.append(part)

    return title_case_speaker_name(" ".join(name_parts)) or value


def clean_speaker_name(value: str) -> str:
    value = name_from_email_or_raw(value)
    value = re.sub(r"^\s*(?:speaker|spk|participant)\s*[_\-\s]*[A-Z0-9]+\s*[-:]\s*", "", value, flags=re.I)
    speaker_paren = re.match(r"^(?:speaker|spk|participant)\s*[_\-\s]*[A-Z0-9]+\s*\(([^)]+)\)$", value, flags=re.I)
    if speaker_paren:
        value = name_from_email_or_raw(speaker_paren.group(1))
    trailing_generic = re.match(r"^(.+?)\s*\((?:speaker|spk|participant)\s*[_\-\s]*[A-Z0-9]+\)$", value, flags=re.I)
    if trailing_generic:
        value = name_from_email_or_raw(trailing_generic.group(1))
    return value.strip()


def ordered_unique(values: Iterable[str]) -> List[str]:
    seen = []

    for value in values:
        value = str(value or "").strip()

        if value and value not in seen:
            seen.append(value)

    return seen


# -----------------------------
# Audio helpers
# -----------------------------


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)

    return np.mean(audio, axis=0 if audio.shape[0] < audio.shape[-1] else 1).astype(
        np.float32
    )


def rms_dbfs(samples: np.ndarray, eps: float = 1e-12) -> float:
    if samples.size == 0:
        return float("nan")

    rms = np.sqrt(np.mean(np.square(samples.astype(np.float64))))
    return 20.0 * math.log10(max(rms, eps))


def read_wav_builtin(file_path: str) -> Tuple[np.ndarray, int]:
    with wave.open(file_path, "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 1:
        audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        audio = audio / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        signed = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed | ~0xFFFFFF, signed)
        audio = signed.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32)
        audio = audio / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if channels > 1:
        audio = audio.reshape(-1, channels)
        audio = np.mean(audio, axis=1)

    return np.nan_to_num(audio.astype(np.float32)), sample_rate


def read_wav(file_path: str) -> Tuple[np.ndarray, int]:
    return read_wav_builtin(file_path)


def resample_linear(audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    if original_sr == target_sr:
        return audio.astype(np.float32)

    if len(audio) == 0:
        return audio.astype(np.float32)

    duration = len(audio) / float(original_sr)
    target_len = int(duration * target_sr)

    if target_len <= 0:
        return np.array([], dtype=np.float32)

    old_x = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    new_x = np.linspace(0.0, duration, num=target_len, endpoint=False)

    return np.interp(new_x, old_x, audio).astype(np.float32)


@st.cache_resource(show_spinner=False)
def load_silero_vad():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Torch is required for Silero VAD. Install torch.") from exc

    torch.set_num_threads(1)

    try:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
    except TypeError:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )

    get_speech_timestamps = utils[0]
    return model, get_speech_timestamps


def build_speech_mask_from_seconds(
    length_samples: int,
    speech_timestamps: Sequence[Dict[str, float]],
    sample_rate: int,
) -> np.ndarray:
    mask = np.zeros(length_samples, dtype=bool)

    for segment in speech_timestamps:
        start = max(0, int(safe_float(segment.get("start")) * sample_rate))
        end = min(length_samples, int(safe_float(segment.get("end")) * sample_rate))

        if end > start:
            mask[start:end] = True

    return mask


def is_finite_number(value) -> bool:
    return isinstance(value, (int, float, np.floating)) and math.isfinite(float(value))


def confidence_from_seconds(primary_seconds: float, secondary_seconds: float) -> str:
    if primary_seconds >= 10.0 and secondary_seconds >= 5.0:
        return "HIGH"

    if primary_seconds >= 3.0 and secondary_seconds >= 1.0:
        return "MEDIUM"

    return "LOW"


def lower_confidence(*labels: str) -> str:
    rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    reverse_rank = {value: key for key, value in rank.items()}
    lowest = min(rank.get(label, 0) for label in labels if label)
    return reverse_rank[lowest]


def apply_audio_thresholds_and_confidence(result: Dict) -> Dict:
    snr = result.get("snr_vad_db")
    silence_floor = result.get("silence_floor_dbfs")
    speech_seconds = safe_float(result.get("speech_duration_sec_vad"))
    silence_seconds = safe_float(result.get("silence_duration_sec_vad"))

    result["client_snr_threshold_db"] = CLIENT_SNR_MIN_DB
    result["client_silence_threshold_dbfs"] = CLIENT_SILENCE_MAX_DBFS
    result["safe_pass_snr_threshold_db"] = SAFE_PASS_SNR_MIN_DB
    result["safe_pass_silence_threshold_dbfs"] = SAFE_PASS_SILENCE_MAX_DBFS
    result["snr_margin_db"] = np.nan
    result["silence_floor_margin_db"] = np.nan
    result["client_threshold_status"] = "MANUAL REVIEW"
    result["safe_pass_status"] = "MANUAL REVIEW"
    result["snr_confidence"] = "LOW"
    result["silence_floor_confidence"] = "LOW"
    result["metric_confidence"] = "LOW"
    result["confidence_notes"] = result.get("audio_qc_error", "")

    if is_finite_number(snr):
        result["snr_margin_db"] = round(float(snr) - CLIENT_SNR_MIN_DB, 2)

    if is_finite_number(silence_floor):
        result["silence_floor_margin_db"] = round(
            CLIENT_SILENCE_MAX_DBFS - float(silence_floor),
            2,
        )

    if result.get("audio_qc_status") != "OK":
        return result

    result["snr_confidence"] = confidence_from_seconds(speech_seconds, silence_seconds)
    result["silence_floor_confidence"] = confidence_from_seconds(
        silence_seconds,
        speech_seconds,
    )

    metric_confidence = lower_confidence(
        result["snr_confidence"],
        result["silence_floor_confidence"],
    )

    snr_margin = safe_float(result["snr_margin_db"])
    silence_margin = safe_float(result["silence_floor_margin_db"])

    if min(abs(snr_margin), abs(silence_margin)) < 1.0:
        metric_confidence = lower_confidence(metric_confidence, "MEDIUM")

    result["metric_confidence"] = metric_confidence

    client_pass = snr_margin >= 0.0 and silence_margin >= 0.0
    safe_pass = (
        is_finite_number(snr)
        and is_finite_number(silence_floor)
        and float(snr) >= SAFE_PASS_SNR_MIN_DB
        and float(silence_floor) <= SAFE_PASS_SILENCE_MAX_DBFS
        and metric_confidence != "LOW"
    )

    result["client_threshold_status"] = "PASS" if client_pass else "MANUAL REVIEW"
    result["safe_pass_status"] = "SAFE PASS" if safe_pass else "MANUAL REVIEW"
    result["confidence_notes"] = (
        f"{speech_seconds:.2f}s speech and {silence_seconds:.2f}s silence used by Silero VAD"
    )

    return result


def classify_wav(path: str) -> str:
    name = Path(path).stem.lower()

    speaker_patterns = [
        r"(^|[^a-z0-9])spk[_\-\s]*[a-z0-9]*([^a-z0-9]|$)",
        r"(^|[^a-z0-9])speaker[_\-\s]*[a-z0-9]*([^a-z0-9]|$)",
        r"(^|[^a-z0-9])participant[_\-\s]*[a-z0-9]*([^a-z0-9]|$)",
        r"(^|[^a-z0-9])track[_\-\s]*[a-z0-9]*([^a-z0-9]|$)",
        r"(^|[^a-z0-9])ch(?:annel)?[_\-\s]*[a-z0-9]+([^a-z0-9]|$)",
        r"(^|[^a-z0-9])mic[_\-\s]*[a-z0-9]*([^a-z0-9]|$)",
        r"(^|[^a-z0-9])isolated([^a-z0-9]|$)",
        r"(^|[^a-z0-9])separate([^a-z0-9]|$)",
    ]

    mixed_patterns = [
        r"(^|[^a-z0-9])mix(?:ed)?([^a-z0-9]|$)",
        r"(^|[^a-z0-9])whole([^a-z0-9]|$)",
        r"(^|[^a-z0-9])full([^a-z0-9]|$)",
        r"(^|[^a-z0-9])conversation([^a-z0-9]|$)",
        r"(^|[^a-z0-9])combined([^a-z0-9]|$)",
        r"(^|[^a-z0-9])master([^a-z0-9]|$)",
        r"(^|[^a-z0-9])main([^a-z0-9]|$)",
    ]

    if any(re.search(pattern, name) for pattern in mixed_patterns):
        return "mixed"

    if any(re.search(pattern, name) for pattern in speaker_patterns):
        return "speaker_channel"

    return "wav"


def calculate_audio_qc(file_path: str) -> Dict:
    audio, original_sr = read_wav(file_path)

    result = {
        "audio_file": Path(file_path).name,
        "_audio_path": file_path,
        "audio_role": classify_wav(file_path),
        "audio_duration_sec": round(len(audio) / original_sr, 2) if original_sr else np.nan,
        "sample_rate": original_sr,
        "vad_sample_rate": "",
        "vad_speech_segments": 0,
        "speech_duration_sec_vad": np.nan,
        "silence_duration_sec_vad": np.nan,
        "speech_level_dbfs": np.nan,
        "silence_floor_dbfs": np.nan,
        "snr_vad_db": np.nan,
        "audio_qc_status": "ERROR",
        "audio_qc_error": "",
        "_speech_intervals_vad": [],
    }

    if audio.size == 0:
        result["audio_qc_error"] = "Audio file is empty"
        return apply_audio_thresholds_and_confidence(result)

    if not original_sr:
        result["audio_qc_error"] = "Missing sample rate"
        return apply_audio_thresholds_and_confidence(result)

    silero_sr = 8000 if original_sr == 8000 else 16000
    vad_audio = resample_linear(audio, original_sr, silero_sr)
    result["vad_sample_rate"] = silero_sr

    if vad_audio.size == 0:
        result["audio_qc_error"] = "Audio is too short"
        return apply_audio_thresholds_and_confidence(result)

    try:
        import torch

        model, get_speech_timestamps = load_silero_vad()
        wav_tensor = torch.from_numpy(vad_audio).float()

        speech_timestamps = get_speech_timestamps(
            wav_tensor,
            model,
            sampling_rate=silero_sr,
            return_seconds=True,
        )
    except Exception as exc:
        result["audio_qc_error"] = f"Silero VAD failed: {exc}"
        return apply_audio_thresholds_and_confidence(result)

    result["vad_speech_segments"] = len(speech_timestamps)
    result["_speech_intervals_vad"] = [
        (float(segment["start"]), float(segment["end"]))
        for segment in speech_timestamps
        if safe_float(segment.get("end")) > safe_float(segment.get("start"))
    ]

    if not speech_timestamps:
        result["audio_qc_status"] = "NO_SPEECH"
        result["audio_qc_error"] = "Silero VAD detected no speech"
        result["silence_duration_sec_vad"] = round(vad_audio.size / silero_sr, 2)
        result["silence_floor_dbfs"] = round(rms_dbfs(vad_audio), 2)
        return apply_audio_thresholds_and_confidence(result)

    speech_mask = build_speech_mask_from_seconds(
        length_samples=len(vad_audio),
        speech_timestamps=speech_timestamps,
        sample_rate=silero_sr,
    )

    speech_samples = vad_audio[speech_mask]
    silence_samples = vad_audio[~speech_mask]

    result["speech_duration_sec_vad"] = round(speech_samples.size / silero_sr, 2)
    result["silence_duration_sec_vad"] = round(silence_samples.size / silero_sr, 2)

    if speech_samples.size:
        result["speech_level_dbfs"] = round(rms_dbfs(speech_samples), 2)

    if silence_samples.size:
        result["silence_floor_dbfs"] = round(rms_dbfs(silence_samples), 2)

    if speech_samples.size < 100:
        result["audio_qc_status"] = "INSUFFICIENT_SPEECH"
        result["audio_qc_error"] = "Silero VAD detected too little speech"
        return apply_audio_thresholds_and_confidence(result)

    if silence_samples.size < 100:
        result["audio_qc_status"] = "INSUFFICIENT_SILENCE"
        result["audio_qc_error"] = "Silero VAD detected too little non-speech audio"
        return apply_audio_thresholds_and_confidence(result)

    result["snr_vad_db"] = round(
        result["speech_level_dbfs"] - result["silence_floor_dbfs"], 2
    )
    result["audio_qc_status"] = "OK"

    return apply_audio_thresholds_and_confidence(result)


# -----------------------------
# Google Drive helpers
# -----------------------------


def extract_drive_id(link: str) -> Optional[str]:
    patterns = [
        r"/folders/([a-zA-Z0-9_-]+)",
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, link)

        if match:
            return match.group(1)

    return None


def is_drive_folder_link(link: str) -> bool:
    return "/folders/" in link


def collect_supported_files(root_dir: str) -> List[str]:
    collected = []

    for root, _, files in os.walk(root_dir):
        for file in files:
            path = os.path.join(root, file)

            if Path(path).suffix.lower() in SUPPORTED_EXTENSIONS:
                collected.append(path)

    return sorted(collected)


@contextmanager
def gdown_request_timeout(seconds: int):
    import requests

    original_request = requests.sessions.Session.request

    def request_with_timeout(self, method, url, **kwargs):
        kwargs.setdefault("timeout", seconds)
        return original_request(self, method, url, **kwargs)

    requests.sessions.Session.request = request_with_timeout

    try:
        yield
    finally:
        requests.sessions.Session.request = original_request


def download_from_drive(
    link: str,
    output_dir: str,
    status_callback=None,
    progress_callback=None,
) -> List[str]:
    if gdown is None:
        raise RuntimeError("gdown is not installed. Run: pip install gdown")

    def update_status(message: str) -> None:
        if status_callback:
            status_callback(message)

    with gdown_request_timeout(DRIVE_REQUEST_TIMEOUT_SECONDS):
        if is_drive_folder_link(link):
            update_status("Reading Google Drive folder contents...")
            drive_files = gdown.download_folder(
                link,
                output=output_dir,
                quiet=True,
                use_cookies=False,
                skip_download=True,
            )

            supported_drive_files = [
                drive_file
                for drive_file in drive_files
                if Path(drive_file.local_path).suffix.lower() in SUPPORTED_EXTENSIONS
            ]

            if not supported_drive_files:
                return []

            downloaded_paths = []
            total_files = len(supported_drive_files)

            for index, drive_file in enumerate(supported_drive_files, start=1):
                local_path = drive_file.local_path
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                file_name = Path(drive_file.path).name
                update_status(f"Downloading {index}/{total_files}: {file_name}")

                def progress(
                    bytes_downloaded,
                    total_size,
                    current_index=index,
                    current_file=file_name,
                ):
                    if progress_callback:
                        progress_callback(
                            current_index,
                            total_files,
                            current_file,
                            bytes_downloaded,
                            total_size,
                        )

                downloaded_path = gdown.download(
                    id=drive_file.id,
                    output=local_path,
                    quiet=True,
                    use_cookies=False,
                    progress=progress,
                )

                if downloaded_path:
                    downloaded_paths.append(downloaded_path)

            update_status("Google Drive download completed.")
            return collect_supported_files(output_dir)

        file_id = extract_drive_id(link)

        if not file_id:
            raise ValueError("Invalid Google Drive link")

        update_status("Downloading Google Drive file...")

        def progress(bytes_downloaded, total_size):
            if progress_callback:
                progress_callback(1, 1, "Drive file", bytes_downloaded, total_size)

        downloaded_path = gdown.download(
            id=file_id,
            output=output_dir + os.sep,
            quiet=True,
            use_cookies=False,
            progress=progress,
        )

        if downloaded_path and Path(downloaded_path).suffix.lower() in SUPPORTED_EXTENSIONS:
            return [downloaded_path]

    return collect_supported_files(output_dir)


# -----------------------------
# File selection / pairing
# -----------------------------


def choose_best_json(json_paths: Sequence[str]) -> Optional[str]:
    if not json_paths:
        return None

    preferred = [
        path
        for path in json_paths
        if re.search(
            r"assembly|assemblyai|transcript|diar|words|word",
            Path(path).stem.lower(),
        )
    ]

    candidates = preferred if preferred else list(json_paths)
    return max(candidates, key=lambda path: os.path.getsize(path))


def choose_best_txt(txt_paths: Sequence[str]) -> Optional[str]:
    if not txt_paths:
        return None

    preferred = [
        path
        for path in txt_paths
        if re.search(r"riverside|transcript", Path(path).stem.lower())
    ]

    candidates = preferred if preferred else list(txt_paths)
    return max(candidates, key=lambda path: os.path.getsize(path))


def conversation_key_from_paths(paths: Sequence[str], fallback: str) -> str:
    if not paths:
        return fallback

    try:
        common = Path(os.path.commonpath(paths))
        key = common.name if common.is_dir() else common.parent.name
    except Exception:
        key = fallback

    return key or fallback


def pair_files(paths: Sequence[str]) -> List[Dict]:
    wavs = [path for path in paths if Path(path).suffix.lower() == ".wav"]
    jsons = [path for path in paths if Path(path).suffix.lower() == ".json"]
    txts = [path for path in paths if Path(path).suffix.lower() == ".txt"]

    if not paths:
        return []

    if len(jsons) <= 1 and len(txts) <= 1:
        return [
            {
                "conversation_key": conversation_key_from_paths(paths, "conversation"),
                "wav_paths": sorted(wavs),
                "json_path": jsons[0] if jsons else None,
                "txt_path": txts[0] if txts else None,
                "all_jsons": jsons,
                "all_txts": txts,
            }
        ]

    folder_groups = defaultdict(lambda: {"wavs": [], "jsons": [], "txts": []})

    for path in paths:
        suffix = Path(path).suffix.lower()
        parent = str(Path(path).parent)

        if suffix == ".wav":
            folder_groups[parent]["wavs"].append(path)
        elif suffix == ".json":
            folder_groups[parent]["jsons"].append(path)
        elif suffix == ".txt":
            folder_groups[parent]["txts"].append(path)

    pairs = []

    for folder, group in folder_groups.items():
        if not group["wavs"] and not group["jsons"] and not group["txts"]:
            continue

        pairs.append(
            {
                "conversation_key": Path(folder).name or "conversation",
                "wav_paths": sorted(group["wavs"]),
                "json_path": choose_best_json(group["jsons"]),
                "txt_path": choose_best_txt(group["txts"]),
                "all_jsons": group["jsons"],
                "all_txts": group["txts"],
            }
        )

    return sorted(pairs, key=lambda item: item["conversation_key"])


# -----------------------------
# Riverside speaker-name reference
# -----------------------------


def add_name_mapping(mapping: Dict[str, str], key: str, value: str) -> None:
    if not key or not value:
        return

    variants = {
        key,
        key.strip(),
        key.lower().strip(),
        normalize_key(key),
    }

    for variant in variants:
        if variant:
            mapping[variant] = value


def parse_riverside_names(txt_path: Optional[str]) -> Dict[str, str]:
    if not txt_path or not Path(txt_path).exists():
        return {}

    try:
        text = Path(txt_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {}

    timestamp = r"\[?\(?\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\)?\]?"
    speaker_counts = Counter()
    speaker_order = []

    lines = [line.strip() for line in text.splitlines()]

    for line_index, line in enumerate(lines):
        if not line:
            continue

        candidates = []

        parsed = parse_riverside_line(line)
        if parsed and parsed.get("speaker"):
            candidates.append(parsed["speaker"])

        voice_match = re.search(r"<v\s+([^>]+)>", line, flags=re.I)
        if voice_match:
            candidates.append(voice_match.group(1).strip())

        header_match = re.match(
            r"^(.+?)\s*[\[(]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?[\])]?\s*$",
            line,
        )
        if header_match:
            candidates.append(header_match.group(1).strip())

        colon_match = re.match(
            rf"^([^\[:()]+?)\s*(?:{timestamp})?\s*:\s+",
            line,
        )
        if colon_match:
            candidates.append(colon_match.group(1).strip())

        timestamp_match = re.match(
            rf"^(.+?)\s+{timestamp}(?:\s|$)",
            line,
        )
        if timestamp_match:
            candidates.append(timestamp_match.group(1).strip())

        speaker_line_match = re.match(
            rf"^({timestamp})\s+(.+?)\s*:\s+",
            line,
        )
        if speaker_line_match:
            candidates.append(speaker_line_match.group(2).strip())

        speaker_name_match = re.match(
            r"^(?:speaker|spk|participant)\s*[_\-\s]*(\d+|[A-Z])\s*[-:]\s*([A-Za-z][A-Za-z0-9 ._'()-]{1,80})(?:$|\s{2,})",
            line,
            flags=re.I,
        )
        if speaker_name_match:
            candidates.append(speaker_name_match.group(2).strip())

        speaker_plain_name_match = re.match(
            r"^(?:speaker|spk|participant)\s*[_\-\s]*(\d+|[A-Z])\s+([A-Za-z][A-Za-z0-9 ._'()-]{1,80})$",
            line,
            flags=re.I,
        )
        if speaker_plain_name_match and count_text_words(line) <= 8:
            candidates.append(speaker_plain_name_match.group(2).strip())

        speaker_paren_match = re.match(
            r"^(?:speaker|spk|participant)\s*[_\-\s]*[A-Z0-9]+\s*\(([^)]+)\)",
            line,
            flags=re.I,
        )
        if speaker_paren_match:
            candidates.append(speaker_paren_match.group(1).strip())

        name_then_generic_match = re.match(
            r"^([A-Za-z][A-Za-z0-9 ._'()-]{1,80})\s*\((?:speaker|spk|participant)\s*[_\-\s]*[A-Z0-9]+\)",
            line,
            flags=re.I,
        )
        if name_then_generic_match:
            candidates.append(name_then_generic_match.group(1).strip())

        name_dash_generic_match = re.match(
            r"^([A-Za-z][A-Za-z0-9 ._'()-]{1,80})\s*[-\u2013\u2014]\s*(?:speaker|spk|participant)\s*[_\-\s]*[A-Z0-9]+$",
            line,
            flags=re.I,
        )
        if name_dash_generic_match:
            candidates.append(name_dash_generic_match.group(1).strip())

        line_has_timestamp = parse_timestamp_to_seconds(line) is not None or bool(re.search(r"\d{1,2}:\d{2}(?::\d{2})?", line))
        timestamp_line = r"\[?\(?\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\)?\]?(?:\s*(?:-->|-|to)\s*\[?\(?\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\)?\]?)?"
        previous_is_timestamp_only = line_index > 0 and bool(
            re.fullmatch(timestamp_line, lines[line_index - 1], flags=re.I)
        )
        next_is_timestamp_only = line_index + 1 < len(lines) and bool(
            re.fullmatch(timestamp_line, lines[line_index + 1], flags=re.I)
        )
        short_name_like_line = (
            not line_has_timestamp
            and ":" not in line
            and len(line) <= 80
            and count_text_words(line) <= 6
            and re.search(r"[A-Za-z]", line)
            and not re.search(r"[.!?]{1,}$", line)
        )

        line_two_before = lines[line_index - 2] if line_index >= 2 else ""
        line_two_before_looks_like_name = (
            bool(line_two_before)
            and ":" not in line_two_before
            and parse_timestamp_to_seconds(line_two_before) is None
            and len(line_two_before) <= 80
            and count_text_words(line_two_before) <= 6
            and re.search(r"[A-Za-z]", line_two_before)
        )
        previous_line = lines[line_index - 1] if line_index >= 1 else ""
        previous_line_looks_like_name_after_timestamp = (
            bool(previous_line)
            and ":" not in previous_line
            and parse_timestamp_to_seconds(previous_line) is None
            and len(previous_line) <= 80
            and count_text_words(previous_line) <= 6
            and re.search(r"[A-Za-z]", previous_line)
            and line_index >= 2
            and bool(re.fullmatch(timestamp_line, lines[line_index - 2], flags=re.I))
        )

        if short_name_like_line and next_is_timestamp_only and not previous_line_looks_like_name_after_timestamp:
            candidates.append(line)

        if short_name_like_line and previous_is_timestamp_only and not line_two_before_looks_like_name:
            candidates.append(line)

        for candidate in candidates:
            candidate = clean_speaker_name(candidate)
            lowered = candidate.lower()

            if lowered in {"transcript", "note", "notes"} or is_generic_speaker_label(candidate):
                continue

            speaker_counts[candidate] += 1
            if candidate not in speaker_order:
                speaker_order.append(candidate)

    names = speaker_order or [name for name, _ in speaker_counts.most_common()]
    mapping: Dict[str, str] = {}

    for idx, name in enumerate(names):
        zero_based = str(idx)
        one_based = str(idx + 1)
        letter = chr(ord("A") + idx)
        lower_letter = letter.lower()

        for key in {
            letter,
            lower_letter,
            zero_based,
            one_based,
            f"speaker_{lower_letter}",
            f"speaker {lower_letter}",
            f"speaker_{letter}",
            f"speaker {letter}",
            f"Speaker {letter}",
            f"speaker_{zero_based}",
            f"speaker {zero_based}",
            f"Speaker {zero_based}",
            f"speaker_{one_based}",
            f"speaker {one_based}",
            f"Speaker {one_based}",
            f"spk_{zero_based}",
            f"spk {zero_based}",
            f"spk_{one_based}",
            f"spk {one_based}",
            f"participant_{one_based}",
            f"participant {one_based}",
            f"channel_{one_based}",
            f"channel {one_based}",
            f"ch_{one_based}",
            f"ch {one_based}",
            name,
        }:
            add_name_mapping(mapping, key, name)

    return mapping


def parse_riverside_line(line: str) -> Optional[Dict]:
    timestamp = r"\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?"
    line = line.strip()

    if not line:
        return None

    voice_match = re.match(rf"^<v\s+(?P<speaker>[^>]+)>\s*(?P<text>.*?)(?:</v>)?$", line, flags=re.I)
    if voice_match:
        speaker = clean_speaker_name(voice_match.group("speaker"))
        text = voice_match.group("text").strip()

        if speaker and speaker.lower() not in {"transcript", "note", "notes"}:
            return {"speaker": speaker, "start": None, "end": None, "text": text}

    patterns = [
        rf"^\[?(?P<start>{timestamp})\]?\s*(?:-->|-|to)\s*\[?(?P<end>{timestamp})\]?\s+(?P<speaker>.+?)\s*:\s*(?P<text>.*)$",
        rf"^\[?(?P<start>{timestamp})\]?\s*[-:]\s*(?P<speaker>.+?)\s*:\s*(?P<text>.*)$",
        rf"^\[?(?P<start>{timestamp})\]?\s+(?P<speaker>.+?)\s*:\s*(?P<text>.*)$",
        rf"^\[?(?P<start>{timestamp})\]?\s+(?:speaker|spk|participant)\s*[_\-\s]*[A-Z0-9]+\s*[-:]\s*(?P<speaker>.+?)\s*:\s*(?P<text>.*)$",
        rf"^(?P<speaker>.+?)\s*[\[(](?P<start>{timestamp})(?:\s*-\s*(?P<end>{timestamp}))?[\])]\s*:?\s*(?P<text>.*)$",
        rf"^(?P<speaker>.+?)\s+(?P<start>{timestamp})(?:\s*-\s*(?P<end>{timestamp}))?\s*:?\s*(?P<text>.*)$",
        rf"^(?P<speaker>.+?)\s*:\s*(?P<text>.+)$",
    ]

    for pattern in patterns:
        match = re.match(pattern, line)

        if not match:
            continue

        groups = match.groupdict()
        speaker = clean_speaker_name(groups.get("speaker") or "")
        text = (groups.get("text") or "").strip()
        start = parse_timestamp_to_seconds(groups.get("start", ""))
        end = parse_timestamp_to_seconds(groups.get("end", ""))

        if not speaker:
            continue

        if speaker.lower() in {"transcript", "note", "notes"}:
            continue

        return {
            "speaker": speaker,
            "start": start,
            "end": end,
            "text": text,
        }

    return None


def parse_riverside_turns(txt_path: Optional[str]) -> List[Dict]:
    if not txt_path or not Path(txt_path).exists():
        return []

    try:
        text = Path(txt_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    lines = [line.strip() for line in text.splitlines()]
    timestamp_line = r"\[?\(?\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\)?\]?(?:\s*(?:-->|-|to)\s*\[?\(?\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\)?\]?)?"

    def timestamp_from_line(line: str) -> Optional[float]:
        match = re.search(r"\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?", line)
        return parse_timestamp_to_seconds(match.group(0)) if match else None

    def is_timestamp_line(line: str) -> bool:
        return bool(re.fullmatch(timestamp_line, line, flags=re.I))

    def name_like(line: str) -> Optional[str]:
        if not line or ":" in line or is_timestamp_line(line):
            return None

        if len(line) > 80 or count_text_words(line) > 6:
            return None

        if not re.search(r"[A-Za-z]", line) or re.search(r"[.!?]{1,}$", line):
            return None

        name = clean_speaker_name(line)

        if not name or is_generic_speaker_label(name):
            return None

        return name

    def starts_structured_turn(index: int, expected_layout: Optional[str] = None) -> bool:
        if index < 0 or index >= len(lines):
            return False

        current_line = lines[index]
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        current_name_next_timestamp = bool(name_like(current_line) and is_timestamp_line(next_line))
        current_timestamp_next_name = bool(is_timestamp_line(current_line) and name_like(next_line))

        if expected_layout == "name_timestamp":
            return current_name_next_timestamp

        if expected_layout == "timestamp_name":
            return current_timestamp_next_name

        return bool(
            current_name_next_timestamp
            or current_timestamp_next_name
            or (parse_riverside_line(current_line) and parse_riverside_line(current_line).get("start") is not None)
        )

    raw_turns = []
    index = 0

    while index < len(lines):
        line = lines[index]

        if not line:
            index += 1
            continue

        speaker = None
        start = None
        end = None
        text_lines = []
        layout = None

        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        line_name = name_like(line)
        next_name = name_like(next_line)

        if line_name and is_timestamp_line(next_line):
            speaker = line_name
            start = timestamp_from_line(next_line)
            layout = "name_timestamp"
            index += 2
        elif is_timestamp_line(line) and next_name:
            speaker = next_name
            start = timestamp_from_line(line)
            layout = "timestamp_name"
            index += 2
        else:
            parsed = parse_riverside_line(line)

            if parsed:
                speaker = parsed.get("speaker")
                start = parsed.get("start")
                end = parsed.get("end")

                if parsed.get("text"):
                    text_lines.append(parsed["text"])

                index += 1
            else:
                index += 1
                continue

        while index < len(lines) and not starts_structured_turn(index, layout):
            if lines[index]:
                text_lines.append(lines[index])

            index += 1

        if speaker:
            raw_turns.append(
                {
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "text": " ".join(text_lines).strip(),
                }
            )

    timed_turns = [turn for turn in raw_turns if turn.get("start") is not None]
    timed_turns.sort(key=lambda turn: (turn["start"], turn["speaker"]))

    for index, turn in enumerate(timed_turns):
        if turn.get("end") is not None and turn["end"] > turn["start"]:
            inferred_end = turn["end"]
        else:
            word_count = max(1, count_text_words(turn.get("text", "")))
            estimated_duration = max(0.35, word_count / RIVERSIDE_WORDS_PER_SECOND)
            estimated_end = turn["start"] + estimated_duration
            next_start = (
                timed_turns[index + 1]["start"]
                if index + 1 < len(timed_turns)
                else None
            )

            if next_start is not None:
                inferred_end = min(
                    estimated_end,
                    next_start + RIVERSIDE_MAX_INFERRED_OVERLAP_TAIL_SECONDS,
                )
                inferred_end = max(inferred_end, turn["start"] + 0.2)
            else:
                inferred_end = estimated_end

        turn["end"] = inferred_end
        turn["duration"] = max(0.0, inferred_end - turn["start"])
        turn["word_count"] = count_text_words(turn.get("text", ""))
        turn["source"] = "riverside_txt"

    return timed_turns


def map_speaker_name(raw_speaker: str, speaker_name_map: Dict[str, str]) -> str:
    raw_speaker = str(raw_speaker or "unknown").strip() or "unknown"

    for key in (raw_speaker, raw_speaker.lower(), normalize_key(raw_speaker)):
        if key in speaker_name_map:
            return speaker_name_map[key]

    return raw_speaker


def ordered_mapped_names(speaker_name_map: Dict[str, str]) -> List[str]:
    names = []

    for name in speaker_name_map.values():
        if name not in names:
            names.append(name)

    return names


def add_ordered_speaker_mappings(
    speaker_name_map: Dict[str, str],
    names: Sequence[str],
) -> None:
    for idx, raw_name in enumerate(names):
        name = clean_speaker_name(raw_name)

        if not name or is_generic_speaker_label(name):
            continue

        zero_based = str(idx)
        one_based = str(idx + 1)
        letter = chr(ord("A") + idx)
        lower_letter = letter.lower()

        for key in {
            letter,
            lower_letter,
            zero_based,
            one_based,
            f"speaker_{letter}",
            f"speaker {letter}",
            f"speaker_{lower_letter}",
            f"speaker {lower_letter}",
            f"speaker_{zero_based}",
            f"speaker {zero_based}",
            f"speaker_{one_based}",
            f"speaker {one_based}",
            f"spk_{zero_based}",
            f"spk {zero_based}",
            f"spk_{one_based}",
            f"spk {one_based}",
            f"participant_{one_based}",
            f"participant {one_based}",
            f"channel_{one_based}",
            f"channel {one_based}",
            f"ch_{one_based}",
            f"ch {one_based}",
            name,
        }:
            add_name_mapping(speaker_name_map, key, name)


def ordered_riverside_names_from_turns(turns: Sequence[Dict]) -> List[str]:
    return ordered_unique(
        clean_speaker_name(turn.get("speaker", ""))
        for turn in turns
        if turn.get("speaker") and not is_generic_speaker_label(turn.get("speaker"))
    )


def build_riverside_name_reference(
    txt_path: Optional[str],
) -> Tuple[Dict[str, str], List[Dict], List[str]]:
    speaker_name_map = parse_riverside_names(txt_path)
    riverside_turns = parse_riverside_turns(txt_path)
    riverside_names = ordered_riverside_names_from_turns(riverside_turns)

    if riverside_names:
        add_ordered_speaker_mappings(speaker_name_map, riverside_names)

    for turn in riverside_turns:
        turn["speaker"] = map_speaker_name(turn["speaker"], speaker_name_map)

    names = ordered_unique(
        [
            *ordered_riverside_names_from_turns(riverside_turns),
            *ordered_mapped_names(speaker_name_map),
        ]
    )

    return speaker_name_map, riverside_turns, names


def infer_assembly_to_riverside_mapping_by_text(
    words: Sequence[Dict],
    riverside_turns: Sequence[Dict],
    speaker_name_map: Dict[str, str],
) -> Dict[str, str]:
    names = ordered_riverside_names_from_turns(riverside_turns) or ordered_mapped_names(speaker_name_map)

    if not words or not names:
        return {}

    label_text = defaultdict(list)

    for word in words:
        label = str(word.get("speaker") or "").strip()

        if not label:
            continue

        label_text[label].append(str(word.get("word") or ""))

    label_order = sorted(label_text.keys(), key=lambda item: normalize_key(item) or item)
    label_bags = {
        label: token_bag(" ".join(tokens))
        for label, tokens in label_text.items()
    }
    name_text = defaultdict(list)

    for turn in riverside_turns:
        name = clean_speaker_name(turn.get("speaker", ""))

        if not name or is_generic_speaker_label(name):
            continue

        name_text[name].append(str(turn.get("text") or ""))

    name_bags = {
        name: token_bag(" ".join(parts))
        for name, parts in name_text.items()
    }

    mapping = {}
    used_names = set()
    scored_pairs = []

    for label in label_order:
        label_bag = label_bags.get(label, Counter())

        for name in names:
            score = cosine_similarity(label_bag, name_bags.get(name, Counter()))

            scored_pairs.append((score, label, name))

    for score, label, name in sorted(scored_pairs, reverse=True):
        if label in mapping or name in used_names:
            continue

        if score >= 0.01:
            mapping[label] = name
            used_names.add(name)

    remaining_names = [name for name in names if name not in used_names]

    for label in label_order:
        if label in mapping:
            continue

        if not is_generic_speaker_label(label):
            continue

        if remaining_names:
            mapping[label] = remaining_names.pop(0)

    for label, name in mapping.items():
        add_name_mapping(speaker_name_map, label, name)

    return mapping


def apply_speaker_mapping_to_words(words: Sequence[Dict], mapping: Dict[str, str]) -> None:
    for word in words:
        raw_label = str(word.get("speaker") or "").strip()

        if raw_label in mapping:
            word["speaker_name"] = mapping[raw_label]
        elif normalize_key(raw_label) in mapping:
            word["speaker_name"] = mapping[normalize_key(raw_label)]


def fill_generic_speaker_names_from_riverside_order(
    words: Sequence[Dict],
    riverside_names: Sequence[str],
) -> Dict[str, str]:
    names = [
        clean_speaker_name(name)
        for name in riverside_names
        if clean_speaker_name(name) and not is_generic_speaker_label(name)
    ]

    if not words or not names:
        return {}

    used_names = {
        word.get("speaker_name")
        for word in words
        if word.get("speaker_name") and not is_generic_speaker_label(word.get("speaker_name"))
    }
    remaining_names = [name for name in ordered_unique(names) if name not in used_names]
    raw_labels = ordered_unique(word.get("speaker", "") for word in words)
    fallback_mapping = {}

    for raw_label in raw_labels:
        if not remaining_names:
            break

        current_name = next(
            (
                word.get("speaker_name")
                for word in words
                if str(word.get("speaker") or "").strip() == str(raw_label or "").strip()
            ),
            raw_label,
        )

        if is_generic_speaker_label(current_name):
            fallback_mapping[str(raw_label).strip()] = remaining_names.pop(0)

    if fallback_mapping:
        apply_speaker_mapping_to_words(words, fallback_mapping)

    return fallback_mapping


def speaker_mapping_status(words: Sequence[Dict]) -> str:
    speakers = ordered_unique(word.get("speaker_name", "") for word in words)
    unmapped = [speaker for speaker in speakers if is_generic_speaker_label(speaker)]

    if not unmapped:
        return "MAPPED"

    return f"UNMAPPED: {', '.join(unmapped)}"


def mapped_speaker_names(words: Sequence[Dict]) -> str:
    speakers = ordered_unique(word.get("speaker_name", "") for word in words)
    names = [speaker for speaker in speakers if speaker and not is_generic_speaker_label(speaker)]

    return ", ".join(names)


# -----------------------------
# AssemblyAI JSON parsing
# -----------------------------


def normalize_assembly_time(value) -> float:
    return safe_float(value) / 1000.0


def extract_word_text(item: Dict) -> str:
    return str(
        item.get("text")
        or item.get("word")
        or item.get("punctuated_word")
        or ""
    ).strip()


def extract_raw_words_from_assembly(data) -> List[Dict]:
    if isinstance(data, dict) and isinstance(data.get("utterances"), list):
        raw_words = []

        for utterance_index, utterance in enumerate(data["utterances"]):
            if not isinstance(utterance, dict):
                continue

            utterance_speaker = str(utterance.get("speaker", "")).strip()
            utterance_start = utterance.get("start")
            utterance_end = utterance.get("end")

            for word in utterance.get("words", []):
                if not isinstance(word, dict):
                    continue

                copied = dict(word)
                copied["_utterance_index"] = utterance_index
                copied["_utterance_start"] = utterance_start
                copied["_utterance_end"] = utterance_end

                if not copied.get("speaker"):
                    copied["speaker"] = utterance_speaker

                raw_words.append(copied)

        if raw_words:
            return raw_words

    if isinstance(data, dict) and isinstance(data.get("words"), list):
        return [word for word in data["words"] if isinstance(word, dict)]

    if isinstance(data, list):
        return [word for word in data if isinstance(word, dict)]

    return []


def extract_words_from_assembly_json(
    json_path: str,
    speaker_name_map: Dict[str, str],
) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8", errors="ignore") as file:
        data = json.load(file)

    raw_word_items = extract_raw_words_from_assembly(data)
    raw_speaker_order = []

    for item in raw_word_items:
        raw_speaker = (
            item.get("speaker")
            or item.get("speaker_label")
            or item.get("speaker_id")
            or item.get("channel")
            or ""
        )
        raw_speaker = str(raw_speaker).strip()

        if raw_speaker and raw_speaker not in raw_speaker_order:
            raw_speaker_order.append(raw_speaker)

    mapped_names = ordered_mapped_names(speaker_name_map)

    for index, raw_speaker in enumerate(raw_speaker_order):
        if index >= len(mapped_names):
            break

        if map_speaker_name(raw_speaker, speaker_name_map) == raw_speaker and is_generic_speaker_label(raw_speaker):
            add_name_mapping(speaker_name_map, raw_speaker, mapped_names[index])

    words = []

    for idx, item in enumerate(raw_word_items):
        text = extract_word_text(item)

        if not text:
            continue

        start_raw = item.get("start")
        end_raw = item.get("end")

        if start_raw is None or end_raw is None:
            continue

        start = normalize_assembly_time(start_raw)
        end = normalize_assembly_time(end_raw)

        if end <= start:
            continue

        speaker = (
            item.get("speaker")
            or item.get("speaker_label")
            or item.get("speaker_id")
            or item.get("channel")
            or "unknown"
        )
        speaker = str(speaker).strip() or "unknown"
        speaker_name = map_speaker_name(speaker, speaker_name_map)

        words.append(
            {
                "index": idx,
                "word": text,
                "clean_word": clean_word(text),
                "start": float(start),
                "end": float(end),
                "duration": float(end - start),
                "speaker": speaker,
                "speaker_name": speaker_name,
                "_utterance_index": item.get("_utterance_index"),
                "_utterance_start": (
                    normalize_assembly_time(item.get("_utterance_start"))
                    if item.get("_utterance_start") is not None
                    else None
                ),
                "_utterance_end": (
                    normalize_assembly_time(item.get("_utterance_end"))
                    if item.get("_utterance_end") is not None
                    else None
                ),
            }
        )

    return sorted(words, key=lambda word: (word["start"], word["end"], word["speaker_name"]))


# -----------------------------
# Transcript QC calculations
# -----------------------------


def build_speaker_speech_regions(
    words: Sequence[Dict],
    max_internal_pause_seconds: float,
) -> Dict[str, List[Dict]]:
    by_speaker = defaultdict(list)

    for word in words:
        by_speaker[word["speaker_name"]].append(word)

    speaker_regions = {}

    for speaker, speaker_words in by_speaker.items():
        sorted_words = sorted(speaker_words, key=lambda word: (word["start"], word["end"]))

        if not sorted_words:
            continue

        regions = []
        current_words = [sorted_words[0]]
        current_start = sorted_words[0]["start"]
        current_end = sorted_words[0]["end"]

        for word in sorted_words[1:]:
            gap = word["start"] - current_end

            if gap <= max_internal_pause_seconds:
                current_words.append(word)
                current_end = max(current_end, word["end"])
                continue

            regions.append(build_turn(speaker, current_start, current_end, current_words))
            current_words = [word]
            current_start = word["start"]
            current_end = word["end"]

        regions.append(build_turn(speaker, current_start, current_end, current_words))
        speaker_regions[speaker] = regions

    return speaker_regions


def speaker_regions_to_intervals(
    speaker_regions: Dict[str, List[Dict]]
) -> Dict[str, List[Tuple[float, float]]]:
    return {
        speaker: [(region["start"], region["end"]) for region in regions]
        for speaker, regions in speaker_regions.items()
    }


def build_speaker_word_intervals(words: Sequence[Dict]) -> Dict[str, List[Tuple[float, float]]]:
    speaker_intervals = defaultdict(list)

    for word in words:
        speaker_intervals[word["speaker_name"]].append((word["start"], word["end"]))

    return {
        speaker: merge_intervals(intervals)
        for speaker, intervals in speaker_intervals.items()
    }


def conversation_bounds(words: Sequence[Dict]) -> Tuple[float, float, float]:
    if not words:
        raise ValueError("No word-level timestamps found")

    start = min(word["start"] for word in words)
    end = max(word["end"] for word in words)
    duration = end - start

    if duration <= 0:
        raise ValueError("Invalid conversation duration from JSON word timestamps")

    return start, end, duration


def calculate_density(
    words: Sequence[Dict],
    max_internal_pause_seconds: float,
) -> Tuple[pd.DataFrame, Dict]:
    _, _, conversation_length = conversation_bounds(words)

    utterance_spans = {}

    for word in words:
        utterance_index = word.get("_utterance_index")
        utterance_start = word.get("_utterance_start")
        utterance_end = word.get("_utterance_end")

        if (
            utterance_index is None
            or utterance_start is None
            or utterance_end is None
            or utterance_end <= utterance_start
        ):
            continue

        key = (word["speaker_name"], utterance_index)
        utterance_spans[key] = (utterance_start, utterance_end)

    speaker_seconds = defaultdict(float)
    density_timing_source = "assembly_utterance_duration_sum"

    if utterance_spans:
        for (speaker, _), (start, end) in utterance_spans.items():
            speaker_seconds[speaker] += max(0.0, end - start)
    else:
        density_timing_source = "assembly_word_duration_sum"

        for word in words:
            speaker = word["speaker_name"]
            speaker_seconds[speaker] += max(0.0, word["end"] - word["start"])

    speakers = sorted(speaker_seconds.keys())
    participant_count = len(speakers)

    if participant_count == 0:
        raise ValueError("No speakers detected in word-level JSON")

    expected_pct = 100.0 / participant_count
    min_allowed_pct = 0.5 * expected_pct
    max_allowed_pct = 1.5 * expected_pct if participant_count == 2 else 2.0 * expected_pct

    rows = []
    review_speakers = []

    for speaker in speakers:
        speaking_time = speaker_seconds[speaker]
        speaking_pct = (speaking_time / conversation_length) * 100.0
        status = "PASS" if min_allowed_pct <= speaking_pct <= max_allowed_pct else "REVIEW"

        if status == "REVIEW":
            review_speakers.append(speaker)

        rows.append(
            {
                "timing_source": density_timing_source,
                "speaker": speaker,
                "speaking_time": format_mmss(speaking_time),
                "speaking_time_sec": round(speaking_time, 2),
                "speaking_pct_of_duration": round(speaking_pct, 2),
                "expected_pct": round(expected_pct, 2),
                "min_allowed_pct": round(min_allowed_pct, 2),
                "max_allowed_pct": round(max_allowed_pct, 2),
                "density_status": status,
            }
        )

    rows.sort(key=lambda row: row["speaking_time_sec"], reverse=True)

    summary = {
        "duration_min_from_timing": seconds_to_minutes(conversation_length),
        "participant_count": participant_count,
        "density_expected_pct": round(expected_pct, 2),
        "density_min_allowed_pct": round(min_allowed_pct, 2),
        "density_max_allowed_pct": round(max_allowed_pct, 2),
        "density_status": "PASS" if not review_speakers else "REVIEW",
        "density_review_speakers": ", ".join(review_speakers),
    }

    return pd.DataFrame(rows), summary


def calculate_interval_overlap(
    speaker_intervals: Dict[str, List[Tuple[float, float]]],
    denominator_seconds: float,
    source: str,
) -> Tuple[Dict, pd.DataFrame]:
    if denominator_seconds <= 0:
        return {
            "overlap_source": source,
            "overlap_segment_count": 0,
            "overlap_duration_sec": 0.0,
            "overlap_duration_min": 0.0,
            "overlap_ratio_pct": 0.0,
            "two_speaker_overlap_sec": 0.0,
            "two_speaker_overlap_min": 0.0,
            "three_speaker_overlap_sec": 0.0,
            "three_speaker_overlap_min": 0.0,
            "four_plus_speaker_overlap_sec": 0.0,
            "four_plus_speaker_overlap_min": 0.0,
        }, pd.DataFrame()

    events = []
    all_intervals = []

    for speaker, intervals in speaker_intervals.items():
        for start, end in merge_intervals(intervals):
            if end > start:
                all_intervals.append((start, end))
                events.append((start, "start", speaker))
                events.append((end, "end", speaker))

    events.sort(key=lambda item: item[0])
    non_silence_duration = interval_duration(merge_intervals(all_intervals))
    ratio_denominator = non_silence_duration if non_silence_duration > 0 else denominator_seconds
    active = set()
    previous_time = events[0][0] if events else 0.0
    rows = []
    duration_by_count = defaultdict(float)

    index = 0
    while index < len(events):
        current_time = events[index][0]

        if current_time > previous_time and len(active) >= 2:
            active_speakers = sorted(active)
            active_count = len(active_speakers)
            duration = current_time - previous_time
            duration_by_count[active_count] += duration

            rows.append(
                {
                    "overlap_source": source,
                    "overlap_start_sec": round(previous_time, 3),
                    "overlap_end_sec": round(current_time, 3),
                    "overlap_start": format_seconds(previous_time),
                    "overlap_end": format_seconds(current_time),
                    "overlap_duration_sec": round(duration, 3),
                    "overlap_duration_min": seconds_to_minutes(duration),
                    "active_speaker_count": active_count,
                    "active_speakers": ", ".join(active_speakers),
                }
            )

        same_time_events = []

        while index < len(events) and events[index][0] == current_time:
            same_time_events.append(events[index])
            index += 1

        for _, event_type, speaker in same_time_events:
            if event_type == "end":
                active.discard(speaker)

        for _, event_type, speaker in same_time_events:
            if event_type == "start":
                active.add(speaker)

        previous_time = current_time

    overlap_duration = sum(duration_by_count.values())

    summary = {
        "overlap_source": source,
        "overlap_ratio_denominator": "total_non_silence_time",
        "non_silence_duration_sec": round(ratio_denominator, 2),
        "non_silence_duration_min": seconds_to_minutes(ratio_denominator),
        "overlap_segment_count": len(rows),
        "overlap_duration_sec": round(overlap_duration, 2),
        "overlap_duration_min": seconds_to_minutes(overlap_duration),
        "overlap_ratio_pct": round((overlap_duration / ratio_denominator) * 100.0, 2),
        "two_speaker_overlap_sec": round(duration_by_count.get(2, 0.0), 2),
        "two_speaker_overlap_min": seconds_to_minutes(duration_by_count.get(2, 0.0)),
        "three_speaker_overlap_sec": round(duration_by_count.get(3, 0.0), 2),
        "three_speaker_overlap_min": seconds_to_minutes(duration_by_count.get(3, 0.0)),
    }

    four_plus_overlap = sum(
        duration for count, duration in duration_by_count.items() if count >= 4
    )
    summary["four_plus_speaker_overlap_sec"] = round(four_plus_overlap, 2)
    summary["four_plus_speaker_overlap_min"] = seconds_to_minutes(four_plus_overlap)

    return summary, pd.DataFrame(rows)


def calculate_overlap(
    words: Sequence[Dict],
    max_internal_pause_seconds: float,
) -> Tuple[Dict, pd.DataFrame]:
    _, _, conversation_length = conversation_bounds(words)
    speaker_regions = build_speaker_speech_regions(words, max_internal_pause_seconds)
    speaker_intervals = speaker_regions_to_intervals(speaker_regions)

    return calculate_interval_overlap(
        speaker_intervals,
        conversation_length,
        "assembly_word_regions",
    )


def segment_turns(words: Sequence[Dict], gap_seconds: float) -> List[Dict]:
    if not words:
        return []

    words_sorted = sorted(words, key=lambda word: (word["start"], word["end"]))
    turns = []
    current_speaker = words_sorted[0]["speaker_name"]
    current_words = [words_sorted[0]]
    current_start = words_sorted[0]["start"]
    current_end = words_sorted[0]["end"]

    for word in words_sorted[1:]:
        speaker = word["speaker_name"]
        gap = word["start"] - current_end

        if speaker == current_speaker and gap <= gap_seconds:
            current_words.append(word)
            current_end = max(current_end, word["end"])
            continue

        turns.append(build_turn(current_speaker, current_start, current_end, current_words))
        current_speaker = speaker
        current_words = [word]
        current_start = word["start"]
        current_end = word["end"]

    turns.append(build_turn(current_speaker, current_start, current_end, current_words))
    return turns


def build_turn(
    speaker: str,
    start: float,
    end: float,
    words: Sequence[Dict],
) -> Dict:
    return {
        "speaker": speaker,
        "start": start,
        "end": end,
        "duration": end - start,
        "word_count": len(words),
        "text": " ".join(word["word"] for word in words),
    }


def calculate_monologues(
    turns: Sequence[Dict],
    min_seconds: float = MONOLOGUE_MIN_SECONDS,
    source: str = "assembly_word_regions",
) -> Tuple[Dict, pd.DataFrame]:
    rows = []

    for turn in turns:
        if turn["duration"] <= min_seconds:
            continue

        rows.append(
            {
                "timing_source": source,
                "speaker": turn["speaker"],
                "start_sec": round(turn["start"], 2),
                "end_sec": round(turn["end"], 2),
                "start": format_seconds(turn["start"]),
                "end": format_seconds(turn["end"]),
                "duration_sec": round(turn["duration"], 2),
                "duration_min": seconds_to_minutes(turn["duration"]),
                "word_count": turn["word_count"],
                "preview": turn["text"][:300],
            }
        )

    return {"monologue_over_60s_count": len(rows)}, pd.DataFrame(rows)


def calculate_turn_behavior(turns: Sequence[Dict], duration_seconds: float) -> Dict:
    if not turns:
        return {
            "turn_count": 0,
            "speaker_switch_count": 0,
            "speaker_switches_per_min": 0.0,
            "backchannel_count": 0,
            "text_cutoff_hint_count": 0,
            "paralinguistic_event_count": 0,
            "filled_pause_count": 0,
            "disfluency_hint_count": 0,
            "outsized_silence_count": 0,
            "cadence_status": "NEEDS REVIEW",
            "max_turn_duration_sec": 0.0,
        }

    switches = 0
    backchannels = 0
    cutoff_hints = 0
    paralinguistic_events = 0
    filled_pauses = 0
    disfluency_hints = 0
    outsized_silences = 0
    max_duration = 0.0

    for index, turn in enumerate(turns):
        if index > 0 and turn["speaker"] != turns[index - 1]["speaker"]:
            switches += 1

        if index > 0:
            gap = safe_float(turn.get("start")) - safe_float(turns[index - 1].get("end"))
            if gap > OUTSIZED_SILENCE_SECONDS:
                outsized_silences += 1

        clean_words = [
            clean_word(word)
            for word in re.findall(r"\b[\w'-]+\b", turn.get("text", ""))
            if clean_word(word)
        ]

        turn_text = str(turn.get("text", ""))

        if (
            0 < len(clean_words) <= 3
            and all(word in BACKCHANNEL_WORDS for word in clean_words)
        ):
            backchannels += 1

        if re.search(r"[-/]\s*$|\.{2,}\s*$", turn.get("text", "").strip()):
            cutoff_hints += 1

        if re.search(r"\b(um+|uh+|hmm+|erm|er)\b", turn_text, flags=re.I):
            filled_pauses += 1

        if any(re.search(pattern, turn_text, flags=re.I) for pattern in PARALINGUISTIC_PATTERNS):
            paralinguistic_events += 1

        if any(re.search(pattern, turn_text, flags=re.I) for pattern in DISFLUENCY_PATTERNS):
            disfluency_hints += 1

        max_duration = max(max_duration, safe_float(turn.get("duration")))

    switches_per_min = switches / max(duration_seconds / 60.0, 1e-9)
    cadence_status = (
        "PASS"
        if switches_per_min >= 2.0
        and (backchannels + cutoff_hints + filled_pauses + disfluency_hints) > 0
        and outsized_silences == 0
        else "MANUAL REVIEW"
    )

    return {
        "turn_count": len(turns),
        "speaker_switch_count": switches,
        "speaker_switches_per_min": round(switches_per_min, 2),
        "backchannel_count": backchannels,
        "text_cutoff_hint_count": cutoff_hints,
        "paralinguistic_event_count": paralinguistic_events,
        "filled_pause_count": filled_pauses,
        "disfluency_hint_count": disfluency_hints,
        "outsized_silence_count": outsized_silences,
        "cadence_status": cadence_status,
        "max_turn_duration_sec": round(max_duration, 2),
    }


def calculate_transcript_qc(
    json_path: str,
    txt_path: Optional[str],
    turn_gap_seconds: float,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    speaker_name_map, riverside_turns, riverside_names = build_riverside_name_reference(txt_path)
    words = extract_words_from_assembly_json(json_path, speaker_name_map)

    if not words:
        raise ValueError("No valid word-level timestamps found in AssemblyAI JSON")

    text_mapping = infer_assembly_to_riverside_mapping_by_text(
        words,
        riverside_turns,
        speaker_name_map,
    )

    if text_mapping:
        apply_speaker_mapping_to_words(words, text_mapping)

    fallback_mapping = fill_generic_speaker_names_from_riverside_order(words, riverside_names)

    _, _, json_duration = conversation_bounds(words)

    density_df, density_summary = calculate_density(words, turn_gap_seconds)
    overlap_summary, overlap_df = calculate_overlap(words, turn_gap_seconds)

    speaker_regions = build_speaker_speech_regions(words, turn_gap_seconds)
    json_turns = sorted(
        [
            region
            for regions in speaker_regions.values()
            for region in regions
        ],
        key=lambda region: (region["start"], region["end"], region["speaker"]),
    )

    monologue_summary, monologues_df = calculate_monologues(
        json_turns,
        source="assembly_word_regions",
    )
    turn_behavior = calculate_turn_behavior(json_turns, json_duration)

    word_preview_df = pd.DataFrame(
        [
            {
                "speaker": word["speaker_name"],
                "raw_speaker_label": word["speaker"],
                "word": word["word"],
                "start_sec": round(word["start"], 3),
                "end_sec": round(word["end"], 3),
            }
            for word in words[:300]
        ]
    )

    summary = {
        "json_file": Path(json_path).name,
        "txt_file": Path(txt_path).name if txt_path else "",
        "word_count_from_json": len(words),
        "riverside_timed_turn_count": len(riverside_turns),
        "riverside_names_found": ", ".join(riverside_names),
        "speaker_name_mapping_status": speaker_mapping_status(words),
        "mapped_speaker_names": mapped_speaker_names(words),
        "speaker_name_mapping_fallback_used": "YES" if fallback_mapping else "NO",
        "turn_count_from_word_timestamps": len(json_turns),
        "primary_turn_source": "assembly_word_regions",
        **density_summary,
        "json_overlap_ratio_pct": overlap_summary.get("overlap_ratio_pct", 0.0),
        **overlap_summary,
        **monologue_summary,
        **turn_behavior,
    }

    return summary, density_df, overlap_df, monologues_df, word_preview_df


# -----------------------------
# Streamlit UI
# -----------------------------


def render_download_button(label: str, df: pd.DataFrame, filename: str) -> None:
    if df.empty:
        return

    export_df = frontend_df(df)

    st.download_button(
        label,
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def build_pairing_preview(pairs: Sequence[Dict]) -> pd.DataFrame:
    rows = []

    for pair in pairs:
        wav_paths = pair.get("wav_paths", [])
        role_counts = Counter(classify_wav(path) for path in wav_paths)

        rows.append(
            {
                "wav_count": len(wav_paths),
                "mixed_wav_count": role_counts.get("mixed", 0),
                "speaker_channel_wav_count": role_counts.get("speaker_channel", 0),
                "json_used": Path(pair["json_path"]).name if pair.get("json_path") else "",
                "txt_used": Path(pair["txt_path"]).name if pair.get("txt_path") else "",
                "wav_files": ", ".join(Path(path).name for path in wav_paths),
            }
        )

    return pd.DataFrame(rows)


def speaker_label_from_audio_file(path: str, speaker_name_map: Dict[str, str]) -> str:
    stem = Path(path).stem
    normalized_stem = normalize_key(stem)

    for mapped_key, name in speaker_name_map.items():
        if mapped_key and len(mapped_key) > 1 and mapped_key in normalized_stem:
            return name

    number_match = re.search(r"(?:speaker|spk|ch|channel|track|participant)[_\-\s]*(\d+)", stem, re.I)

    if number_match:
        number = number_match.group(1)
        for key in (number, str(max(0, int(number) - 1)), f"speaker_{number}", f"spk_{number}"):
            mapped = speaker_name_map.get(key) or speaker_name_map.get(normalize_key(key))
            if mapped:
                return mapped

    return stem


def calculate_audio_channel_overlap(
    audio_qc_rows: Sequence[Dict],
    speaker_name_map: Dict[str, str],
) -> Tuple[Dict, pd.DataFrame]:
    if not audio_qc_rows:
        return {}, pd.DataFrame()

    mixed_rows = [row for row in audio_qc_rows if row.get("audio_role") == "mixed"]
    channel_rows = [
        row
        for row in audio_qc_rows
        if row.get("audio_role") == "speaker_channel" and row.get("_speech_intervals_vad")
    ]

    if len(channel_rows) < 2:
        non_mixed_rows = [
            row
            for row in audio_qc_rows
            if row.get("audio_role") != "mixed" and row.get("_speech_intervals_vad")
        ]

        if len(non_mixed_rows) >= 2:
            channel_rows = non_mixed_rows

    if len(channel_rows) < 2:
        return {}, pd.DataFrame()

    if mixed_rows and is_finite_number(mixed_rows[0].get("audio_duration_sec")):
        denominator_seconds = float(mixed_rows[0]["audio_duration_sec"])
    else:
        denominator_seconds = max(
            safe_float(row.get("audio_duration_sec"))
            for row in channel_rows
        )

    speaker_intervals = {}

    for row in channel_rows:
        label = speaker_label_from_audio_file(row.get("_audio_path") or row["audio_file"], speaker_name_map)
        speaker_intervals[label] = row.get("_speech_intervals_vad") or []

    return calculate_interval_overlap(
        speaker_intervals,
        denominator_seconds,
        "separate_channel_silero_vad",
    )


def choose_final_overlap_summary(
    transcript_summary: Dict,
    audio_overlap_summary: Dict,
) -> Dict:
    audio_ratio = safe_float(audio_overlap_summary.get("overlap_ratio_pct"))

    if audio_ratio > 0:
        final_summary = dict(audio_overlap_summary)
        final_summary["json_overlap_ratio_pct"] = transcript_summary.get("json_overlap_ratio_pct", 0.0)
        final_summary["audio_channel_overlap_ratio_pct"] = audio_overlap_summary.get("overlap_ratio_pct", 0.0)
        return final_summary

    final_summary = {
        key: value
        for key, value in transcript_summary.items()
        if key.startswith("overlap_")
        or key in {
            "two_speaker_overlap_sec",
            "two_speaker_overlap_min",
            "three_speaker_overlap_sec",
            "three_speaker_overlap_min",
            "four_plus_speaker_overlap_sec",
            "four_plus_speaker_overlap_min",
        }
    }
    final_summary["json_overlap_ratio_pct"] = transcript_summary.get("json_overlap_ratio_pct", 0.0)
    final_summary["audio_channel_overlap_ratio_pct"] = audio_overlap_summary.get("overlap_ratio_pct", 0.0)
    return final_summary


def select_mixed_audio_row(audio_qc_rows: Sequence[Dict]) -> Optional[Dict]:
    mixed_rows = [row for row in audio_qc_rows if row.get("audio_role") == "mixed"]

    if mixed_rows:
        return max(mixed_rows, key=lambda row: safe_float(row.get("audio_duration_sec")))

    if len(audio_qc_rows) == 1:
        return audio_qc_rows[0]

    return None


def overlap_component_score(overlap_ratio: float) -> float:
    if overlap_ratio <= 0:
        return 3.0

    if 7.0 <= overlap_ratio <= 15.0:
        return 20.0

    if overlap_ratio < 7.0:
        return max(5.0, 8.0 + overlap_ratio * 1.5)

    return max(4.0, 20.0 - (overlap_ratio - 15.0) * 0.8)


def calculate_naturalness(
    transcript_summary: Dict,
    mixed_audio_row: Optional[Dict],
) -> Tuple[Dict, pd.DataFrame]:
    if not mixed_audio_row:
        return {
            "naturalness_score": np.nan,
            "naturalness_status": "NO MIXED AUDIO",
            "naturalness_threshold": NATURALNESS_PASS_SCORE,
            "naturalness_notes": "Needs mixed audio plus transcript metrics.",
        }, pd.DataFrame()

    density_score = 25.0 if transcript_summary.get("density_status") == "PASS" else 10.0

    monologues = int(safe_float(transcript_summary.get("monologue_over_60s_count")))
    monologue_score = max(0.0, 20.0 - monologues * 8.0)

    overlap_ratio = safe_float(transcript_summary.get("overlap_ratio_pct"))
    overlap_score = min(20.0, overlap_component_score(overlap_ratio))

    if mixed_audio_row.get("safe_pass_status") == "SAFE PASS":
        audio_score = 20.0
    elif mixed_audio_row.get("client_threshold_status") == "PASS":
        audio_score = 16.0
    elif mixed_audio_row.get("audio_qc_status") == "OK":
        audio_score = 8.0
    else:
        audio_score = 4.0

    switches_per_min = safe_float(transcript_summary.get("speaker_switches_per_min"))
    backchannels = int(safe_float(transcript_summary.get("backchannel_count")))
    cutoff_hints = int(safe_float(transcript_summary.get("text_cutoff_hint_count")))
    filled_pauses = int(safe_float(transcript_summary.get("filled_pause_count")))
    disfluencies = int(safe_float(transcript_summary.get("disfluency_hint_count")))
    paralinguistic = int(safe_float(transcript_summary.get("paralinguistic_event_count")))
    outsized_silences = int(safe_float(transcript_summary.get("outsized_silence_count")))
    cadence_status = transcript_summary.get("cadence_status", "MANUAL REVIEW")
    cadence_score = min(
        15.0,
        switches_per_min * 1.2
        + min(3.0, backchannels)
        + min(2.0, cutoff_hints)
        + min(3.0, filled_pauses + disfluencies)
        + min(2.0, paralinguistic),
    )
    if outsized_silences:
        cadence_score = max(0.0, cadence_score - min(6.0, outsized_silences * 2.0))

    score = round(density_score + monologue_score + overlap_score + audio_score + cadence_score, 1)

    if score >= NATURALNESS_PASS_SCORE:
        status = "PASS"
    elif score >= 60:
        status = "MANUAL REVIEW"
    else:
        status = "NEEDS REVIEW"

    component_rows = [
        {
            "component": "Density balance",
            "score": round(density_score, 1),
            "max_score": 25,
            "evidence": f"Speaker time balance is {transcript_summary.get('density_status', '')}.",
        },
        {
            "component": "Monologues",
            "score": round(monologue_score, 1),
            "max_score": 20,
            "evidence": f"{monologues} long solo speaking stretches above 60 seconds.",
        },
        {
            "component": "Overlap",
            "score": round(overlap_score, 1),
            "max_score": 20,
            "evidence": f'{overlap_ratio:.2f}% overlap from {transcript_summary.get("overlap_source", "unknown")}.',
        },
        {
            "component": "Mixed audio SNR/silence",
            "score": round(audio_score, 1),
            "max_score": 20,
            "evidence": f'Audio quality is {mixed_audio_row.get("client_threshold_status", "")}.',
        },
        {
            "component": "Cadence and spontaneity",
            "score": round(cadence_score, 1),
            "max_score": 15,
            "evidence": (
                f"{cadence_status}. {switches_per_min:.2f} speaker changes/min, {backchannels} backchannels, "
                f"{cutoff_hints} interruption/cutoff hints, {filled_pauses + disfluencies} filler/disfluency hints, "
                f"{paralinguistic} paralinguistic cues, {outsized_silences} long silences."
            ),
        },
    ]

    return {
        "naturalness_score": score,
        "naturalness_status": status,
        "naturalness_threshold": NATURALNESS_PASS_SCORE,
        "naturalness_notes": "How natural the mixed conversation is likely to feel when listened to. Higher is better. Target: balanced speakers, no long solo stretches, some natural overlap around 10%, and clean mixed audio.",
    }, pd.DataFrame(component_rows)


def process_pairs(
    pairs: Sequence[Dict],
    turn_gap_seconds: float,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    audio_rows = []
    transcript_rows = []
    density_frames = []
    overlap_frames = []
    monologue_frames = []
    word_preview_frames = []
    naturalness_frames = []
    error_rows = []

    progress = st.progress(0.0)
    status_text = st.empty()
    total_steps = max(1, sum(len(pair.get("wav_paths", [])) + 1 for pair in pairs))
    completed_steps = 0

    for pair in pairs:
        conversation_key = pair["conversation_key"]
        pair_audio_rows = []

        for wav_path in pair.get("wav_paths", []):
            status_text.write(f"Processing audio: {Path(wav_path).name}")
            audio_qc = calculate_audio_qc(wav_path)
            audio_row = {"conversation_key": conversation_key, **audio_qc}
            audio_rows.append(audio_row)
            pair_audio_rows.append(audio_row)

            if audio_qc["audio_qc_status"] != "OK":
                error_rows.append(
                    {
                        "conversation_key": conversation_key,
                        "file_name": Path(wav_path).name,
                        "stage": "audio_qc",
                        "error": audio_qc["audio_qc_error"],
                    }
                )

            completed_steps += 1
            progress.progress(min(1.0, completed_steps / total_steps))

        json_path = pair.get("json_path")
        txt_path = pair.get("txt_path")
        speaker_name_map, _, _ = build_riverside_name_reference(txt_path)
        audio_overlap_summary, audio_overlap_df = calculate_audio_channel_overlap(
            pair_audio_rows,
            speaker_name_map,
        )

        if not audio_overlap_df.empty:
            audio_overlap_df.insert(0, "conversation_key", conversation_key)
            overlap_frames.append(audio_overlap_df)

        status_text.write(
            f"Processing transcript: {Path(json_path).name if json_path else 'missing JSON'}"
        )

        if json_path:
            try:
                (
                    transcript_summary,
                    density_df,
                    overlap_df,
                    monologues_df,
                    word_preview_df,
                ) = calculate_transcript_qc(json_path, txt_path, turn_gap_seconds)

                final_overlap_summary = choose_final_overlap_summary(
                    transcript_summary,
                    audio_overlap_summary,
                )
                transcript_summary.update(final_overlap_summary)

                naturalness_summary, naturalness_df = calculate_naturalness(
                    transcript_summary,
                    select_mixed_audio_row(pair_audio_rows),
                )
                transcript_summary.update(naturalness_summary)

                transcript_rows.append(
                    {"conversation_key": conversation_key, **transcript_summary}
                )

                if not density_df.empty:
                    density_df.insert(0, "conversation_key", conversation_key)
                    density_frames.append(density_df)

                if not overlap_df.empty:
                    overlap_df.insert(0, "conversation_key", conversation_key)
                    overlap_frames.append(overlap_df)

                if not monologues_df.empty:
                    monologues_df.insert(0, "conversation_key", conversation_key)
                    monologue_frames.append(monologues_df)

                if not word_preview_df.empty:
                    word_preview_df.insert(0, "conversation_key", conversation_key)
                    word_preview_frames.append(word_preview_df)

                if not naturalness_df.empty:
                    naturalness_df.insert(0, "conversation_key", conversation_key)
                    naturalness_frames.append(naturalness_df)

            except Exception as exc:
                error_rows.append(
                    {
                        "conversation_key": conversation_key,
                        "file_name": Path(json_path).name,
                        "stage": "transcript_qc",
                        "error": str(exc),
                    }
                )
        else:
            error_rows.append(
                {
                    "conversation_key": conversation_key,
                    "file_name": "",
                    "stage": "transcript_qc",
                    "error": "AssemblyAI JSON missing. Density, overlap, and monologue checks require word-level JSON.",
                }
            )

        completed_steps += 1
        progress.progress(min(1.0, completed_steps / total_steps))

    status_text.empty()
    progress.empty()

    audio_df = pd.DataFrame(audio_rows)
    transcript_df = pd.DataFrame(transcript_rows)
    density_df = pd.concat(density_frames, ignore_index=True) if density_frames else pd.DataFrame()
    overlap_df = pd.concat(overlap_frames, ignore_index=True) if overlap_frames else pd.DataFrame()
    monologues_df = (
        pd.concat(monologue_frames, ignore_index=True) if monologue_frames else pd.DataFrame()
    )
    word_preview_df = (
        pd.concat(word_preview_frames, ignore_index=True)
        if word_preview_frames
        else pd.DataFrame()
    )
    naturalness_df = (
        pd.concat(naturalness_frames, ignore_index=True)
        if naturalness_frames
        else pd.DataFrame()
    )
    errors_df = pd.DataFrame(error_rows)

    return (
        audio_df,
        transcript_df,
        density_df,
        overlap_df,
        monologues_df,
        word_preview_df,
        naturalness_df,
        errors_df,
    )


def build_summary_df(audio_df: pd.DataFrame, transcript_df: pd.DataFrame) -> pd.DataFrame:
    if audio_df.empty:
        audio_summary = pd.DataFrame()
    else:
        audio_summary = (
            audio_df.groupby("conversation_key", dropna=False)
            .agg(
                audio_files_processed=("audio_file", "count"),
                audio_files_ok=("audio_qc_status", lambda series: int((series == "OK").sum())),
                audio_client_pass=(
                    "client_threshold_status",
                    lambda series: int((series == "PASS").sum()),
                ),
                audio_safe_pass=(
                    "safe_pass_status",
                    lambda series: int((series == "SAFE PASS").sum()),
                ),
                min_snr_vad_db=("snr_vad_db", "min"),
                max_silence_floor_dbfs=("silence_floor_dbfs", "max"),
            )
            .reset_index()
        )

    if transcript_df.empty and audio_summary.empty:
        return pd.DataFrame()

    if transcript_df.empty:
        return audio_summary

    if audio_summary.empty:
        return transcript_df

    return pd.merge(transcript_df, audio_summary, on="conversation_key", how="outer")


def render_results(
    audio_df: pd.DataFrame,
    transcript_df: pd.DataFrame,
    density_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    monologues_df: pd.DataFrame,
    word_preview_df: pd.DataFrame,
    naturalness_df: pd.DataFrame,
    errors_df: pd.DataFrame,
) -> None:
    summary_df = build_summary_df(audio_df, transcript_df)

    if summary_df.empty and audio_df.empty and errors_df.empty:
        st.warning("No results were generated.")
        return

    (
        tab_summary,
        tab_audio,
        tab_density,
        tab_overlap,
        tab_cadence,
        tab_monologues,
        tab_naturalness,
    ) = st.tabs(
        [
            "Summary",
            "Audio QC",
            "Density",
            "Overlap",
            "Cadence",
            "Monologues",
            "Natural Feel",
        ]
    )

    with tab_summary:
        if not summary_df.empty:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Sessions", len(summary_df))

            audio_client_pass = (
                int(summary_df.get("audio_client_pass", pd.Series(dtype=float)).fillna(0).sum())
                if "audio_client_pass" in summary_df
                else 0
            )
            c2.metric("Audio Client Pass", audio_client_pass)

            avg_overlap = (
                round(summary_df["overlap_ratio_pct"].fillna(0).mean(), 2)
                if "overlap_ratio_pct" in summary_df
                else 0.0
            )
            c3.metric("Avg Overlap Ratio %", avg_overlap)

            total_monologues = (
                int(summary_df["monologue_over_60s_count"].fillna(0).sum())
                if "monologue_over_60s_count" in summary_df
                else 0
            )
            c4.metric("Monologues > 60s", total_monologues)

            naturalness_pass = (
                int((summary_df["naturalness_status"] == "PASS").sum())
                if "naturalness_status" in summary_df
                else 0
            )
            c5.metric("Natural Feel Pass", naturalness_pass)

            st.dataframe(frontend_df(summary_df), width="stretch")
            render_download_button(
                "Download Summary CSV",
                summary_df,
                "conversation_qc_summary.csv",
            )
        else:
            st.info("No summary rows available.")

    with tab_audio:
        st.markdown(
            """
**Manual Review Guide**

- If `SNR` is below `20 dB`, check whether speech sounds too quiet, muffled, clipped, or buried under noise.
- If `silence_floor` is above `-40 dBFS`, listen to quiet parts for fan noise, room hum, hiss, keyboard noise, or another speaker leaking into the channel.
- If confidence is `LOW`, there was not enough clear speech or silence for a strong measurement, so spot-check the audio by ear.
"""
        )

        if not audio_df.empty:
            st.dataframe(frontend_df(audio_df), width="stretch")
            render_download_button("Download Audio QC CSV", audio_df, "audio_qc_all_wavs.csv")
        else:
            st.info("No WAV files found.")

    with tab_density:
        if not density_df.empty:
            st.dataframe(
                frontend_df(density_df),
                width="stretch",
                column_config={
                    "speaker": st.column_config.TextColumn("Speaker"),
                    "speaking_time": st.column_config.TextColumn("Speaking Time (mm:ss)"),
                    "speaking_pct_of_duration": st.column_config.NumberColumn(
                        "Speaking % of Duration",
                        format="%.2f%%",
                    ),
                    "expected_pct": st.column_config.NumberColumn("Expected %", format="%.2f%%"),
                    "min_allowed_pct": st.column_config.NumberColumn("Min Allowed %", format="%.2f%%"),
                    "max_allowed_pct": st.column_config.NumberColumn("Max Allowed %", format="%.2f%%"),
                    "density_status": st.column_config.TextColumn("Status"),
                },
            )
            render_download_button(
                "Download Density CSV",
                density_df,
                "speaker_density_word_level.csv",
            )
        else:
            st.info("No density details available.")

    with tab_overlap:
        if not transcript_df.empty:
            overlap_summary_cols = [
                "overlap_source",
                "overlap_ratio_denominator",
                "non_silence_duration_sec",
                "json_overlap_ratio_pct",
                "audio_channel_overlap_ratio_pct",
                "overlap_segment_count",
                "overlap_duration_sec",
                "overlap_duration_min",
                "overlap_ratio_pct",
                "two_speaker_overlap_sec",
                "two_speaker_overlap_min",
                "three_speaker_overlap_sec",
                "three_speaker_overlap_min",
                "four_plus_speaker_overlap_sec",
                "four_plus_speaker_overlap_min",
            ]
            existing_cols = [col for col in overlap_summary_cols if col in transcript_df.columns]
            st.dataframe(transcript_df[existing_cols], width="stretch")

        if not overlap_df.empty:
            st.dataframe(frontend_df(overlap_df), width="stretch")
            render_download_button(
                "Download Overlap Details CSV",
                overlap_df,
                "overlap_segments_word_level.csv",
            )
        else:
            st.info("No overlapping word-level speaker intervals detected.")

    with tab_cadence:
        st.markdown(
            """
**Cadence Check**

Good conversational data should sound spontaneous: people naturally take turns, sometimes overlap, use short responses like “yeah” or “right,” include fillers such as “um” or “uh,” and should not contain long empty pauses.
"""
        )
        cadence_cols = [
            "cadence_status",
            "speaker_switches_per_min",
            "speaker_switch_count",
            "backchannel_count",
            "text_cutoff_hint_count",
            "filled_pause_count",
            "disfluency_hint_count",
            "paralinguistic_event_count",
            "outsized_silence_count",
            "overlap_ratio_pct",
            "overlap_source",
            "non_silence_duration_sec",
        ]

        if not transcript_df.empty:
            existing_cols = [col for col in cadence_cols if col in transcript_df.columns]
            st.dataframe(frontend_df(transcript_df[existing_cols]), width="stretch")
        else:
            st.info("Cadence details are shown after transcript processing.")

    with tab_monologues:
        if not monologues_df.empty:
            st.dataframe(frontend_df(monologues_df), width="stretch")
            render_download_button(
                "Download Monologues CSV",
                monologues_df,
                "monologues_over_60s_word_level.csv",
            )
        else:
            st.info("No monologues over 60 seconds detected.")

    with tab_naturalness:
        st.info(
            "This score estimates how natural the mixed conversation would feel after listening: balanced speaker time, normal back-and-forth, no long monologues, some natural overlap, and clean audio."
        )

        naturalness_summary_cols = [
            "naturalness_score",
            "naturalness_status",
            "naturalness_threshold",
            "naturalness_notes",
            "density_status",
            "overlap_ratio_pct",
            "overlap_source",
            "monologue_over_60s_count",
            "cadence_status",
            "speaker_switches_per_min",
            "backchannel_count",
            "filled_pause_count",
            "disfluency_hint_count",
            "outsized_silence_count",
        ]

        if not transcript_df.empty:
            existing_cols = [col for col in naturalness_summary_cols if col in transcript_df.columns]
            st.dataframe(frontend_df(transcript_df[existing_cols]), width="stretch")

        if not naturalness_df.empty:
            st.dataframe(frontend_df(naturalness_df), width="stretch")
            render_download_button(
                "Download Naturalness CSV",
                naturalness_df,
                "conversation_naturalness.csv",
            )
        else:
            st.info("Naturalness is shown when a mixed audio file and transcript metrics are available.")

    if not errors_df.empty:
        with st.expander("Processing messages", expanded=True):
            st.dataframe(frontend_df(errors_df), width="stretch")
            render_download_button("Download Errors CSV", errors_df, "qc_errors.csv")


def main() -> None:
    st.set_page_config(page_title="Audio QC Tool", layout="wide")
    apply_app_styles()

    st.title("Audio QC Tool")
    st.caption("Audio quality, speaker balance, overlap, cadence, and natural conversation feel.")

    turn_gap_seconds = DEFAULT_TURN_GAP_SECONDS

    st.markdown(
        "**Fixed segmentation rule:** same-speaker speech is split when an internal pause is greater than `0.2s`."
    )

    all_paths: List[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        drive_link = st.text_input("Google Drive folder or file link")
        run_clicked = st.button("Run QC", type="primary", width="stretch")

        if run_clicked:
            if not drive_link.strip():
                st.error("Paste a Google Drive link.")
                return

            try:
                drive_status = st.empty()
                drive_progress = st.progress(0.0)

                def drive_status_callback(message: str) -> None:
                    drive_status.info(message)

                def drive_progress_callback(
                    file_index: int,
                    total_files: int,
                    file_name: str,
                    bytes_downloaded: int,
                    total_size: Optional[int],
                ) -> None:
                    file_fraction = 0.0

                    if total_size:
                        file_fraction = min(1.0, bytes_downloaded / total_size)

                    overall_fraction = (
                        (file_index - 1) + file_fraction
                    ) / max(1, total_files)

                    drive_progress.progress(min(1.0, overall_fraction))

                    if total_size:
                        mb_done = bytes_downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        drive_status.info(
                            f"Downloading {file_index}/{total_files}: {file_name} "
                            f"({mb_done:.1f}/{mb_total:.1f} MB)"
                        )

                all_paths = download_from_drive(
                    drive_link.strip(),
                    tmpdir,
                    status_callback=drive_status_callback,
                    progress_callback=drive_progress_callback,
                )
                drive_progress.empty()
                drive_status.empty()
            except Exception as exc:
                st.error(f"Drive download failed: {exc}")
                return

        if not run_clicked:
            return

        if not all_paths:
            st.error("No supported files found. Supported extensions: WAV, JSON, TXT.")
            return

        pairs = pair_files(all_paths)

        st.subheader("Detected Inputs")
        pairing_preview = build_pairing_preview(pairs)
        st.dataframe(pairing_preview, width="stretch")

        (
            audio_df,
            transcript_df,
            density_df,
            overlap_df,
            monologues_df,
            word_preview_df,
            naturalness_df,
            errors_df,
        ) = process_pairs(pairs, turn_gap_seconds=turn_gap_seconds)

        render_results(
            audio_df,
            transcript_df,
            density_df,
            overlap_df,
            monologues_df,
            word_preview_df,
            naturalness_df,
            errors_df,
        )


if __name__ == "__main__":
    main()
