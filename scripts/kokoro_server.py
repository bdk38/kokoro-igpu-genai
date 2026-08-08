#!/usr/bin/env python3
"""
kokoro_server.py — OpenAI-compatible TTS server for Kokoro on bdk-server.

OpenAI-compatible Kokoro TTS server. Product default (v2.0.0 genai appliance): **ovgenai-gpu**; fallback: ovgenai-cpu (official Kokoro-82M int8 GenAI on Intel iGPU) per Nexus cutover after I0-GO-default-candidate. Fallback: KOKORO_BACKEND=ort-cpu (v0.19 ONNX). Legacy: ov-gpu patched ONNX (I0.5). Novel shapes still pay first-infer tax; prefer KOKORO_TTS_CACHE=1 and chunk-shaped KOKORO_WARM_TEXT in deploy.

Endpoints:
  POST /v1/audio/speech     OpenAI-compatible TTS (input, voice, speed,
                            response_format)
  GET  /v1/audio/voices     list available voices
  GET  /v1/models           minimal OpenAI-style model list
  GET  /health              backend + model status

Configuration (env vars):
  KOKORO_MODEL      path to ONNX model
                    default: <repo>/models/kokoro-v0_19.onnx
                    (use the patched gpu4d.stft model for OV backends)
                    For ovgenai-*: if this points at a directory containing
                    openvino_model.xml, it is also accepted as the GenAI pack.
  KOKORO_GENAI_MODEL  directory of official OpenVINO GenAI Kokoro pack
                    (openvino_model.xml + voices/*.bin). Used by
                    ovgenai-gpu / ovgenai-cpu.
                    default: <repo>/models/kokoro-82M-int8-ov
                    Note: official pack is v1.0-family int8 — a different
                    checkpoint from the ship v0.19 ONNX (not a silent swap).
  KOKORO_VOICES     path to voices NPZ (ort/ov token-id backends)
                    default: <repo>/models/voices-v1.0.bin
  KOKORO_BACKEND    ort-cpu | ov-cpu | ov-gpu | ovgenai-gpu | ovgenai-cpu
                    (default ort-cpu; ovgenai-gpu Prototype; ov-gpu LEGACY proof)
                    (default: ovgenai-gpu; ovgenai-cpu without GPU)
  KOKORO_GPU_PRECISION  f32 | f16                 (default: f32; f16 is
                        broken upstream — MatMul compile bug; ov-gpu only)
  KOKORO_CACHE      OpenVINO compile-cache dir (unchanged semantics)
                    default: <repo>/cache/openvino
  KOKORO_TTS_CACHE  0|1 — opt-in TTS disk cache (default: 0)
  KOKORO_TTS_CACHE_DIR  TTS response/chunk store root
                    default: <repo>/cache/tts
  KOKORO_TTS_CACHE_MAX_MB  lazy LRU-by-mtime size cap (default: 500)
  KOKORO_TTS_CACHE_TIER  response | chunk | both (default: both)
                    When KOKORO_TTS_CACHE=1: "response" = C1 full-request
                    only; "chunk" = C2 per-chunk only; "both" =
                    C1 then C2 on miss. Schema v3: per-chunk pcm roundtrip;
                    C2 text_unit is "c2ids:" + token-ids (ort/ov) or
                    "c2txt:" + exact chunk string (ovgenai). backend_id
                    firewalls cross-backend keys.
                    X-Kokoro-Cache may be hit | partial | miss.
  KOKORO_WARM_BUCKETS   comma list of token buckets to pre-warm at startup
                    on OV backends, e.g. "96,192". Warm state is SHAPE-keyed
                    (output sample count / internal duration lattice), NOT
                    bucket-wide and NOT content-transferring (notes/19–20).
                    Near-capacity REAL text is synthesized per bucket so the
                    pre-warm path matches production; all-zero pads do NOT
                    warm (notes/18). This only retires cold cost for that
                    pre-warm shape — varied Read Aloud still pays cold on
                    novel shapes. Steady ~0.9 RTF is for REPEATS of a warmed
                    shape. KOKORO_CACHE shortens OV compile, NOT shape warm.
                    (Skipped for ovgenai-* — no _bucket attribute.)
  KOKORO_WARM_TEXT  optional; '|' -separated exact phrases to synthesize at
                    startup (after bucket pre-warm). Use to pin demo/canned
                    sentences so the first user hit of that exact text is warm
                    (~0.9 RTF on ov-gpu once shape is hot). Example:
                    KOKORO_WARM_TEXT='The quick brown fox jumps over the lazy dog.'
  KOKORO_DEFAULT_VOICE  (default: af_bella)

Run:
  cd <repo> && source venv/bin/activate
  pip install fastapi uvicorn
  python scripts/kokoro_server.py --host 0.0.0.0 --port 8880

Open WebUI wiring (Admin -> Settings -> Audio):
  TTS Engine: OpenAI
  API Base URL: http://<bdk-server>:8880/v1
  API Key: anything (not checked)
  TTS Voice: af_bella (or any /v1/audio/voices entry, or OpenAI aliases)
  Response format: wav (mp3 works if ffmpeg is installed on the host)
  Response Splitting: depends on backend (see notes/15):
    ort-cpu (RTF ~0.4)  -> Punctuation is fine and gives fast first-audio
    ovgenai-gpu (steady RTF ~0.7) -> Paragraphs OK; first novel shape
      still multi-second — enable KOKORO_TTS_CACHE=1 for repeats.
    ort-cpu (RTF ~0.4) -> Punctuation fine.
    legacy ov-gpu (RTF 4-6) -> None/Paragraphs only (notes/15).

Notes:
  - Long input is chunked at sentence boundaries to stay under the model's
    510-token limit; chunks are stitched with a short gap.
  - OV backends compile per token-length; the server pads chunk tokens to
    bucket sizes (96/192/288/384/512) and caches compiled models per
    bucket, so steady-state requests reuse compiles. ORT is fully dynamic.
  - The NSF vocoder renders trailing pad tokens as a short breath-like
    burst after a quiet gap at chunk ends. OV-path chunks are therefore
    trimmed after inference: audio is segmented into gap-separated speech
    groups, and trailing groups are stripped only if they look like pad
    energy (gap-separated AND weak AND short). Natural mid-sentence
    pauses never trigger a cut because the speech that follows them fails
    the weak/short tests. Always on; fails safe (keeps audio when in
    doubt). ORT path never pads, so it is never trimmed.
    Set KOKORO_TRIM_DEBUG=1 to log per-chunk trim decisions.
  - mp3/opus/flac need ffmpeg on PATH; otherwise the server falls back to
    wav and says so in the X-Kokoro-Format header.
"""

import argparse
from pathlib import Path as _Path

# Repo root = parent of scripts/ — portable defaults (R0: no hardcoded host paths)
REPO_ROOT = str(_Path(__file__).resolve().parent.parent)
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import wave

import numpy as np

# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------

SR = 24000
MAX_TOKENS = 510
PAD_BUCKETS = [96, 192, 288, 384, 512]
CHUNK_GAP_S = 0.12

# pad-tail trim (OV bucket-padded chunks only)
TRIM_FRAME_S = 0.02        # RMS analysis frame (20 ms)
TRIM_MARGIN_S = 0.03       # keep this much after the last speech frame
TRIM_FADE_S = 0.01         # fade length at the cut
TRIM_SEARCH_FACTOR = 1.5   # search window = pad fraction x this
TRIM_RMS_RATIO = 0.15      # frame counts as speech at >= 15% of ref RMS
TRIM_QUIET_S = 0.10        # gap length separating speech groups
TRIM_MOAN_MAX_S = 0.6      # pad burst must be shorter than this
                           # (measured ov-gpu bursts: 0.22-0.40 s)
TRIM_MOAN_RMS_RATIO = 0.9  # pad burst peak must stay below this x ref RMS
                           # (measured ov-gpu bursts: 0.64-0.83x ref;
                           # measured real continuation speech: ~1.39x)
TRIM_PAD_GAP_S = 0.15      # burst must be detached: gap before it >= this
                           # (measured: intra-word stop closure ~0.10 s,
                           # pre-moan pad gaps 0.46-0.48 s)
TRIM_REF_FLOOR = 1e-3      # refuse to trim when speech reference RMS is
                           # this low (silence-referenced clip)
TRIM_DEBUG = os.environ.get("KOKORO_TRIM_DEBUG", "0") == "1"


MODEL_PATH = os.environ.get(
    "KOKORO_MODEL", os.path.join(REPO_ROOT, "models", "kokoro-v0_19.onnx"))
GENAI_MODEL_DEFAULT = os.path.join(REPO_ROOT, "models", "kokoro-82M-int8-ov")
VOICES_PATH = os.environ.get(
    "KOKORO_VOICES", os.path.join(REPO_ROOT, "models", "voices-v1.0.bin"))
BACKEND = os.environ.get("KOKORO_BACKEND", "ovgenai-gpu")
GPU_PRECISION = os.environ.get("KOKORO_GPU_PRECISION", "f32")
CACHE_DIR = os.environ.get(
    "KOKORO_CACHE", os.path.join(REPO_ROOT, "cache", "openvino"))
# TTS response/chunk cache (C1/C2). Distinct from KOKORO_CACHE (OV compile).
TTS_CACHE_ON = os.environ.get("KOKORO_TTS_CACHE", "0") == "1"
TTS_CACHE_DIR = os.environ.get(
    "KOKORO_TTS_CACHE_DIR", os.path.join(REPO_ROOT, "cache", "tts"))
TTS_CACHE_MAX_MB = float(os.environ.get("KOKORO_TTS_CACHE_MAX_MB", "500"))
TTS_CACHE_TIER = os.environ.get("KOKORO_TTS_CACHE_TIER", "both").strip().lower()
# C1 full-request; C2 per-chunk (shared store when either on).
TTS_RESPONSE_CACHE = TTS_CACHE_ON and TTS_CACHE_TIER in ("response", "both")
TTS_CHUNK_CACHE = TTS_CACHE_ON and TTS_CACHE_TIER in ("chunk", "both")
# schema_ver 3: per-chunk int16 pcm roundtrip (from v2) + c2txt: keys for
# ovgenai backends (exact post-chunker text strings). Bump invalidates prior
# v2 disk entries honestly rather than mixing layouts (notes/41, I0 G2).
TTS_CACHE_SCHEMA_VER = 3
TTS_SAMPLE_FMT = "24000:s16le:mono"
# Soft char budget for GenAI text chunks (phoneme budget may tighten further).
GENAI_CHUNK_SOFT_CHARS = 400
DEFAULT_VOICE = os.environ.get("KOKORO_DEFAULT_VOICE", "af_bella")
WARM_BUCKETS = [int(x) for x in
                os.environ.get("KOKORO_WARM_BUCKETS", "").split(",")
                if x.strip().isdigit()]
# Exact phrases to pin (shape-keyed warm). Split on | so commas in text are OK.
WARM_TEXTS = [t.strip() for t in
              os.environ.get("KOKORO_WARM_TEXT", "").split("|")
              if t.strip()]

# OpenAI voice aliases -> kokoro voices
OPENAI_VOICE_MAP = {
    "alloy": "af_alloy", "echo": "am_echo", "fable": "bf_emma",
    "onyx": "am_onyx", "nova": "af_nova", "shimmer": "af_shimmer",
}

# ----------------------------------------------------------------------
# tokenizer (kokoro v0.19) — same as tts_harness.py
# ----------------------------------------------------------------------

def _build_vocab():
    _pad = "$"
    _punctuation = ';:,.!?\u00a1\u00bf\u2014\u2026"\u00ab\u00bb\u201c\u201d '
    _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    _letters_ipa = (
        "\u0251\u0250\u0252\u00e6\u0253\u0299\u03b2\u0254\u0255\u00e7\u0257"
        "\u0256\u00f0\u02a4\u0259\u0258\u025a\u025b\u025c\u025d\u025e\u025f"
        "\u0284\u0261\u0260\u0262\u029b\u0266\u0267\u0127\u0265\u029c\u0268"
        "\u026a\u029d\u026d\u026c\u026b\u026e\u029f\u0271\u026f\u0270\u014b"
        "\u0273\u0272\u0274\u00f8\u0275\u0278\u03b8\u0153\u0276\u0298\u0279"
        "\u027a\u027e\u027b\u0280\u0281\u027d\u0282\u0283\u0288\u02a7\u0289"
        "\u028a\u028b\u2c71\u028c\u0263\u0264\u028d\u03c7\u028e\u028f\u0291"
        "\u0290\u0292\u0294\u02a1\u0295\u02a2\u01c0\u01c1\u01c2\u01c3\u02c8"
        "\u02cc\u02d0\u02d1\u02bc\u02b4\u02b0\u02b1\u02b2\u02b7\u02e0\u02e4"
        "\u02de\u2193\u2191\u2192\u2197\u2198'\u0329'\u1d7b"
    )
    symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
    return {s: i for i, s in enumerate(symbols)}


VOCAB = _build_vocab()
_ESPEAK_BACKENDS = {}


def _espeak(lang):
    if lang not in _ESPEAK_BACKENDS:
        from phonemizer.backend import EspeakBackend
        _ESPEAK_BACKENDS[lang] = EspeakBackend(
            language=lang, preserve_punctuation=True, with_stress=True)
    return _ESPEAK_BACKENDS[lang]


def _cleanup_phonemes(ps, lang):
    ps = ps.replace("k\u0259k\u02c8o\u02d0\u0279o\u028a",
                    "k\u02c8o\u028ak\u0259\u0279o\u028a")
    ps = ps.replace("k\u0259k\u02c8\u0254\u02d0\u0279\u0259\u028a",
                    "k\u02c8\u0259\u028ak\u0259\u0279\u0259\u028a")
    ps = ps.replace("\u02b2", "j").replace("r", "\u0279")
    ps = ps.replace("x", "k").replace("\u026c", "l")
    ps = re.sub(r"(?<=[a-z\u0279\u02d0])(?=h\u02c8\u028cnd\u0279\u026ad)", " ", ps)
    ps = re.sub(r" z(?=[;:,.!?\u00a1\u00bf\u2014\u2026\"\u00ab\u00bb\u201c\u201d ]|$)",
                "z", ps)
    if lang == "en-us":
        ps = re.sub(r"(?<=n\u02c8a\u026an)ti(?!\u02d0)", "di", ps)
    return ps


def phonemes_to_ids(text, lang):
    ps = _espeak(lang).phonemize([text])
    ps = ps[0].strip() if ps else ""
    ps = _cleanup_phonemes(ps, lang)
    return [VOCAB[c] for c in ps if c in VOCAB]


_SENT_SPLIT = re.compile(r"(?<=[.!?\u2026])\s+")


def chunk_text(text, lang):
    """sentences -> token-id chunks, each under MAX_TOKENS."""
    sentences = [s for s in _SENT_SPLIT.split(text.strip()) if s]
    chunks, current = [], []
    for sent in sentences:
        ids = phonemes_to_ids(sent, lang)
        while len(ids) > MAX_TOKENS - 2:          # pathological run-on
            chunks.append(ids[:MAX_TOKENS - 2])
            ids = ids[MAX_TOKENS - 2:]
        if len(current) + len(ids) + 1 > MAX_TOKENS - 2:
            if current:
                chunks.append(current)
            current = list(ids)
        else:
            current = current + ([VOCAB[" "]] if current else []) + ids
    if current:
        chunks.append(current)
    return chunks


def chunk_text_strings(text, lang):
    """Sentence-merge text chunks for GenAI path.

    Returns list[str] of exact chunk strings used for generate() AND for
    C2 cache keys (``c2txt:`` + chunk). Merges short sentences; keeps each
    chunk under a soft char budget and under MAX_TOKENS phoneme ids when
    the espeak path is cheap enough to check.
    """
    sentences = [s for s in _SENT_SPLIT.split(text.strip()) if s]
    if not sentences:
        return []

    def _over_budget(s):
        if len(s) > GENAI_CHUNK_SOFT_CHARS:
            return True
        try:
            return len(phonemes_to_ids(s, lang)) + 2 >= MAX_TOKENS
        except Exception:
            return len(s) > GENAI_CHUNK_SOFT_CHARS

    def _split_long(s):
        """Hard-split a pathological run-on by soft char budget at spaces."""
        out = []
        while len(s) > GENAI_CHUNK_SOFT_CHARS:
            cut = s.rfind(" ", 0, GENAI_CHUNK_SOFT_CHARS)
            if cut < GENAI_CHUNK_SOFT_CHARS // 2:
                cut = GENAI_CHUNK_SOFT_CHARS
            piece = s[:cut].strip()
            if piece:
                out.append(piece)
            s = s[cut:].strip()
        if s:
            out.append(s)
        return out

    chunks, current = [], ""
    for sent in sentences:
        if _over_budget(sent) and not current:
            # Single sentence already too large — emit hard-split pieces.
            pieces = _split_long(sent)
            # Re-merge trailing short tail into next if needed: emit all
            # but last into chunks; last becomes current if under budget.
            for p in pieces[:-1]:
                chunks.append(p)
            if pieces:
                current = pieces[-1]
            continue
        candidate = (current + " " + sent).strip() if current else sent
        if current and _over_budget(candidate):
            chunks.append(current)
            if _over_budget(sent):
                pieces = _split_long(sent)
                chunks.extend(pieces[:-1])
                current = pieces[-1] if pieces else ""
            else:
                current = sent
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def resolve_genai_model_dir():
    """Directory of official GenAI Kokoro pack (xml + voices/*.bin)."""
    explicit = os.environ.get("KOKORO_GENAI_MODEL")
    if explicit:
        return explicit
    # Allow KOKORO_MODEL to point at a pack directory with openvino_model.xml.
    if (os.path.isdir(MODEL_PATH)
            and os.path.isfile(os.path.join(MODEL_PATH, "openvino_model.xml"))):
        return MODEL_PATH
    return GENAI_MODEL_DEFAULT


# ----------------------------------------------------------------------
# pad-tail trim
# ----------------------------------------------------------------------

def trim_pad_tail(audio, n_real, n_bucket, sr=SR):
    """Remove the voiced tail the vocoder synthesizes from bucket pad tokens.

    Measured failure shape (see 10-status-trim-and-skips.md):

        real speech -> short quiet gap -> weak voiced burst (moan) -> silence

    v1.1.1 cut at the *first* sustained quiet and fired on comma pauses.
    v1.1.2/3 segmented into speech groups but a soft word-final syllable
    after a stop-closure gap ("pass^port") matched the weak+short profile
    of a moan, and a leading-silence head broke the reference RMS.

    A trailing group is stripped only when it passes ALL of:

      1. weak:     peak RMS < TRIM_MOAN_RMS_RATIO x speech reference
                   (measured moans 0.23-0.77x; real speech >= 1.05x)
      2. short:    duration < TRIM_MOAN_MAX_S (moans 0.12-0.40 s)
      3. detached: gap before it >= TRIM_PAD_GAP_S (a stop-closure gap
                   ~0.10 s keeps word-final syllables attached; pad gaps
                   measure 0.40-0.78 s)
      4. in the pad search window (head is sacred)

    A terminal-silence gate existed in v1.1.4 and was removed: across the
    full probe set every group it kept was an ear-confirmed pad moan and
    it never protected real speech (Kokoro renders final words attached
    to preceding speech, not detached). Tail length is still logged as
    data in case a weak+short+detached real word ever appears.

    The speech reference is the p90 of frames at >= 10% of the clip's max
    frame RMS (not a positional head, which can be leading silence), with
    a hard floor below which trimming is refused. If no group qualifies,
    only trailing silence after the last group is trimmed. No confident
    structure -> audio returned unchanged.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if n_bucket <= n_real or audio.size == 0:
        return audio
    pad_frac = (n_bucket - n_real) / float(n_bucket)
    search = int(audio.size * min(1.0, pad_frac * TRIM_SEARCH_FACTOR))
    frame = max(1, int(sr * TRIM_FRAME_S))
    # always keep a head region as the speech-RMS reference, even when the
    # pad fraction is large (short sentence in a big bucket)
    ref_min = min(audio.size // 2, max(frame * 5, int(sr * 0.2)))
    keep_min = max(audio.size - search, ref_min)  # never cut before this
    if audio.size - keep_min < frame or keep_min < frame:
        return audio

    n_fr = audio.size // frame
    if n_fr < 3:
        return audio
    rms = np.sqrt(np.mean(
        audio[:n_fr * frame].reshape(-1, frame).astype(np.float64) ** 2,
        axis=1))

    # speech reference from the loudest material anywhere in the clip;
    # a positional head window can be leading silence (whisper probe:
    # ref collapsed to 2e-4 and every ratio went nonsensical)
    loud = rms[rms >= 0.1 * float(rms.max())]
    ref = float(np.percentile(loud, 90)) if loud.size else 0.0
    if ref < TRIM_REF_FLOOR:
        return audio
    thresh = ref * TRIM_RMS_RATIO

    speech_idx = np.nonzero(rms >= thresh)[0]
    if speech_idx.size == 0:
        return audio

    # merge speech frames into groups; gaps shorter than TRIM_QUIET_S do
    # not separate groups (they are intra-speech texture, not structure)
    need = max(2, int(round(TRIM_QUIET_S / TRIM_FRAME_S)))
    groups = []                     # (start_frame, end_frame_exclusive)
    g_start = prev = int(speech_idx[0])
    for j in speech_idx[1:]:
        j = int(j)
        if j - prev - 1 >= need:
            groups.append((g_start, prev + 1))
            g_start = j
        prev = j
    groups.append((g_start, prev + 1))

    moan_max = max(1, int(round(TRIM_MOAN_MAX_S / TRIM_FRAME_S)))
    burst_lvl = ref * TRIM_MOAN_RMS_RATIO

    pad_gap = max(1, int(round(TRIM_PAD_GAP_S / TRIM_FRAME_S)))

    end_f = groups[-1][1]
    gi = len(groups) - 1
    stripped = 0
    while gi > 0:
        s, e = groups[gi]
        peak = float(rms[s:e].max())
        gap_before = s - groups[gi - 1][1]
        after = (groups[gi + 1][0] if gi + 1 < len(groups) else n_fr) - e
        if s * frame < keep_min:
            verdict = "kept:in-head"
        elif (e - s) > moan_max:
            verdict = "kept:too-long"
        elif peak >= burst_lvl:
            verdict = "kept:too-loud"
        elif gap_before < pad_gap:
            verdict = "kept:attached"
        else:
            verdict = "stripped"
        if TRIM_DEBUG:
            print(f"[trim]   g{gi}: {s * frame / sr:.2f}-{e * frame / sr:.2f}s "
                  f"dur={(e - s) * frame / sr:.2f}s peak/ref={peak / ref:.2f} "
                  f"gap={gap_before * frame / sr:.2f}s "
                  f"tail={after * frame / sr:.2f}s -> {verdict}", flush=True)
        if verdict != "stripped":
            break
        gi -= 1
        end_f = groups[gi][1]
        stripped += 1

    cut = min(audio.size, end_f * frame + int(sr * TRIM_MARGIN_S))
    if TRIM_DEBUG:
        print(f"[trim] n_real={n_real} n_bucket={n_bucket} "
              f"audio={audio.size / sr:.2f}s groups={len(groups)} "
              f"stripped={stripped} cut={cut / sr:.2f}s "
              f"ref={ref:.4f} thresh={thresh:.4f}", flush=True)
    if cut >= audio.size:
        return audio

    out = audio[:cut].copy()
    fade = min(int(sr * TRIM_FADE_S), out.size)
    if fade > 0:
        out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return out


# ----------------------------------------------------------------------
# backends
# ----------------------------------------------------------------------

class OrtCpuBackend:
    name = "ort-cpu"

    def __init__(self, model_path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(
            model_path, sess_options=so, providers=["CPUExecutionProvider"])

    def infer(self, tokens, style, speed):
        return self.sess.run(None, {
            "tokens": tokens, "style": style, "speed": speed})[0]


class OvBackend:
    """OpenVINO direct runtime. Compiles per padded bucket length and
    caches compiled models in-process (plus OV CACHE_DIR on disk)."""

    def __init__(self, model_path, device, precision, cache_dir):
        import openvino as ov
        self.ov = ov
        self.core = ov.Core()
        print(f"[{{}}] openvino={{}}".format(
            f"ov-{device.lower()}", ov.get_version()), flush=True)
        self.model_path = model_path
        self.device = device
        self.precision = precision
        self.cache_dir = cache_dir
        self.name = f"ov-{device.lower()}"
        self._compiled = {}     # bucket -> (compiled, infer_request)
        # one infer request per bucket + FastAPI thread pool = "Infer
        # Request is busy" under concurrent posts; serialize infers (the
        # device serializes anyway) and protect compile-dict updates
        self._lock = threading.Lock()

    def _bucket(self, n):
        for b in PAD_BUCKETS:
            if n <= b:
                return b
        return PAD_BUCKETS[-1]

    def _get(self, bucket):
        if bucket in self._compiled:
            return self._compiled[bucket]
        ov = self.ov
        model = self.core.read_model(self.model_path)
        want = {"tokens": ov.PartialShape([1, bucket]),
                "style": ov.PartialShape([1, 256]),
                "speed": ov.PartialShape([1])}
        reshape_map = {}
        for inp in model.inputs:
            for key, psh in want.items():
                if key in inp.get_names() or inp.any_name == key:
                    reshape_map[inp.any_name] = psh
        model.reshape(reshape_map)
        config = {"PERFORMANCE_HINT": "LATENCY",
                  "INFERENCE_PRECISION_HINT": self.precision}
        if self.cache_dir:
            config["CACHE_DIR"] = self.cache_dir
        t0 = time.time()
        compiled = self.core.compile_model(model, self.device, config)
        print(f"[{self.name}] compiled bucket={bucket} "
              f"in {time.time()-t0:.1f}s", flush=True)
        pair = (compiled, compiled.create_infer_request())
        self._compiled[bucket] = pair
        return pair

    def infer(self, tokens, style, speed):
        n = tokens.shape[1]
        bucket = self._bucket(n)
        padded = n < bucket
        if padded:  # pad with pad-token 0; the vocoder voices these -> trim
            tokens = np.pad(tokens, ((0, 0), (0, bucket - n)))
        with self._lock:
            _, req = self._get(bucket)
            result = req.infer(
                {"tokens": tokens, "style": style, "speed": speed})
        audio = np.asarray(list(result.values())[0]).reshape(-1)
        if padded:
            audio = trim_pad_tail(audio, n, bucket)
        return audio


class OvGenAIBackend:
    """OpenVINO GenAI Text2SpeechPipeline (official Kokoro int8 pack).

    Token-id infer() is not used — GenAI owns G2P/tokenization. Synthesize
    branches on ``kind == "genai"`` and calls generate_text() per text chunk.
    Speed is native GenAI ``speed=`` (no server-side resample double-apply).
    """

    kind = "genai"

    def __init__(self, model_dir, device):
        # Optional G2P helper used by S0 scripts (nice-to-have on host).
        try:
            import espeakng_loader
            os.environ.setdefault(
                "MISAKI_ESPEAK_LIBRARY", espeakng_loader.get_library_path())
            os.environ.setdefault(
                "ESPEAK_DATA_PATH", espeakng_loader.get_data_path())
        except Exception as e:
            print(f"[ovgenai] espeakng_loader setup skip: {e}", flush=True)

        import openvino as ov
        import openvino_genai as og

        self.ov = ov
        self.model_dir = model_dir
        self.device = device  # "GPU" or "CPU"
        self.name = f"ovgenai-{device.lower()}"
        ov_ver = getattr(ov, "__version__", ov.get_version())
        genai_ver = getattr(og, "__version__", "unknown")
        print(f"[{self.name}] openvino={ov_ver} genai={genai_ver}",
              flush=True)
        print(f"[{self.name}] model_dir={model_dir}", flush=True)
        t0 = time.time()
        self.pipe = og.Text2SpeechPipeline(str(model_dir), device)
        print(f"[{self.name}] Text2SpeechPipeline loaded in "
              f"{time.time() - t0:.1f}s", flush=True)
        self._emb_shape = tuple(self.pipe.get_speaker_embedding_shape())
        self._emb_cache = {}  # voice name -> ov.Tensor
        self._lock = threading.Lock()
        # Populate voice list before any request so parse_voice_spec works.
        global _GENAI_VOICES
        names = self.list_voice_names()
        _GENAI_VOICES = set(names)
        print(f"[{self.name}] voices={len(names)} emb_shape={self._emb_shape}",
              flush=True)

    def list_voice_names(self):
        voices_dir = os.path.join(self.model_dir, "voices")
        if not os.path.isdir(voices_dir):
            return []
        return sorted(
            f[:-4] for f in os.listdir(voices_dir)
            if f.endswith(".bin") and os.path.isfile(os.path.join(voices_dir, f))
        )

    def _speaker_embedding(self, voice):
        if voice in self._emb_cache:
            return self._emb_cache[voice]
        path = os.path.join(self.model_dir, "voices", f"{voice}.bin")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"genai voice embedding missing: {path}")
        emb = np.fromfile(path, dtype=np.float32)
        expected = int(np.prod(self._emb_shape))
        if emb.size != expected:
            raise ValueError(
                f"voice {voice!r} size {emb.size} != expected {expected} "
                f"for shape {self._emb_shape}")
        tensor = self.ov.Tensor(emb.reshape(self._emb_shape))
        self._emb_cache[voice] = tensor
        return tensor

    def generate_text(self, text, voice, speed, lang):
        """Return float32 mono audio for one text chunk (native speed)."""
        emb = self._speaker_embedding(voice)
        with self._lock:
            gen = self.pipe.generate(
                text, emb, language=lang, speed=float(speed))
        audio = np.array(gen.speeches[0].data, dtype=np.float32).reshape(-1)
        return np.nan_to_num(audio)

    def infer(self, tokens, style, speed):
        raise NotImplementedError(
            "OvGenAIBackend does not implement token-id infer(); "
            "use generate_text() via kind='genai' synthesize path")


def make_backend(name):
    if name == "ort-cpu":
        return OrtCpuBackend(MODEL_PATH)
    if name == "ov-cpu":
        return OvBackend(MODEL_PATH, "CPU", "f32", CACHE_DIR)
    if name == "ov-gpu":
        # LEGACY I0.5 — patched ONNX demo; superseded for steady by ovgenai-gpu
        return OvBackend(MODEL_PATH, "GPU", GPU_PRECISION, CACHE_DIR)
    if name == "ovgenai-gpu":
        return OvGenAIBackend(resolve_genai_model_dir(), "GPU")
    if name == "ovgenai-cpu":
        return OvGenAIBackend(resolve_genai_model_dir(), "CPU")
    raise ValueError(f"unknown backend {name!r}")


# ----------------------------------------------------------------------
# synthesis
# ----------------------------------------------------------------------

_VOICES = None
# Set of pack voice stems when an ovgenai backend is active; None otherwise.
_GENAI_VOICES = None


def voices():
    """NPZ voices for ort/ov token-id backends."""
    global _VOICES
    if _VOICES is None:
        _VOICES = np.load(VOICES_PATH)
    return _VOICES


def available_voice_names():
    """Voice name set for the active backend (genai pack or NPZ)."""
    if _GENAI_VOICES is not None:
        return _GENAI_VOICES
    return set(voices().keys())


# Common leftover / alternate names from older Kokoro-FastAPI installs.
VOICE_ALIASES = {
    "bf_v0isabella": "bf_isabella",
    "bf_v0emma": "bf_emma",
    "bf_v0alice": "bf_alice",
    "bf_v0lily": "bf_lily",
    "af_v0bella": "af_bella",
    "af_v0sarah": "af_sarah",
    "af_v0nicole": "af_nicole",
    "af_v0sky": "af_sky",
    "am_v0adam": "am_adam",
    "am_v0michael": "am_michael",
    "bm_v0george": "bm_george",
    "bm_v0lewis": "bm_lewis",
}


def _canon_voice_token(token):
    token = (token or "").strip()
    if not token:
        return None
    token = OPENAI_VOICE_MAP.get(token, token)
    token = VOICE_ALIASES.get(token, token)
    known = available_voice_names()
    if token not in known and "_v0" in token:
        alt = token.replace("_v0", "_", 1)
        if alt in known:
            token = alt
    return token if token in known else None


_BLEND_PART = re.compile(
    r"^\s*([A-Za-z0-9_]+)(?:\s*\(\s*([0-9]*\.?[0-9]+)\s*\))?\s*$")


def parse_voice_spec(spec):
    """Parse plain voice or Kokoro-FastAPI blend: a(1)+b(2)+c."""
    spec = (spec or DEFAULT_VOICE).strip()
    if not spec:
        spec = DEFAULT_VOICE
    single = _canon_voice_token(spec)
    if single is not None and "+" not in spec:
        return [(single, 1.0)], single
    parts = []
    for raw in spec.split("+"):
        m = _BLEND_PART.match(raw)
        if not m:
            return None
        name = _canon_voice_token(m.group(1))
        if name is None:
            return None
        w = float(m.group(2)) if m.group(2) is not None else 1.0
        if w < 0:
            return None
        parts.append((name, w))
    if not parts:
        return None
    total = sum(w for _, w in parts)
    if total <= 0:
        return None
    parts = [(n, w / total) for n, w in parts]
    label = "+".join(f"{n}({w:g})" for n, w in parts)
    return parts, label


def resolve_voice(name):
    parsed = parse_voice_spec(name)
    if not parsed:
        return None
    parts, _ = parsed
    if len(parts) == 1:
        return parts[0][0]
    return parts


def style_for_parts(parts, n_tokens):
    acc = None
    for vname, w in parts:
        ref = voices()[vname]
        row = ref[min(n_tokens, ref.shape[0] - 1)].astype(np.float32)
        acc = row * w if acc is None else acc + row * w
    return acc.reshape(1, 256)


def _prewarm_text_for(bucket, lang="en-us"):
    """Natural text grown to just under the bucket's token capacity, so
    synthesize() runs one chunk that fills the target bucket with mostly
    REAL tokens. Near-capacity matters: a short real text padded into the
    bucket (the 15-token startup warmup) was measured NOT to warm a later
    55-token request, while a ~bucket-sized text was (notes/18 follow-up).
    Word steps are ~3-8 tokens, so the landing is a few tokens under the
    bucket, never over."""
    pool = ("the quick brown fox jumps over the lazy dog while seven "
            "silver swans swim smoothly south past bright blue boxes").split()
    text = ""
    for i in range(1200):
        cand = (text + " " + pool[i % len(pool)]).strip()
        if len(phonemes_to_ids(cand, lang)) + 2 > bucket:
            break
        text = cand
    return text or "Warm up"


def synthesize(backend, text, voice_spec, speed, chunk_cache=None):
    """Synthesize text to float audio.

    Always quantizes each chunk to int16 PCM and dequants back to float before
    concat (and gap zeros stay float). That path is identical with cache off or
    on so C2 partial assembly matches cache-off byte-for-byte.

    Dual path:
      - ort/ov: token-id chunks; C2 key ``c2ids:`` + comma-joined ids
      - genai (kind=='genai'): text-string chunks; C2 key ``c2txt:`` + exact
        chunk string; native GenAI speed (no server resample)

    Returns (audio, total_tokens, label, c2_hits, c2_misses). On unknown voice
    or genai blend: (None, 0, None, 0, 0).
    """
    parsed = parse_voice_spec(voice_spec)
    if not parsed:
        return None, 0, None, 0, 0
    parts, label = parsed
    primary = parts[0][0]
    lang = "en-gb" if primary.startswith(("bf_", "bm_")) else "en-us"
    is_genai = getattr(backend, "kind", None) == "genai"

    if is_genai:
        # I0.2: single voice only — blends not supported on GenAI path.
        if "+" in (voice_spec or "") or len(parts) > 1:
            print("[synthesize] ovgenai backends support a single voice only "
                  f"(no blends); got {voice_spec!r}", flush=True)
            return None, 0, None, 0, 0
        text_chunks = chunk_text_strings(text, lang)
        if not text_chunks:
            return np.zeros(0, dtype=np.float32), 0, label, 0, 0
        gap = np.zeros(int(SR * CHUNK_GAP_S), dtype=np.float32)
        pieces = []
        total_tokens = 0
        c2_hits = 0
        c2_misses = 0
        for chunk_str in text_chunks:
            total_tokens += len(chunk_str)
            key = None
            pcm = None
            if chunk_cache is not None:
                # Exact post-chunker string (I0 G2 / notes/54).
                text_unit = "c2txt:" + chunk_str
                key = chunk_cache.build_key(label, speed, text_unit)
                pcm = chunk_cache.read_entry(key)
            if pcm is not None:
                c2_hits += 1
            else:
                audio = backend.generate_text(
                    chunk_str, primary, speed, lang)
                audio = np.nan_to_num(
                    np.asarray(audio, dtype=np.float32)).reshape(-1)
                pcm = audio_float_to_pcm_int16_bytes(audio)
                c2_misses += 1
                if chunk_cache is not None and key is not None:
                    meta = {
                        "schema_ver": TTS_CACHE_SCHEMA_VER,
                        "backend_id": chunk_cache.backend_id,
                        "model_fp": chunk_cache.model_fp,
                        "voice": label,
                        "speed": f"{speed:.6g}",
                        "sample_fmt": TTS_SAMPLE_FMT,
                        "tier": "chunk",
                        "text_unit_prefix": ("c2txt:" + chunk_str)[:64],
                        "created_unix": time.time(),
                        "n_samples": len(pcm) // 2,
                        "key_hex": key,
                    }
                    try:
                        chunk_cache.write_entry(key, pcm, meta)
                    except Exception as e:
                        print(f"[cache] c2 write failed: {e}", flush=True)
            piece = pcm_int16_bytes_to_float(pcm)
            pieces.append(piece)
            pieces.append(gap)
        if pieces:
            pieces = pieces[:-1]
        audio = (np.concatenate(pieces) if pieces
                 else np.zeros(0, np.float32))
        return audio, total_tokens, label, c2_hits, c2_misses

    # --- ort / ov token-id path (unchanged) ---
    chunks = chunk_text(text, lang)
    if not chunks:
        return np.zeros(0, dtype=np.float32), 0, label, 0, 0
    gap = np.zeros(int(SR * CHUNK_GAP_S), dtype=np.float32)
    pieces = []
    total_tokens = 0
    c2_hits = 0
    c2_misses = 0
    for ids in chunks:
        total_tokens += len(ids)
        key = None
        pcm = None
        if chunk_cache is not None:
            text_unit = "c2ids:" + ",".join(str(i) for i in ids)
            key = chunk_cache.build_key(label, speed, text_unit)
            pcm = chunk_cache.read_entry(key)
        if pcm is not None:
            c2_hits += 1
        else:
            style = style_for_parts(parts, len(ids))
            tokens = np.array([[0, *ids, 0]], dtype=np.int64)
            audio = np.asarray(backend.infer(
                tokens, style, np.array([speed], dtype=np.float32))).reshape(-1)
            audio = np.nan_to_num(audio.astype(np.float32)).reshape(-1)
            pcm = audio_float_to_pcm_int16_bytes(audio)
            c2_misses += 1
            if chunk_cache is not None and key is not None:
                meta = {
                    "schema_ver": TTS_CACHE_SCHEMA_VER,
                    "backend_id": chunk_cache.backend_id,
                    "model_fp": chunk_cache.model_fp,
                    "voice": label,
                    "speed": f"{speed:.6g}",
                    "sample_fmt": TTS_SAMPLE_FMT,
                    "tier": "chunk",
                    "text_unit_prefix": text_unit[:64],
                    "created_unix": time.time(),
                    "n_samples": len(pcm) // 2,
                    "key_hex": key,
                }
                try:
                    chunk_cache.write_entry(key, pcm, meta)
                except Exception as e:
                    print(f"[cache] c2 write failed: {e}", flush=True)
        piece = pcm_int16_bytes_to_float(pcm)
        pieces.append(piece)
        pieces.append(gap)
    if pieces:
        pieces = pieces[:-1]
    audio = np.concatenate(pieces) if pieces else np.zeros(0, np.float32)
    return audio, total_tokens, label, c2_hits, c2_misses


def audio_float_to_pcm_int16_bytes(audio):
    """Clip/scale float audio to int16 LE PCM (same as to_wav_bytes body)."""
    a = np.clip(audio, -1.0, 1.0)
    pcm = (a * 32767.0).astype(np.int16)
    return pcm.tobytes()


def pcm_int16_bytes_to_float(pcm_bytes):
    """int16 LE PCM bytes -> float32 in ~[-1, 1] (inverse of encode path)."""
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0


def pcm_int16_bytes_to_wav(pcm_bytes):
    """Wrap raw int16 LE mono @ SR into a WAV container (shared hit/miss path)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm_bytes)
    return buf.getvalue()


def to_wav_bytes(audio):
    """Float audio -> WAV. Byte-identical to cache-hit path for same samples."""
    return pcm_int16_bytes_to_wav(audio_float_to_pcm_int16_bytes(audio))


def transcode(wav_bytes, fmt):
    """wav -> mp3/opus/flac via ffmpeg if available; else None."""
    if shutil.which("ffmpeg") is None:
        return None
    codec = {"mp3": ["-f", "mp3"], "opus": ["-f", "opus"],
             "flac": ["-f", "flac"], "aac": ["-f", "adts"]}.get(fmt)
    if codec is None:
        return None
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", "pipe:0", *codec, "pipe:1"],
        input=wav_bytes, capture_output=True)
    return p.stdout if p.returncode == 0 and p.stdout else None


# ----------------------------------------------------------------------
# C1 response-level TTS disk cache
# ----------------------------------------------------------------------

def sha256_file(path):
    """Full-file sha256 hex (strong model fingerprint; once at startup)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_fingerprint_for_cache():
    """model_fp field for TTS cache keys.

    GenAI pack: sha256 of openvino_model.bin if present else openvino_model.xml.
    ONNX file backends: sha256 of MODEL_PATH.
    """
    if BACKEND.startswith("ovgenai"):
        model_dir = resolve_genai_model_dir()
        bin_path = os.path.join(model_dir, "openvino_model.bin")
        xml_path = os.path.join(model_dir, "openvino_model.xml")
        if os.path.isfile(bin_path):
            return sha256_file(bin_path)
        if os.path.isfile(xml_path):
            return sha256_file(xml_path)
        raise FileNotFoundError(
            f"genai model fingerprint: no openvino_model.bin/.xml in {model_dir}")
    return sha256_file(MODEL_PATH)


def active_model_path():
    """Human-facing model path for health / startup logs."""
    if BACKEND.startswith("ovgenai"):
        return resolve_genai_model_dir()
    return MODEL_PATH


def build_tts_cache_key(schema_ver, backend_id, model_fp, voice, speed,
                        sample_fmt, text_unit):
    """sha256 hex of pipe-joined key fields.

    Stable delimiter is ASCII '|'. Field order (do not reorder without
    bumping schema_ver):
      schema_ver | backend_id | model_fp | voice | speed | sample_fmt | text_unit
    speed is already clipped; formatted with '{speed:.6g}'.
    text_unit is the exact request string passed to synthesize.
    """
    speed_s = speed if isinstance(speed, str) else f"{speed:.6g}"
    material = "|".join((
        str(schema_ver),
        backend_id,
        model_fp,
        voice,
        speed_s,
        sample_fmt,
        text_unit,
    ))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class TtsResponseCache:
    """Shared C1/C2 PCM cache on disk under $KOKORO_TTS_CACHE_DIR/v1/.

    Layout: v1/<ab>/<fullhex>.pcm + .json
    C1 text_unit = full request string; C2 = 'c2ids:' + comma-joined token ids
    (ort/ov) or 'c2txt:' + exact chunk string (ovgenai). backend_id in the key
    firewalls cross-backend serving.
    Meta may include tier: "response" | "chunk". Atomic writes via *.tmp +
    os.replace. Process-local lock on miss-fill write and eviction.
    """

    def __init__(self, root_dir, max_mb, backend_id, model_fp):
        self.root = os.path.join(root_dir, "v1")
        self.max_bytes = int(float(max_mb) * 1024 * 1024)
        self.backend_id = backend_id
        self.model_fp = model_fp
        self._lock = threading.Lock()
        os.makedirs(self.root, exist_ok=True)

    def build_key(self, voice, speed, text_unit):
        return build_tts_cache_key(
            TTS_CACHE_SCHEMA_VER, self.backend_id, self.model_fp,
            voice, speed, TTS_SAMPLE_FMT, text_unit)

    def pcm_path(self, key_hex):
        return os.path.join(self.root, key_hex[:2], f"{key_hex}.pcm")

    def json_path(self, key_hex):
        return os.path.join(self.root, key_hex[:2], f"{key_hex}.json")

    def read_entry(self, key_hex):
        """Return raw int16 PCM bytes on hit, else None. Touches mtime (LRU)."""
        pcm_p = self.pcm_path(key_hex)
        json_p = self.json_path(key_hex)
        try:
            if not (os.path.isfile(pcm_p) and os.path.isfile(json_p)):
                return None
            with open(pcm_p, "rb") as f:
                pcm = f.read()
            if not pcm or (len(pcm) % 2) != 0:
                return None
            now = time.time()
            try:
                os.utime(pcm_p, (now, now))
                os.utime(json_p, (now, now))
            except OSError:
                pass
            return pcm
        except OSError:
            return None

    def write_entry(self, key_hex, pcm_bytes, meta):
        """Atomic write of pcm+json; then lazy eviction if over cap."""
        pcm_p = self.pcm_path(key_hex)
        json_p = self.json_path(key_hex)
        d = os.path.dirname(pcm_p)
        os.makedirs(d, exist_ok=True)
        meta_bytes = json.dumps(meta, ensure_ascii=False, separators=(",", ":"))
        meta_bytes = meta_bytes.encode("utf-8")
        with self._lock:
            self._atomic_write(pcm_p, pcm_bytes)
            self._atomic_write(json_p, meta_bytes)
            # Never delete the entry we just wrote; allow single-entry overshoot
            # when one utterance alone exceeds MAX_MB (tiny-cap / long audio).
            self._evict_if_needed(protect_key=key_hex)

    @staticmethod
    def _atomic_write(path, data):
        tmp = path + ".tmp"
        try:
            with open(tmp, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _evict_if_needed(self, protect_key=None):
        """If v1 tree > max_bytes, delete oldest-mtime .pcm+.json pairs.

        protect_key: if set, never unlink that entry (just-written path).
        If a single protected entry is larger than the cap, the tree may
        remain over budget until older entries exist to reclaim — better
        than immediately discarding the write we just served.
        """
        total = 0
        # (mtime, base_without_ext, key_hex) for each .pcm
        pairs = []
        for dirpath, _dirnames, filenames in os.walk(self.root):
            for name in filenames:
                if name.endswith(".tmp"):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                total += st.st_size
                if name.endswith(".pcm"):
                    base = path[:-4]
                    key_hex = name[:-4]
                    pairs.append((st.st_mtime, base, key_hex))
        if total <= self.max_bytes:
            return
        pairs.sort(key=lambda x: x[0])  # oldest first
        for _mtime, base, key_hex in pairs:
            if total <= self.max_bytes:
                break
            if protect_key is not None and key_hex == protect_key:
                continue
            for path in (base + ".pcm", base + ".json"):
                try:
                    sz = os.path.getsize(path)
                    os.unlink(path)
                    total -= sz
                except OSError:
                    pass


# ----------------------------------------------------------------------
# app
# ----------------------------------------------------------------------

def build_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import Response
    from pydantic import BaseModel

    app = FastAPI(title="Kokoro TTS (intel-igpu-tts)", version="2.0.0")
    state = {"backend": None, "tts_cache": None}

    class SpeechRequest(BaseModel):
        # Open WebUI / Kokoro-FastAPI may send extra keys; ignore unknowns.
        model_config = {"extra": "ignore"}

        input: str
        model: str = "kokoro"
        voice: str = DEFAULT_VOICE
        response_format: str = "wav"
        speed: float = 1.0

    @app.on_event("startup")
    def _startup():
        model_disp = active_model_path()
        print(f"[server] model={model_disp}")
        if BACKEND.startswith("ovgenai"):
            print(f"[server] voices=pack:{model_disp}/voices/*.bin")
        else:
            print(f"[server] voices={VOICES_PATH}")
        print(f"[server] backend={BACKEND}"
              + (f" precision={GPU_PRECISION}" if BACKEND == "ov-gpu" else ""))
        state["backend"] = make_backend(BACKEND)
        # Shared C1/C2 store when either tier is active.
        if TTS_RESPONSE_CACHE or TTS_CHUNK_CACHE:
            t_fp = time.time()
            print(f"[server] TTS cache: hashing model for model_fp...",
                  flush=True)
            model_fp = model_fingerprint_for_cache()
            state["tts_cache"] = TtsResponseCache(
                TTS_CACHE_DIR, TTS_CACHE_MAX_MB, BACKEND, model_fp)
            tiers = []
            if TTS_RESPONSE_CACHE:
                tiers.append("response(C1)")
            if TTS_CHUNK_CACHE:
                tiers.append("chunk(C2)")
            print(f"[server] TTS cache ON tiers={'+'.join(tiers)} "
                  f"dir={TTS_CACHE_DIR} max_mb={TTS_CACHE_MAX_MB:g} "
                  f"tier_env={TTS_CACHE_TIER} schema_ver={TTS_CACHE_SCHEMA_VER} "
                  f"model_fp={model_fp[:16]}... "
                  f"({time.time() - t_fp:.2f}s)",
                  flush=True)
        else:
            state["tts_cache"] = None
            if TTS_CACHE_ON:
                print(f"[server] TTS cache flag on but tier={TTS_CACHE_TIER!r}; "
                      f"no C1/C2 (need response|chunk|both)",
                      flush=True)
            else:
                print("[server] TTS cache OFF", flush=True)
        # warm the default path so first request isn't cold
        try:
            warm_cc = (state["tts_cache"] if TTS_CHUNK_CACHE else None)
            synthesize(state["backend"], "Warm up.", DEFAULT_VOICE, 1.0,
                       chunk_cache=warm_cc)
            print("[server] warmup OK")
            # optional OV pre-warm. Warm is SHAPE-keyed (notes/19–20), not
            # bucket-wide: near-capacity bucket text only warms that shape;
            # KOKORO_WARM_TEXT pins exact demo phrases for repeat traffic.
            # GenAI has no _bucket — WARM_BUCKETS is skipped (hasattr gate).
            be = state["backend"]
            if WARM_BUCKETS and hasattr(be, "_bucket"):
                for b in WARM_BUCKETS:
                    b = be._bucket(int(b))
                    t0 = time.time()
                    synthesize(be, _prewarm_text_for(b),
                               DEFAULT_VOICE, 1.0, chunk_cache=warm_cc)
                    print(f"[server] pre-warmed bucket={b} via "
                          f"synthesize in {time.time() - t0:.1f}s "
                          f"(shape-keyed; not all traffic)",
                          flush=True)
            for wt in WARM_TEXTS:
                t0 = time.time()
                synthesize(be, wt, DEFAULT_VOICE, 1.0, chunk_cache=warm_cc)
                preview = wt if len(wt) <= 48 else wt[:45] + "..."
                print(f"[server] pre-warmed text={preview!r} in "
                      f"{time.time() - t0:.1f}s",
                      flush=True)
        except Exception as e:
            print(f"[server] warmup failed: {e}")

    @app.get("/health")
    def health():
        body = {
            "status": "ok",
            "backend": BACKEND,
            "model": active_model_path(),
            "gpu_precision": (
                GPU_PRECISION if BACKEND == "ov-gpu" else None),
        }
        if BACKEND.startswith("ovgenai"):
            body["genai"] = True
        return body

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [
            {"id": "kokoro", "object": "model", "owned_by": "local"},
            {"id": "tts-1", "object": "model", "owned_by": "local"},
        ]}

    @app.get("/v1/audio/voices")
    def list_voices():
        return {"voices": sorted(available_voice_names()),
                "openai_aliases": OPENAI_VOICE_MAP,
                "default": DEFAULT_VOICE}

    def _speech_response(wav_bytes, fmt, headers):
        """Shared container encode path for cache hit and miss."""
        if fmt in ("wav", "pcm", ""):
            return Response(wav_bytes, media_type="audio/wav", headers=headers)
        enc = transcode(wav_bytes, fmt)
        if enc is None:
            return Response(wav_bytes, media_type="audio/wav", headers=headers)
        headers = dict(headers)
        headers["X-Kokoro-Format"] = fmt
        media = {"mp3": "audio/mpeg", "opus": "audio/ogg",
                 "flac": "audio/flac", "aac": "audio/aac"}[fmt]
        return Response(enc, media_type=media, headers=headers)

    @app.post("/v1/audio/speech")
    def speech(req: SpeechRequest):
        if not req.input or not req.input.strip():
            raise HTTPException(400, "empty input")
        parsed = parse_voice_spec(req.voice)
        if parsed is None:
            blend_hint = (
                " (ovgenai backends: single voice only, no blends)"
                if BACKEND.startswith("ovgenai")
                else " (blends like a(1)+b(2) supported)")
            raise HTTPException(400, f"unknown voice {req.voice!r}; "
                                     f"see /v1/audio/voices{blend_hint}")
        _parts, voice_label = parsed
        if (BACKEND.startswith("ovgenai")
                and ("+" in (req.voice or "") or len(_parts) > 1)):
            raise HTTPException(
                400,
                f"ovgenai backends support a single voice only "
                f"(no blends); got {req.voice!r}")
        speed = float(np.clip(req.speed, 0.5, 2.0))
        # Exact text unit passed to synthesize (and used as C1 cache key field).
        text_unit = req.input
        fmt = req.response_format.lower()
        cache = state.get("tts_cache")
        t0 = time.time()

        # --- C1 response cache lookup (skip synthesize on hit) ---
        key_hex = None
        if cache is not None and TTS_RESPONSE_CACHE:
            key_hex = cache.build_key(voice_label, speed, text_unit)
            pcm = cache.read_entry(key_hex)
            if pcm is not None:
                wall_s = time.time() - t0
                n_samples = len(pcm) // 2
                dur = n_samples / float(SR)
                rtf = wall_s / max(dur, 1e-6)
                print(f"[speech] voice={voice_label} tokens=- "
                      f"audio={dur:.2f}s infer={wall_s:.2f}s "
                      f"rtf={rtf:.2f} cache=hit", flush=True)
                wav_bytes = pcm_int16_bytes_to_wav(pcm)
                headers = {
                    "X-Kokoro-Backend": BACKEND,
                    "X-Kokoro-RTF": f"{rtf:.2f}",
                    "X-Kokoro-Format": "wav",
                    "X-Kokoro-Cache": "hit",
                }
                return _speech_response(wav_bytes, fmt, headers)

        chunk_cache = cache if (cache is not None and TTS_CHUNK_CACHE) else None
        audio, n_tokens, voice_label, c2_hits, c2_misses = synthesize(
            state["backend"], text_unit, req.voice, speed,
            chunk_cache=chunk_cache)
        infer_s = time.time() - t0
        if audio is None:
            raise HTTPException(400, f"unknown voice {req.voice!r}")
        if audio.size == 0:
            raise HTTPException(400, "no synthesizable content")
        dur = audio.size / SR
        rtf = infer_s / max(dur, 1e-6)

        # Cache status for header/log: C2 all-hit counts as hit; mixed=partial.
        cache_status = None
        if c2_misses == 0 and c2_hits > 0:
            cache_status = "hit"
        elif c2_hits > 0 and c2_misses > 0:
            cache_status = "partial"
        elif TTS_RESPONSE_CACHE or TTS_CHUNK_CACHE:
            cache_status = "miss"

        log_extra = ""
        if cache_status is not None:
            log_extra = f" cache={cache_status}"
            if TTS_CHUNK_CACHE:
                log_extra += f" c2_hits={c2_hits} c2_misses={c2_misses}"
        print(f"[speech] voice={voice_label} tokens={n_tokens} "
              f"audio={dur:.2f}s infer={infer_s:.2f}s "
              f"rtf={rtf:.2f}{log_extra}", flush=True)

        # Encode once via shared int16 path so cache store matches served WAV.
        pcm_bytes = audio_float_to_pcm_int16_bytes(audio)
        wav_bytes = pcm_int16_bytes_to_wav(pcm_bytes)
        headers = {"X-Kokoro-Backend": BACKEND,
                   "X-Kokoro-RTF": f"{rtf:.2f}",
                   "X-Kokoro-Format": "wav"}
        if cache_status is not None:
            headers["X-Kokoro-Cache"] = cache_status

        # Always write C1 full-response entry when response tier is on.
        if cache is not None and TTS_RESPONSE_CACHE:
            if key_hex is None:
                key_hex = cache.build_key(voice_label, speed, text_unit)
            meta = {
                "schema_ver": TTS_CACHE_SCHEMA_VER,
                "backend_id": BACKEND,
                "model_fp": cache.model_fp,
                "voice": voice_label,
                "speed": f"{speed:.6g}",
                "sample_fmt": TTS_SAMPLE_FMT,
                "tier": "response",
                "text_sha256": hashlib.sha256(
                    text_unit.encode("utf-8")).hexdigest(),
                "created_unix": time.time(),
                "n_samples": int(audio.size),
                "duration_s": float(dur),
                "key_hex": key_hex,
            }
            try:
                cache.write_entry(key_hex, pcm_bytes, meta)
            except Exception as e:
                print(f"[cache] c1 write failed: {e}", flush=True)

        return _speech_response(wav_bytes, fmt, headers)

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8880)
    args = ap.parse_args()
    import uvicorn
    uvicorn.run(build_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()