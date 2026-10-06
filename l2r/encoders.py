"""Frozen pretrained encoders. The router never updates them; each attribute is read by the encoder suited to it.

    clap         LAION-CLAP audio embedding (general sound and music), 512-d
    muq_mulan    MuQ-MuLan music embedding, 512-d
    whisper      Whisper encoder, mean-pooled (speaking style, age, number of speakers), 512-d
    emotion2vec  emotion2vec+ utterance embedding (emotion), 1024-d
    wavlm_sv     WavLM speaker-verification x-vector (gender), 512-d
    birdnet      BirdNET embedding (bird species, a new domain), 1024-d
    BEATs        AudioSet-fine-tuned BEATs frames (sound events and activity), 768-d per frame
    PANNs        Cnn14 frame features (event timeline), 2048-d per frame

`build_encoder(name)` returns an object with `.embed(paths) -> [N, D]` (L2-normalised) for the first six.
BEATs and PANNs are frame-level and are loaded with `load_beats` / `load_panns`.
"""
from __future__ import annotations

import sys

import numpy as np

from l2r.common import ckpt, load_config

SR = 16000


def _load_audio(path, sr=SR):
    import librosa
    return librosa.load(str(path), sr=sr, mono=True)[0]


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _unit(e):
    return e / (np.linalg.norm(e, axis=-1, keepdims=True) + 1e-8)


class ClapEncoder:
    """LAION-CLAP audio embeddings, deterministic per file.

    CLAP was trained on 10 s windows. A longer file is cut to its centre 10 s here, so the library never
    draws a random window; a shorter file is repeat-padded by the library, which is deterministic."""
    SR = 48000
    WIN = 480000

    def __init__(self, cfg, enable_fusion=False, **_):
        import laion_clap
        self.m = laion_clap.CLAP_Module(enable_fusion=enable_fusion)
        self.m.load_ckpt()
        self.bs = cfg["encoders"]["batch_size"]
        self._cache: dict[str, np.ndarray] = {}

    def _window(self, path):
        y = _load_audio(path, sr=self.SR)
        if len(y) > self.WIN:
            y = y[(len(y) - self.WIN) // 2:][:self.WIN]
        return y if len(y) else np.zeros(4800, dtype=np.float32)

    def embed(self, paths):
        paths = [str(p) for p in paths]
        todo = [p for p in dict.fromkeys(paths) if p not in self._cache]
        for i in range(0, len(todo), self.bs):
            group = todo[i:i + self.bs]
            e = np.asarray(self.m.get_audio_embedding_from_data(x=[self._window(p) for p in group], use_tensor=False)).astype(np.float32)
            for p, v in zip(group, e):
                self._cache[p] = v / (np.linalg.norm(v) + 1e-8)
        return np.stack([self._cache[p] for p in paths]).astype(np.float32)


class MuQMuLanEncoder:
    """MuQ-MuLan music embedding: 24 kHz input, centre 10 s window."""

    def __init__(self, cfg, model_id="OpenMuQ/MuQ-MuLan-large", **_):
        import torch
        from muq import MuQMuLan
        self.torch = torch
        self.dev = _device()
        self.m = MuQMuLan.from_pretrained(model_id).to(self.dev).eval()
        self.sr = 24000
        self.bs = 8
        self._cache: dict[str, np.ndarray] = {}

    def _wave(self, path):
        y = _load_audio(path, sr=self.sr)
        n = 10 * self.sr
        if len(y) > n:
            y = y[(len(y) - n) // 2:][:n]
        if len(y) < self.sr // 2:
            y = np.pad(y, (0, self.sr // 2 - len(y)))
        return y

    def embed(self, paths):
        paths = [str(p) for p in paths]
        todo = [p for p in dict.fromkeys(paths) if p not in self._cache]
        for i in range(0, len(todo), self.bs):
            ws = [self._wave(p) for p in todo[i:i + self.bs]]
            n = max(len(w) for w in ws)
            x = self.torch.from_numpy(np.stack([np.pad(w, (0, n - len(w))) for w in ws]).astype(np.float32)).to(self.dev)
            with self.torch.no_grad():
                e = self.m(wavs=x).float().cpu().numpy()
            for p, v in zip(todo[i:i + self.bs], e):
                self._cache[p] = v / (np.linalg.norm(v) + 1e-8)
        return np.stack([self._cache[p] for p in paths]).astype(np.float32)


class WhisperEncoder:
    """The Whisper encoder's last hidden state, mean-pooled over time."""

    def __init__(self, cfg, model_id="openai/whisper-base", **_):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.torch = torch
        self.dev = _device()
        self.proc = WhisperProcessor.from_pretrained(model_id)
        self.m = WhisperForConditionalGeneration.from_pretrained(model_id).to(self.dev).eval()
        self.bs = cfg["encoders"]["batch_size"]

    def embed(self, paths):
        out = []
        for i in range(0, len(paths), self.bs):
            audios = [_load_audio(p) for p in paths[i:i + self.bs]]
            feats = self.proc(audios, sampling_rate=SR, return_tensors="pt").input_features.to(self.dev)
            with self.torch.no_grad():
                h = self.m.model.encoder(feats).last_hidden_state
            out.append(_unit(h.mean(dim=1).cpu().numpy()))
        return np.concatenate(out, 0).astype(np.float32)


class Emotion2vecEncoder:
    """emotion2vec+ (FunASR): utterance-level embedding, and the model's own 9-way emotion scores."""

    def __init__(self, cfg, model_id="emotion2vec/emotion2vec_plus_large", hub="hf", **_):
        from funasr import AutoModel
        self.m = AutoModel(model=model_id, hub=hub, disable_update=True, device=_device())
        self.bs = cfg["encoders"]["batch_size"]

    def embed(self, paths):
        out = []
        for i in range(0, len(paths), self.bs):
            res = self.m.generate([str(p) for p in paths[i:i + self.bs]], granularity="utterance", extract_embedding=True)
            for r in res:
                v = r.get("feats")
                if v is None:
                    v = np.asarray(r["scores"], dtype=np.float32)
                out.append(np.asarray(v, dtype=np.float32).reshape(-1))
        w = max(len(v) for v in out)
        return _unit(np.stack([np.pad(v, (0, w - len(v))) for v in out]))

    def classify(self, paths) -> list[tuple[str, float]]:
        """(emotion label, score) per file, from the model's classification head."""
        res = self.m.generate([str(p) for p in paths], granularity="utterance", extract_embedding=False)
        out = []
        for x in res:
            j = int(np.argmax(x["scores"]))
            out.append((str(x["labels"][j]).lower().split("/")[-1].strip(), float(x["scores"][j])))
        return out


class WavLMSVEncoder:
    """Speaker x-vectors from WavLM fine-tuned for speaker verification."""

    def __init__(self, cfg, model_id="microsoft/wavlm-base-plus-sv", **_):
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.torch = torch
        self.dev = _device()
        self.fe = AutoFeatureExtractor.from_pretrained(model_id)
        self.m = WavLMForXVector.from_pretrained(model_id).to(self.dev).eval()
        self.bs = cfg["encoders"]["batch_size"]

    def embed(self, paths):
        out = []
        for i in range(0, len(paths), self.bs):
            audios = [_load_audio(p) for p in paths[i:i + self.bs]]
            inp = self.fe(audios, sampling_rate=SR, return_tensors="pt", padding=True).to(self.dev)
            with self.torch.no_grad():
                emb = self.m(**inp).embeddings
            out.append(emb.float().cpu().numpy())
        return _unit(np.vstack(out).astype(np.float32))


class BirdNETEncoder:
    """BirdNET v2.4 embedding (ONNX, CPU): 3 s windows at 48 kHz with a 1.5 s hop, at most 8 windows per file;
    the mean of the L2-normalised window embeddings. The ONNX file is the public BirdNET model with its
    global-average-pool tensor exported as an output (l2r.checkpoints.birdnet_export)."""
    SR = 48_000
    WIN = 3 * 48_000
    HOP = int(1.5 * 48_000)

    def __init__(self, cfg, onnx="pretrained/birdnet/birdnet_emb.onnx", max_windows=8, **_):
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = 8; so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(ckpt(onnx)), so, providers=["CPUExecutionProvider"])
        self.emb_name = [o.name for o in self.sess.get_outputs() if "GLOBAL_AVG_POOL" in o.name][0]
        self.max_windows = max_windows
        self._cache: dict[str, np.ndarray] = {}

    def _one(self, path):
        import librosa
        import soundfile as sf
        x, sr = sf.read(str(path), dtype="float32", always_2d=True); x = x.mean(axis=1)
        if sr != self.SR:
            x = librosa.resample(x, orig_sr=sr, target_sr=self.SR)
        if len(x) < self.WIN:
            x = np.pad(x, (0, self.WIN - len(x)))
        starts = list(range(0, max(1, len(x) - self.WIN + 1), self.HOP))[: self.max_windows]
        out = self.sess.run([self.emb_name], {"input": np.stack([x[s:s + self.WIN] for s in starts]).astype(np.float32)})[0]
        return _unit(_unit(out).mean(axis=0)).astype(np.float32)

    def embed(self, paths):
        for p in map(str, paths):
            if p not in self._cache:
                self._cache[p] = self._one(p)
        return np.vstack([self._cache[str(p)] for p in paths])


_ENCODERS = {"clap": ClapEncoder, "muq_mulan": MuQMuLanEncoder, "whisper": WhisperEncoder, "emotion2vec": Emotion2vecEncoder,
             "wavlm_sv": WavLMSVEncoder, "birdnet": BirdNETEncoder}


def build_encoder(name: str, cfg: dict | None = None):
    """The frozen encoder `name`, configured by `encoders.<name>` of the config."""
    cfg = cfg or load_config()
    if name not in _ENCODERS:
        raise ValueError(f"unknown encoder {name!r} (known: {sorted(_ENCODERS)})")
    return _ENCODERS[name](cfg, **(cfg["encoders"].get(name) or {}))


# ----------------------------------------------------------------------------------------- frame-level encoders
def load_beats(dev=None, cfg: dict | None = None):
    """BEATs fine-tuned on AudioSet. The model code (BEATs.py, backbone.py, modules.py, quantizer.py, from
    github.com/microsoft/unilm/tree/master/beats) and the checkpoint are expected in `encoders.beats.dir`."""
    import torch
    spec = (cfg or load_config())["encoders"]["beats"]; d = ckpt(spec["dir"])
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    from BEATs import BEATs, BEATsConfig
    ck = torch.load(d / spec["weights"], map_location="cpu", weights_only=False)
    m = BEATs(BEATsConfig(ck["cfg"])); m.load_state_dict(ck["model"])
    return m.to(dev or _device()).eval()


def beats_frames(m, wav16k):
    """[B, samples at 16 kHz] -> [B, T, 768]: encoder output, frequency patches averaged."""
    fb = m.preprocess(wav16k).unsqueeze(1); x = m.patch_embedding(fb); B, D, Tp, Fp = x.shape
    x = x.reshape(B, D, -1).transpose(1, 2); x = m.layer_norm(x)
    if m.post_extract_proj is not None:
        x = m.post_extract_proj(x)
    x, _ = m.encoder(x, padding_mask=None)
    return x.reshape(B, Tp, Fp, -1).mean(2)


PANNS_SR = 32000


def load_panns(dev=None, cfg: dict | None = None):
    """PANNs Cnn14_DecisionLevelMax (527 AudioSet classes). Returns (model, box); after a forward pass
    `box["fc1"]` holds the [B, T, 2048] frame features its classification layer reads."""
    import torch
    from panns_inference.models import Cnn14_DecisionLevelMax
    spec = (cfg or load_config())["encoders"]["panns"]
    m = Cnn14_DecisionLevelMax(sample_rate=PANNS_SR, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000,
                               classes_num=527, interpolate_mode="nearest")
    m.load_state_dict(torch.load(ckpt(spec["weights"]), map_location="cpu", weights_only=False)["model"])
    m.to(dev or _device()).eval()
    box = {}
    m.fc1.register_forward_hook(lambda mod, i, o: box.__setitem__("fc1", o))
    return m, box


def panns_labels(cfg: dict | None = None) -> list[str]:
    """Display names of PANNs' 527 classes, in output order."""
    import csv
    rows = list(csv.DictReader(open(ckpt((cfg or load_config())["encoders"]["panns"]["labels"]))))
    rows.sort(key=lambda r: int(r["index"]))
    return [r["display_name"] for r in rows]


def beats_windows(m, path, max_s: float = 60.0):
    """A clip as consecutive 10 s windows (a tail is kept when longer than 5 s; shorter windows are zero-padded):
    -> ([n windows, 768] mean BEATs frame embedding per window, [(t0, t1)] in seconds)."""
    import librosa
    import torch
    import torchaudio
    n = 10 * PANNS_SR
    y, _ = librosa.load(str(path), sr=PANNS_SR, mono=True, duration=max_s)
    y = np.pad(y, (0, max(0, PANNS_SR - len(y))))
    wins = [y[s:s + n] for s in range(0, max(len(y) - n // 2, 1), n)] or [y]
    dev = next(m.parameters()).device
    W = torch.tensor(np.stack([np.pad(w, (0, n - len(w))) for w in wins])).float().to(dev)
    with torch.no_grad():
        x = beats_frames(m, torchaudio.functional.resample(W, PANNS_SR, 16000)).mean(1).float().cpu().numpy()
    return x, [(10.0 * i, 10.0 * (i + 1)) for i in range(len(wins))]
