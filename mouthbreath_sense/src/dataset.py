print(">>> ACTIVE dataset.py:", __file__)


import os
import torch
import librosa
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader, random_split
from dataclasses import dataclass
from typing import List, Tuple

# ----------------------------
# Config classes
# ----------------------------

@dataclass
class FeatureConfig:
    sr: int = 22050
    n_mfcc: int = 13
    n_mels: int = 64
    hop_length: int = 512
    n_fft: int = 2048
    duration: float = 5.0  # seconds per sample

@dataclass
class LoaderConfig:
    batch_size: int = 8
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42
    num_workers: int = 2
    pin_memory: bool = True

# ----------------------------
# Utility functions
# ----------------------------

def _scan_dataset(root: str) -> List[str]:
    print(f"[DEBUG] Scanning root: {os.path.abspath(root)}")
    if not os.path.exists(root):
        print(f"[ERROR] Path does not exist: {root}")
        return []

    """Recursively scan for .wav files inside the dataset directory."""
    wav_files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".wav"):
                full_path = os.path.join(dirpath, f)
                wav_files.append(full_path)
    print(f"[dataset] Found {len(wav_files)} .wav files in {root}")
    return wav_files


def _extract_features(path: str, cfg: FeatureConfig) -> np.ndarray:
    """Load a WAV file and extract MFCC + Mel-spectrogram features."""
    try:
        y, sr = librosa.load(path, sr=cfg.sr)
        if len(y) < cfg.duration * sr:
            pad_len = int(cfg.duration * sr - len(y))
            y = np.pad(y, (0, pad_len))
        else:
            y = y[: int(cfg.duration * sr)]

        # MFCC + Mel
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=cfg.n_mfcc)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length, n_mels=cfg.n_mels)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        features = np.concatenate([mfcc, log_mel], axis=0)
        return features
    except Exception as e:
        print(f"[warn] Skipping file {path}: {e}")
        return np.zeros((cfg.n_mfcc + cfg.n_mels, int(cfg.duration * cfg.sr / cfg.hop_length)))


def _label_from_filename(filename: str) -> int:
    """Assign labels based on filename convention."""
    name = os.path.basename(filename).lower()
    # Example heuristic — customize as needed
    if "_cr" in name or "_co" in name:
        return 0  # mouth breathing / crackle
    else:
        return 1  # nasal / normal breathing


# ----------------------------
# PyTorch Dataset
# ----------------------------

class RespiratoryDataset(Dataset):
    def __init__(self, file_paths: List[str], feat_cfg: FeatureConfig):
        self.file_paths = file_paths
        self.feat_cfg = feat_cfg

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.file_paths[idx]
        x = _extract_features(path, self.feat_cfg)
        label = _label_from_filename(path)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        return x_tensor.unsqueeze(0), label


# ----------------------------
# Dataset Preparation
# ----------------------------

def prepare_datasets(root_dir: str, feat_cfg: FeatureConfig, loader_cfg: LoaderConfig, include_coswara=False):
    """
    Scan respiratory dataset (and optionally Coswara),
    preprocess, split into train/val/test datasets.
    """
    base = os.path.join(root_dir, "data", "raw")
    paths = {
        "icbhi": os.path.join(base, "Respiratory_Sound_Database", "audio_and_txt_files"),
        "coswara": os.path.join(base, "public_dataset") if include_coswara else None,
    }

    # --- Scan ICBHI files ---
    print("[dataset] Scanning Respiratory dataset...")
    print(f"[DEBUG] Scanning path for ICBHI: {os.path.abspath(paths['icbhi'])}")

    wav_files = _scan_dataset(paths["icbhi"])
    if len(wav_files) == 0:
        raise RuntimeError(f"No labeled .wav files found in {paths['icbhi']}")

    print(f"[dataset] Example files: {wav_files[:3]}")

    # Optionally include Coswara
    if include_coswara and paths["coswara"] and os.path.exists(paths["coswara"]):
        print("[dataset] Including Coswara dataset...")
        cos_files = _scan_dataset(paths["coswara"])
        wav_files.extend(cos_files)
        print(f"[dataset] Total after merge: {len(wav_files)} files")

    # --- Split datasets ---
    total = len(wav_files)
    val_n = int(total * loader_cfg.val_split)
    test_n = int(total * loader_cfg.test_split)
    train_n = total - val_n - test_n

    g = torch.Generator().manual_seed(loader_cfg.seed)
    train_files, val_files, test_files = random_split(wav_files, [train_n, val_n, test_n], generator=g)

    print(f"[dataset] Split -> Train: {train_n}, Val: {val_n}, Test: {test_n}")

    train_ds = RespiratoryDataset(train_files, feat_cfg)
    val_ds = RespiratoryDataset(val_files, feat_cfg)
    test_ds = RespiratoryDataset(test_files, feat_cfg)

    return train_ds, val_ds, test_ds


# ----------------------------
# Debug entry point
# ----------------------------

if __name__ == "__main__":
    print("[TEST] dataset.py loaded")

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    feat_cfg = FeatureConfig()
    load_cfg = LoaderConfig()

    try:
        train_ds, val_ds, test_ds = prepare_datasets(root, feat_cfg, load_cfg, include_coswara=False)
        print(f"[OK] Dataset ready. Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    except Exception as e:
        print("[ERROR]", e)
