# -*- coding: utf-8 -*-
"""
Modul bersama untuk analisis ketahanan model klasifikasi CNN-LSTM terhadap noise.

Berisi:
  - Definisi arsitektur model (CNN-LSTM, CNN, LSTM) -- disalin persis dari Model_Klasifikasi.ipynb
  - Fungsi penambah noise AWGN (disalin persis dari augmentasi_spektrum.py)
  - Fungsi metrik klasifikasi multilabel
  - Loader data, model, dan hyperparameter

Catatan penting tentang noise:
  Noise yang dipakai adalah Additive White Gaussian Noise (AWGN) dengan daya noise
  ditentukan oleh SNR (dB). Karena SNR adalah rasio daya sinyal/noise, penambahan
  noise bersifat *scale-invariant*: menambahkan AWGN pada spektrum ter-normalisasi
  menghasilkan SNR yang sama dengan menambahkannya pada spektrum mentah, sehingga
  reproduksi noise pada spektrum Original ter-normalisasi tetap setia dengan pipeline.
"""
import os
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (f1_score, hamming_loss, accuracy_score,
                             roc_auc_score)

# ---------------------------------------------------------------------------
# Konfigurasi path
# ---------------------------------------------------------------------------
BASE_DIR    = r"C:\Users\wahyu\OneDrive\TA & Skripsi\TA&Skripsi\Olah data (2)"
DATA_NPZ    = os.path.join(BASE_DIR, "DATA SPLIT.npz")
MODELS_DIR  = os.path.join(BASE_DIR, "models_klasifikasi")
RESULTS_DIR = os.path.join(BASE_DIR, "results_klasifikasi")
HP_JSON     = os.path.join(RESULTS_DIR, "best_hyperparams_klasifikasi.json")

OUT_DIR     = os.path.join(BASE_DIR, "analisis_noise_klasifikasi")
TABLE_DIR   = os.path.join(OUT_DIR, "tabel")
FIG_DIR     = os.path.join(OUT_DIR, "figur")

SEED   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Urutan tingkat noise dari paling bersih ke paling berisik
NOISE_ORDER = ["Original", "SNR_30dB", "SNR_20dB", "SNR_10dB"]
NOISE_LABEL = {"Original": "Original", "SNR_30dB": "30 dB",
               "SNR_20dB": "20 dB", "SNR_10dB": "10 dB"}


# ---------------------------------------------------------------------------
# Arsitektur model (disalin persis dari Model_Klasifikasi.ipynb)
# ---------------------------------------------------------------------------
class CNN_LSTM(nn.Module):
    def __init__(self, n_filters=32, kernel_size=5, n_conv=2, lstm_hidden=64,
                 bidirectional=True, dropout=0.3, n_outputs=4):
        super().__init__()
        convs = []
        in_ch = 1
        for i in range(n_conv):
            out_ch = n_filters * (2 ** i)
            convs += [nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                      nn.BatchNorm1d(out_ch), nn.ReLU(), nn.MaxPool1d(2)]
            in_ch = out_ch
        self.conv = nn.Sequential(*convs)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm = nn.LSTM(in_ch, lstm_hidden, batch_first=True, bidirectional=bidirectional)
        lstm_out = lstm_hidden * (2 if bidirectional else 1)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out, n_outputs)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.dropout1(x)
        x = x.transpose(1, 2)
        _, (h, _) = self.lstm(x)
        if self.lstm.bidirectional:
            h = torch.cat([h[-2], h[-1]], dim=-1)
        else:
            h = h[-1]
        h = self.dropout2(h)
        return self.fc(h)


class CNN_Only(nn.Module):
    def __init__(self, n_filters=32, kernel_size=5, n_conv=3, dropout=0.3,
                 fc_hidden=64, n_outputs=4):
        super().__init__()
        convs = []
        in_ch = 1
        for i in range(n_conv):
            out_ch = n_filters * (2 ** i)
            convs += [nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                      nn.BatchNorm1d(out_ch), nn.ReLU(), nn.MaxPool1d(2)]
            in_ch = out_ch
        self.conv = nn.Sequential(*convs)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_ch, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, n_outputs)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class LSTM_Only(nn.Module):
    def __init__(self, lstm_hidden=64, n_layers=2, bidirectional=True,
                 dropout=0.3, n_outputs=4):
        super().__init__()
        self.downsample = nn.AvgPool1d(8)
        self.lstm = nn.LSTM(1, lstm_hidden, n_layers, batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout if n_layers > 1 else 0.0)
        lstm_out = lstm_hidden * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out, n_outputs)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.downsample(x)
        x = x.transpose(1, 2)
        _, (h, _) = self.lstm(x)
        if self.lstm.bidirectional:
            h = torch.cat([h[-2], h[-1]], dim=-1)
        else:
            h = h[-1]
        h = self.dropout(h)
        return self.fc(h)


TORCH_ARCH = {"CNN-LSTM": CNN_LSTM, "CNN": CNN_Only, "LSTM": LSTM_Only}


# ---------------------------------------------------------------------------
# Noise AWGN (disalin persis dari augmentasi_spektrum.py)
# ---------------------------------------------------------------------------
def add_gaussian_noise(signal: np.ndarray, snr_db: float, rng=None) -> np.ndarray:
    """Tambahkan white Gaussian noise ke sinyal sesuai SNR (dB).

    Identik dengan fungsi pada augmentasi_spektrum.py, namun menerima generator
    acak opsional agar sweep SNR dapat direproduksi.
    """
    if rng is None:
        rng = np.random
    p_signal = np.mean(signal ** 2)
    if p_signal == 0:
        return signal.copy()
    p_noise = p_signal / (10 ** (snr_db / 10.0))
    sigma = np.sqrt(p_noise)
    noise = rng.normal(0.0, sigma, signal.shape)
    result = signal + noise
    return np.clip(result, 0.0, None)   # cacah foton tidak bisa negatif


# ---------------------------------------------------------------------------
# Metrik klasifikasi multilabel (disalin dari Model_Klasifikasi.ipynb)
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_prob, isotop, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    y_true = y_true.astype(int)
    metrics = {
        "f1_macro":     f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_micro":     f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_samples":   f1_score(y_true, y_pred, average="samples", zero_division=0),
        "exact_match":  accuracy_score(y_true, y_pred),
        "hamming_loss": hamming_loss(y_true, y_pred),
    }
    try:
        metrics["roc_auc_macro"] = roc_auc_score(y_true, y_prob, average="macro")
    except ValueError:
        metrics["roc_auc_macro"] = np.nan
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, iso in enumerate(isotop):
        metrics[f"f1_{iso}"] = f1_per[i]
    return metrics


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_data():
    d = np.load(DATA_NPZ, allow_pickle=True)
    return {
        "X_test":  d["X_test"],
        "y_test":  d["y_cls_test"],
        "jenis":   d["jenis_test"].astype(str),
        "kode":    d["kode_test"],
        "isotop":  list(d["isotop_names"]),
    }


def load_best_params():
    with open(HP_JSON) as f:
        return json.load(f)["best_params"]


def load_torch_model(name, best_params):
    """Instansiasi arsitektur dengan HP terbaik lalu muat bobot tersimpan."""
    hp = best_params[name]
    arch_hp = {k: v for k, v in hp.items() if k not in {"lr", "batch_size", "optimizer"}}
    model = TORCH_ARCH[name](**arch_hp).to(DEVICE)
    fname = {"CNN-LSTM": "cnn_lstm_cls.pt", "CNN": "cnn_cls.pt", "LSTM": "lstm_cls.pt"}[name]
    state = torch.load(os.path.join(MODELS_DIR, fname), map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def load_sklearn_model(name):
    fname = {"RandomForest": "rf_cls.pkl", "XGBoost": "xgb_cls.pkl"}[name]
    with open(os.path.join(MODELS_DIR, fname), "rb") as f:
        return pickle.load(f)


def predict_torch(model, X, batch_size=256):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(DEVICE)
            probs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(probs, axis=0)


def predict_sklearn(model, X):
    """MultiOutputClassifier -> probabilitas kelas positif per label."""
    return np.stack([m.predict_proba(X)[:, 1] for m in model.estimators_], axis=1)


def ensure_dirs():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
