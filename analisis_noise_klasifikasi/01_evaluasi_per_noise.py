# -*- coding: utf-8 -*-
"""
SKRIP 1 -- Evaluasi performa model klasifikasi per tingkat noise.

Test set (320 sampel) terdiri dari 4 tingkat noise, masing-masing 80 sampel:
  Original, SNR 30 dB, SNR 20 dB, SNR 10 dB.
Skrip ini menghitung metrik klasifikasi multilabel untuk SETIAP tingkat noise,
sehingga terlihat perbedaan unjuk kerja model pada tiap kondisi noise vs original.

Model utama: CNN-LSTM. Model lain (CNN, LSTM, RandomForest, XGBoost) disertakan
sebagai pembanding ketahanan terhadap noise.

Output:
  tabel/metrik_per_noise_<model>.csv   (semua metrik, per tingkat noise)
  tabel/ringkasan_per_noise.csv        (F1-macro semua model x semua noise)
  figur/perbandingan_metrik_per_noise.png
  figur/f1_per_isotop_per_noise.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C

C.ensure_dirs()
np.random.seed(C.SEED)

# ---------------------------------------------------------------------------
# 1. Muat data + prediksi tersimpan (konsisten dengan hasil laporan)
# ---------------------------------------------------------------------------
data = C.load_data()
y_test, jenis, isotop = data["y_test"], data["jenis"], data["isotop"]

pred_npz = np.load(os.path.join(C.RESULTS_DIR, "test_predictions_klasifikasi.npz"),
                   allow_pickle=True)
MODELS = ["CNN-LSTM", "CNN", "LSTM", "RandomForest", "XGBoost"]
probs_all = {m: pred_npz[m] for m in MODELS}

assert np.allclose(pred_npz["y_test"], y_test), "Prediksi tersimpan tidak align dengan data!"

print("Jumlah sampel test per tingkat noise:")
for lvl in C.NOISE_ORDER:
    print(f"  {C.NOISE_LABEL[lvl]:<10s}: {(jenis == lvl).sum()}")

# ---------------------------------------------------------------------------
# 2. Hitung metrik per tingkat noise untuk tiap model
# ---------------------------------------------------------------------------
KEY_METRICS = ["f1_macro", "f1_micro", "exact_match", "hamming_loss", "roc_auc_macro"]
ringkasan = {}                      # {model: {noise: f1_macro}}

for model in MODELS:
    rows = {}
    for lvl in C.NOISE_ORDER:
        m = jenis == lvl
        met = C.compute_metrics(y_test[m], probs_all[model][m], isotop)
        rows[C.NOISE_LABEL[lvl]] = met
    df = pd.DataFrame(rows).T                      # baris = noise, kolom = metrik
    df.index.name = "Tingkat Noise"
    df.round(4).to_csv(os.path.join(C.TABLE_DIR, f"metrik_per_noise_{model}.csv"))
    ringkasan[model] = {C.NOISE_LABEL[x]: rows[C.NOISE_LABEL[x]]["f1_macro"]
                        for x in C.NOISE_ORDER}
    if model == "CNN-LSTM":
        print(f"\n=== CNN-LSTM: metrik per tingkat noise ===")
        print(df[KEY_METRICS].round(4).to_string())

# Tabel ringkasan F1-macro semua model x noise
df_ring = pd.DataFrame(ringkasan).T
df_ring = df_ring[[C.NOISE_LABEL[x] for x in C.NOISE_ORDER]]
df_ring.index.name = "Model"
df_ring.round(4).to_csv(os.path.join(C.TABLE_DIR, "ringkasan_per_noise.csv"))
print("\n=== Ringkasan F1-macro (semua model x tingkat noise) ===")
print(df_ring.round(4).to_string())

# ---------------------------------------------------------------------------
# 3. Grafik: perbandingan metrik utama CNN-LSTM antar tingkat noise
# ---------------------------------------------------------------------------
levels = [C.NOISE_LABEL[x] for x in C.NOISE_ORDER]
cnnlstm = pd.DataFrame({lvl: C.compute_metrics(y_test[jenis == x],
                        probs_all["CNN-LSTM"][jenis == x], isotop)
                        for x, lvl in zip(C.NOISE_ORDER, levels)}).T

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_metrics = ["f1_macro", "f1_micro", "exact_match", "roc_auc_macro"]
x = np.arange(len(levels)); w = 0.2
for j, met in enumerate(plot_metrics):
    axes[0].bar(x + (j - 1.5) * w, cnnlstm[met].values, w, label=met)
axes[0].set_xticks(x); axes[0].set_xticklabels(levels)
axes[0].set_ylim(0.9, 1.001)
axes[0].set_title("CNN-LSTM: Metrik per Tingkat Noise", fontweight="bold")
axes[0].set_xlabel("Tingkat Noise"); axes[0].set_ylabel("Skor")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3, axis="y")

# Garis F1-macro semua model
for model in MODELS:
    axes[1].plot(levels, df_ring.loc[model].values, "o-", label=model, lw=2)
axes[1].set_title("F1-macro vs Tingkat Noise (semua model)", fontweight="bold")
axes[1].set_xlabel("Tingkat Noise (kiri = bersih, kanan = paling berisik)")
axes[1].set_ylabel("F1-macro")
axes[1].grid(alpha=0.3); axes[1].legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(C.FIG_DIR, "perbandingan_metrik_per_noise.png"),
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Grafik: F1 per isotop x tingkat noise (CNN-LSTM)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(isotop)); w = 0.2
for j, (lvl_key, lvl) in enumerate(zip(C.NOISE_ORDER, levels)):
    m = jenis == lvl_key
    met = C.compute_metrics(y_test[m], probs_all["CNN-LSTM"][m], isotop)
    vals = [met[f"f1_{iso}"] for iso in isotop]
    ax.bar(x + (j - 1.5) * w, vals, w, label=lvl)
ax.set_xticks(x); ax.set_xticklabels(isotop)
ax.set_ylim(0.9, 1.001)
ax.set_title("CNN-LSTM: F1 per Isotop pada Tiap Tingkat Noise", fontweight="bold")
ax.set_xlabel("Radionuklida"); ax.set_ylabel("F1-score")
ax.legend(title="Tingkat Noise", fontsize=9); ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(C.FIG_DIR, "f1_per_isotop_per_noise.png"),
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"\nSelesai. Tabel -> {C.TABLE_DIR}\n         Figur -> {C.FIG_DIR}")
