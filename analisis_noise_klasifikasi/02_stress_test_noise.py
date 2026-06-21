# -*- coding: utf-8 -*-
"""
SKRIP 2 -- Stress-test ketahanan noise (cari batas SNR yang masih bisa digunakan).

Data test hanya berisi noise 10/20/30 dB, dan pada rentang itu model CNN-LSTM
TIDAK menurun sama sekali (lihat skrip 1). Untuk menjawab "seberapa jauh noise
masih bisa digunakan", skrip ini mengambil 80 spektrum Original dari test set lalu
MENYUNTIKKAN AWGN pada rentang SNR yang jauh lebih lebar (mis. 30 dB hingga -10 dB)
memakai fungsi noise yang SAMA PERSIS dengan augmentasi (common.add_gaussian_noise).

Untuk tiap nilai SNR, beberapa realisasi noise acak dirata-rata (REPEAT) agar
kurva tidak bergantung pada satu undian noise. Model yang sudah dilatih kemudian
diuji, dan dicari ambang SNR di mana F1-macro turun di bawah ambang batas.

Output:
  tabel/stress_test_snr.csv             (metrik vs SNR, semua model)
  tabel/ambang_toleransi_noise.csv      (SNR batas per model & kriteria)
  figur/kurva_f1_vs_snr.png
  figur/contoh_spektrum_berbagai_snr.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C

C.ensure_dirs()

# Sweep SNR: dari sangat bersih (40 dB) sampai sangat berisik (-10 dB)
SNR_SWEEP = [40, 35, 30, 25, 20, 15, 12, 10, 8, 6, 4, 2, 0, -2, -4, -6, -8, -10]
REPEAT    = 20            # realisasi noise acak per nilai SNR (dirata-rata)
MODELS    = ["CNN-LSTM", "CNN", "LSTM", "RandomForest", "XGBoost"]
# Ambang penilaian "masih bisa digunakan"
THRESH = {"sangat baik (>=0.95)": 0.95, "baik (>=0.90)": 0.90, "cukup (>=0.80)": 0.80}

# ---------------------------------------------------------------------------
# 1. Ambil spektrum Original dari test set (label & spektrum bersih)
# ---------------------------------------------------------------------------
data = C.load_data()
isotop = data["isotop"]
mask_orig = data["jenis"] == "Original"
X_clean = data["X_test"][mask_orig].astype(np.float64)   # (80, 1024), max=1
y_clean = data["y_test"][mask_orig]
print(f"Spektrum Original untuk stress-test: {X_clean.shape}")

best_params = C.load_best_params()
loaded = {}
for m in MODELS:
    loaded[m] = (C.load_torch_model(m, best_params) if m in C.TORCH_ARCH
                 else C.load_sklearn_model(m))


def predict(model_name, X):
    X32 = X.astype(np.float32)
    if model_name in C.TORCH_ARCH:
        return C.predict_torch(loaded[model_name], X32)
    return C.predict_sklearn(loaded[model_name], X32)


# ---------------------------------------------------------------------------
# 2. Sweep SNR x REPEAT, hitung metrik (rata-rata + std antar realisasi)
# ---------------------------------------------------------------------------
records = []
for snr in SNR_SWEEP:
    per_model_runs = {m: [] for m in MODELS}
    for r in range(REPEAT):
        rng = np.random.RandomState(1000 + r)        # reproducible
        X_noisy = np.stack([C.add_gaussian_noise(X_clean[i], snr, rng)
                            for i in range(len(X_clean))])
        for m in MODELS:
            met = C.compute_metrics(y_clean, predict(m, X_noisy), isotop)
            per_model_runs[m].append((met["f1_macro"], met["exact_match"],
                                      met["roc_auc_macro"], met["hamming_loss"]))
    for m in MODELS:
        arr = np.array(per_model_runs[m])
        records.append({
            "model": m, "snr_db": snr,
            "f1_macro": arr[:, 0].mean(), "f1_macro_std": arr[:, 0].std(),
            "exact_match": arr[:, 1].mean(),
            "roc_auc_macro": arr[:, 2].mean(),
            "hamming_loss": arr[:, 3].mean(),
        })
    cl = [d for d in records if d["snr_db"] == snr and d["model"] == "CNN-LSTM"][0]
    print(f"SNR {snr:>4} dB | CNN-LSTM F1-macro = {cl['f1_macro']:.4f} "
          f"(+/- {cl['f1_macro_std']:.4f})  ExactMatch = {cl['exact_match']:.4f}")

df = pd.DataFrame(records)
df.round(4).to_csv(os.path.join(C.TABLE_DIR, "stress_test_snr.csv"), index=False)

# ---------------------------------------------------------------------------
# 3. Tentukan ambang toleransi noise (SNR terendah yang masih >= kriteria)
# ---------------------------------------------------------------------------
amb_rows = []
for m in MODELS:
    sub = df[df["model"] == m].sort_values("snr_db")     # ascending SNR
    for nama, thr in THRESH.items():
        ok = sub[sub["f1_macro"] >= thr]
        snr_min = ok["snr_db"].min() if len(ok) else np.nan
        amb_rows.append({"model": m, "kriteria": nama,
                         "SNR_minimum_dB": snr_min})
df_amb = pd.DataFrame(amb_rows)
df_amb.to_csv(os.path.join(C.TABLE_DIR, "ambang_toleransi_noise.csv"), index=False)
print("\n=== Ambang toleransi noise (SNR minimum yang masih memenuhi kriteria) ===")
print(df_amb.to_string(index=False))

# ---------------------------------------------------------------------------
# 4. Grafik kurva F1-macro vs SNR
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
sub = df[df["model"] == "CNN-LSTM"].sort_values("snr_db")
ax.plot(sub["snr_db"], sub["f1_macro"], "o-", lw=2, color="tab:blue", label="CNN-LSTM")
ax.fill_between(sub["snr_db"],
                sub["f1_macro"] - sub["f1_macro_std"],
                sub["f1_macro"] + sub["f1_macro_std"],
                alpha=0.15, color="tab:blue")
# Tandai rentang noise yang ada di dataset asli (10-30 dB)
ax.axvspan(10, 30, color="green", alpha=0.07)
ax.text(20, 0.45, "rentang noise\ndataset (10-30 dB)", ha="center",
        fontsize=9, color="green")
ax.invert_xaxis()       # kiri = SNR tinggi (bersih), kanan = SNR rendah (berisik)
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("F1-macro")
ax.set_title("Ketahanan Model CNN-LSTM terhadap Noise (stress-test AWGN)",
             fontweight="bold")
ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(C.FIG_DIR, "kurva_f1_vs_snr.png"),
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Grafik: contoh spektrum yang sama pada berbagai SNR (ilustrasi noise)
# ---------------------------------------------------------------------------
rng = np.random.RandomState(0)
demo_snr = [30, 20, 10, 0, -10]
idx = 0
fig, axes = plt.subplots(len(demo_snr), 1, figsize=(11, 10), sharex=True)
for ax, snr in zip(axes, demo_snr):
    noisy = C.add_gaussian_noise(X_clean[idx], snr, rng)
    ax.plot(X_clean[idx], color="black", lw=1.0, alpha=0.5, label="Original")
    ax.plot(noisy, color="tab:red", lw=0.8, alpha=0.8, label=f"SNR {snr} dB")
    ax.set_ylabel("Counts\n(norm.)"); ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)
axes[-1].set_xlabel("No. Kanal")
fig.suptitle("Contoh Spektrum Co-60 pada Berbagai Tingkat SNR", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(C.FIG_DIR, "contoh_spektrum_berbagai_snr.png"),
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"\nSelesai. Tabel -> {C.TABLE_DIR}\n         Figur -> {C.FIG_DIR}")
