import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, spectrogram
from scipy.ndimage import gaussian_filter


# =========================
# 1. Load data
# =========================
file_path = r"D:\상훈\TI spinal cord_EMG power spectrum\정리본\Flexion_2_4hz(30-36).xlsx"   # 파일명 수정

df = pd.read_excel(file_path, skiprows=4, header=None)

time = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
emg = pd.to_numeric(df.iloc[:, 2], errors="coerce").to_numpy()

valid = np.isfinite(time) & np.isfinite(emg)
time = time[valid]
emg = emg[valid]


# =========================
# 2. Use data from 7 s onward
# =========================
start_time = 30
end_time = 36
idx = (time >= start_time) & (time <=end_time)
time = time[idx]
emg = emg[idx]


# =========================
# 3. Sampling rate
# =========================
fs = 1 / np.mean(np.diff(time))
print(f"Sampling rate: {fs:.1f} Hz")


# =========================
# 4. EMG envelope extraction
# =========================
emg_centered = emg - np.mean(emg)

# Rectification
rectified = np.abs(emg_centered)

# Low-pass filter for envelope
# TI envelope frequency가 1–4 Hz라면 20 Hz cutoff면 충분
cutoff = 20  # Hz
b, a = butter(4, cutoff / (fs / 2), btype="low")
envelope = filtfilt(b, a, rectified)


# =========================
# 5. Spectrogram
# =========================
# 2초 window: low-frequency resolution을 좋게 하기 위함
nperseg = int(fs * 2)
noverlap = int(fs * 1.8)

f, t_spec, Sxx = spectrogram(
    envelope - np.mean(envelope),
    fs=fs,
    nperseg=nperseg,
    noverlap=noverlap,
    scaling="density",
    mode="psd"
)


# =========================
# 6. Frequency range
# =========================
freq_min = 0
freq_max = 10

freq_mask = (f >= freq_min) & (f <= freq_max)

f_plot = f[freq_mask]
Sxx_plot = Sxx[freq_mask, :]


# =========================
# 7. Convert to dB and smooth
# =========================
Sxx_db = 10 * np.log10(Sxx_plot + np.finfo(float).eps)

# sigma 조절:
# 작게: 더 raw하게
# 크게: 더 부드럽게
Sxx_smooth = gaussian_filter(Sxx_db, sigma=(0.7, 0.7))


# =========================
# 8. Plot heatmap
# =========================
plt.figure(figsize=(7, 4.5))

plt.imshow(
    Sxx_smooth,
    aspect="auto",
    origin="lower",
    extent=[
        start_time + t_spec[0],
        start_time + t_spec[-1],
        f_plot[0],
        f_plot[-1]
    ],
    cmap='coolwarm',
    vmin=10,
    vmax=32

)

# stimulation frequency 표시
#stim_freq = 4  # 1, 2, 4 Hz에 맞게 수정
#plt.axhline(stim_freq, linestyle="--", linewidth=1.3)
#plt.text(
#    start_time + t_spec[-1] - 0.1,
#    stim_freq + 0.25,
#    f"{stim_freq} Hz",
#    ha="right",
#    va="bottom"
#)

plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Smoothed Time–Frequency Heatmap of EMG Envelope")

cbar = plt.colorbar()
cbar.set_label("Power (dB)")

plt.tight_layout()
plt.savefig("EMG_smoothed_time_frequency_heatmap.png", dpi=300)
plt.show()