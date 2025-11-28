# utils.py — FINAL WORKING VERSION (NO MORE BUGS)
import os
import numpy as np
import librosa
from tqdm import tqdm
from keras.utils import to_categorical

SR = 22050
DURATION = 6
N_MELS = 128
HOP_LENGTH = 512
IMG_SIZE = (128, 258)

def normalize_audio(y, target_dBFS=-23):
    rms = np.sqrt(np.mean(y**2 + 1e-8))
    current_dBFS = 20 * np.log10(rms)
    gain = 10 ** ((target_dBFS - current_dBFS) / 20)
    return np.clip(y * gain, -1.0, 1.0)

def audio_to_spectrogram(file_path, normalize_volume=True):
    try:
        y, sr = librosa.load(file_path, sr=SR)
        
        if normalize_volume:
            y = normalize_audio(y)

        target_samples = SR * DURATION
        if len(y) > target_samples:
            y = y[:target_samples]
        else:
            y = np.pad(y, (0, target_samples - len(y)), mode='constant')

        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=2048, hop_length=HOP_LENGTH,
            n_mels=N_MELS, fmax=8000, power=2.0
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        S_db = (S_db + 80) / 80  # 0 to 1

        if S_db.shape[1] != IMG_SIZE[1]:
            S_db = librosa.util.fix_length(S_db, size=IMG_SIZE[1], axis=1)

        return np.expand_dims(S_db, axis=-1).astype(np.float32)

    except Exception as e:
        print(f"Error: {file_path} → {e}")
        return None

def load_lung_dataset(base_path="datasets/Asthma Detection Dataset Version 2/Asthma Detection Dataset Version 2"):
    classes = ['asthma', 'Bronchial', 'copd', 'healthy', 'pneumonia']
    X, y = [], []

    print("Loading dataset with volume normalization...")
    for i, cls in enumerate(classes):
        folder = os.path.join(base_path, cls)
        if not os.path.exists(folder):
            print(f"Missing: {folder}")
            continue
        files = [f for f in os.listdir(folder) if f.lower().endswith('.wav')]
        print(f"{cls:10}: {len(files)} files")

        for f in tqdm(files, desc=cls):
            spec = audio_to_spectrogram(os.path.join(folder, f), normalize_volume=True)
            if spec is not None:
                X.append(spec)
                y.append(i)

    X = np.array(X)
    y = to_categorical(y, 5)
    print(f"\nLoaded {len(X)} samples → shape: {X.shape}")
    return X, y, classes