# predict.py — FINAL WORKING VERSION (with volume normalization)
import os
os.makedirs("predictions", exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from keras.models import load_model
from utils import audio_to_spectrogram, IMG_SIZE  # ← uses the fixed one with normalize_volume=True

# Load model and class names
try:
    model = load_model("models/best_model.keras")
    class_names = joblib.load("models/class_names.pkl")
    print("Model loaded successfully!")
except Exception as e:
    print(f"ERROR: Could not load model → {e}")
    print("Run: python train.py first!")
    exit()

def predict_lung_sound(file_path):
    if not os.path.exists(file_path):
        print(f"\nFILE NOT FOUND: {file_path}")
        print("Check the path and filename!")
        return

    print(f"Analyzing: {os.path.basename(file_path)}")
    
    # THIS IS THE KEY: normalize volume exactly like during training!
    spec = audio_to_spectrogram(file_path, normalize_volume=True)
    
    if spec is None or spec.shape != (128, 258, 1):
        print("Failed to create spectrogram.")
        return

    # Predict
    spec_batch = np.expand_dims(spec, axis=0)  # (1, 128, 258, 1)
    pred = model.predict(spec_batch, verbose=0)[0]
    idx = np.argmax(pred)
    confidence = pred[idx]

    # Plot result
    plt.figure(figsize=(15, 6))
    
    # Spectrogram
    plt.subplot(1, 2, 1)
    plt.imshow(spec[:, :, 0], aspect='auto', origin='lower', cmap='magma')
    plt.title("Mel Spectrogram (Volume Normalized)", fontsize=14, pad=20)
    plt.axis('off')

    # Bar chart
    plt.subplot(1, 2, 2)
    bars = plt.barh(class_names, pred, color='lightgray', edgecolor='black', height=0.6)
    bars[idx].set_color('#ff4444')
    bars[idx].set_edgecolor('black')
    
    for i, v in enumerate(pred):
        plt.text(v + 0.01, i, f"{v:.1%}", va='center', fontweight='bold')
    
    plt.xlabel("Confidence", fontsize=12)
    plt.title(f"PREDICTION: {class_names[idx].upper()}\nConfidence: {confidence:.1%}", 
              fontsize=18, color='#ff4444', fontweight='bold', pad=20)
    plt.xlim(0, 1.05)
    plt.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    save_path = "predictions/last_result.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # Final output
    print("\n" + "="*70)
    print(f"       FINAL DIAGNOSIS → {class_names[idx].upper()}")
    print(f"       CONFIDENCE       → {confidence:.2%}")
    print("="*70)
    print(f"   Spectrogram + Result saved → {save_path}")
    print("="*70)


# ——— EASY TESTING ———
if __name__ == "__main__":
    # Change this to any .wav file you want to test!
    test_files = [
        "testing_audio_data/asthmatic_breathing.wav",
        "testing_audio_data/normal_breath_sounds.wav",
        "testing_audio_data/pneumonia_sample.wav",
        # Add more files here!
    ]

    for file in test_files:
        if os.path.exists(file):
            predict_lung_sound(file)
            print("\n")
        else:
            print(f"Not found: {file}\n")