
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.features import extract_log_mel

EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def main(wav_path):
    model = load_model("emotion_cnn.keras")

    features = extract_log_mel(wav_path)
    features = np.expand_dims(features, axis=0)

    preds = model.predict(features)[0]
    idx = np.argmax(preds)

    print(f"Emotion: {EMOTIONS[idx]}")
    print(f"Confidence: {preds[idx] * 100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <audio.wav>")
        sys.exit(1)

    main(sys.argv[1])
