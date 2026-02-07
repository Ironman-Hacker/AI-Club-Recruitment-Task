
import sys
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def extract_log_mel(wav_path, sr=22050, n_mels=128, max_len=128):
    y, sr = librosa.load(wav_path, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel)

    if log_mel.shape[1] < max_len:
        pad_width = max_len - log_mel.shape[1]
        log_mel = np.pad(log_mel, pad_width=((0, 0), (0, pad_width)))
    else:
        log_mel = log_mel[:, :max_len]

    return log_mel[..., np.newaxis]

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
