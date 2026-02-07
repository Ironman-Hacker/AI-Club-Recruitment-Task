# Speech Emotion Recognition (SER) using 2D CNNs

This project was developed as part of the AI Club Recruitment Task.  
The objective is to classify human emotions from speech audio by converting audio signals into Log-Mel Spectrograms and treating them as images for a 2D Convolutional Neural Network (CNN).

---

## Project Overview

The system is trained on the RAVDESS dataset and classifies speech into the following 8 emotion categories:

- Neutral  
- Calm  
- Happy  
- Sad  
- Angry  
- Fearful  
- Disgust  
- Surprised  

This project combines Digital Signal Processing (DSP) and Computer Vision, leveraging CNNs to learn spatial patterns from audio frequency representations.

---

## Technical Results

The model was evaluated on a stratified hold-out test set to ensure balanced representation across all emotion classes.

- Test Accuracy: 74.25% (0.7425)
- Macro F1-Score: 0.73
- Bias Analysis: Performance remains consistent across male and female speakers, indicating no significant pitch bias.

---

## Pipeline Architecture

### Preprocessing and Data Augmentation
- Silence Trimming: Removal of leading and trailing silence using librosa.effects.trim
- Feature Engineering: Conversion of raw waveforms into Log-Mel Spectrograms with uniform padding
- Data Augmentation (applied only to training data):
  - Noise Injection  
  - Pitch Shifting  
  - Time Stretching  

These steps help improve generalization and prevent overfitting on the relatively small dataset.

---

### Model Architecture (2D CNN)

The model follows a structured convolutional design:
- Multiple Conv2D blocks with BatchNormalization and MaxPooling
- Global Average Pooling to reduce parameter count and overfitting
- Dropout for regularization
- Softmax output layer for multi-class classification
- Optimizer: Adam  
- Loss Function: Categorical Cross-Entropy  

---

## Repository Structure

├── notebook.ipynb Complete EDA, preprocessing, training, and evaluation
├── emotion_cnn.keras Saved model weights (best-performing iteration)
└── predict.py Standalone script for live emotion inference

---

## Live Inference

To predict the emotion of an unseen wav file, run:

```bash
python predict.py path_to_audio.wav
Example Output:

Emotion: happy
Confidence: 81.42%
Conclusion

This project demonstrates a complete Speech Emotion Recognition pipeline, emphasizing robust evaluation beyond accuracy.
By prioritizing Macro F1-score, visual EDA, and bias analysis, the model provides a more reliable and interpretable assessment of real-world performance.
