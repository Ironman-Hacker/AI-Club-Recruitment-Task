\# Speech Emotion Recognition using CNN



\## Overview

CNN-based Speech Emotion Recognition system using Log-Mel Spectrograms and RAVDESS dataset.



\## Setup

pip install -r requirements.txt



\## Train

cd src

python train.py



\## Evaluate

python evaluate.py



\## Gender Bias

python gender\_bias.py



\## Predict

python predict.py ../sample\_audio/demo.wav



\## Outputs

models/emotion\_cnn.keras  

models/training\_curves.png  

Confusion matrix shown on screen



