💤 Mouth Breath Sense
A Hybrid CNN + DSP System for Detecting Mouth vs Nose Breathing Using Snoring Audio
🔍 Overview

Mouth Breath Sense is a hybrid audio-analysis system that identifies whether a person is breathing through their mouth or nose during sleep by analyzing snoring audio signals.

The system combines:

Convolutional Neural Networks (CNNs) for snore-segment detection

Digital Signal Processing (DSP) for feature extraction

Pitch-based classification using a validated threshold of 75 Hz

An interactive Gradio UI for easy audio upload and visualization

This project is developed as part of an M.Tech Final Year Dissertation.

🎯 Key Features

Extracts acoustic features: RMS, ZCR, Spectral Centroid, MFCCs, Pitch

CNN-based snore detection using Mel-spectrograms

Pitch-based rule (Avg f₀ > 75 Hz ⇒ Mouth Breathing)

Waveform + Mel-spectrogram visualization

Lightweight & explainable classification pipeline

Interactive Gradio demo interface

🏗 System Architecture
Audio Input → Preprocessing → CNN Snore Detection → Feature Extraction (Librosa)
           → Pitch Extraction → Threshold-Based Classification → Visualization (Gradio)

🧠 Model Details

Input: Mel-spectrograms of snoring audio

CNN Layers:

Conv2D + ReLU

MaxPooling2D

Batch Normalization

Dropout

Dense layers for snore-pattern learning

Output: Snore / Non-snore detection used to refine pitch analysis

Pitch threshold used for breathing classification: 75 Hz

75 Hz → Mouth Breathing

< 75 Hz → Nose Breathing

🛠 Tech Stack

Python 3.9+

Librosa – Audio processing

NumPy / Matplotlib – Feature computation + plotting

TensorFlow / PyTorch – CNN training

Gradio – UI for audio upload and results display

Jupyter Notebook / Google Colab – Development environment

🚀 How to Run

Clone the repository:

git clone https://github.com/<your-username>/Mouth-Breath-Sense.git
cd Mouth-Breath-Sense


Install dependencies:

pip install -r requirements.txt


Launch the Gradio interface:

python app.py


Upload any .wav snoring audio file

View:

Extracted features

Waveform plot

Mel-spectrogram

Breathing classification (Mouth/Nose)

📦 Dataset Availability

The dataset used in this project cannot be uploaded publicly due to file size and licensing restrictions.

📌 If you need the dataset for academic or research purposes, please email me:
👉 <aaronshoby319@gmail.com>

I will share the dataset upon request.

📊 Results Summary

Mouth snoring shows higher pitch & broader frequency spread

Nasal snoring concentrates energy below 75 Hz

CNN improves snore-segment reliability before pitch analysis

End-to-end processing takes only a few seconds per file


🧩 Project Structure
├── app.py                    # Main Gradio application
├── model/                    # CNN model files
├── notebooks/                # Development notebooks
├── utils/                    # Preprocessing & feature extraction scripts
├── requirements.txt
├── README.md

🤝 Contributing

Contributions, improvements, or suggestions are welcome.
Feel free to open issues or submit pull requests.
