# Real-Time Facial Emotion Detection

A deep learning–based model that recognizes **human emotions in real time** using facial expressions.  
This project compares a **baseline CNN** with an advanced **ResEmoteNet architecture** to understand how model design impacts real-time performance.

---

## 🚀 Overview

Facial Emotion Recognition (FER) enables machines to understand human emotions from facial cues.  
This project focuses on building an emotion detection system that works not just on images, but also in **live video and real-world scenarios**.

The system classifies **seven core emotions**:
- Happy
- Sad
- Angry
- Fear
- Surprise
- Disgust
- Neutral

---

## 🔑 Key Features

- Two deep learning models:
  - **CNN** — a strong, efficient baseline
  - **ResEmoteNet** — enhanced with Residual and Squeeze-and-Excitation blocks
- Training and validation performed under **identical conditions** for fair comparison
- Real-time emotion detection using **OpenCV** and webcam/video input
- Evaluation on both **static images** and **dynamic video streams**

---

## 📁 Project Structure

```text
models/
├── cnn.py
├── resnet.py

notebooks/
├── cnn_training.ipynb
├── resnet_training.ipynb
├── exploratory_data_analysis.ipynb

scripts/
├── realtime_inference.py

data/
├── haarcascade_face.xml
├── sample_input_video.mp4

```

## ▶️ How to Run
- Install dependencies
```
pip install -r requirements.txt
```

- Run real-time emotion detection
```
python scripts/realtime_inference.py
```

## 📊 Dataset & Training

- **Dataset**: FER2013
- **Input size**: 48×48 grayscale images
- **Train / Validation split**: 80 / 20
- **Optimizer**: Adam (learning rate = 0.001)
- **Loss**: Categorical cross-entropy
- **Epochs**: 50
- **Batch size**: 64

Data augmentation techniques such as rotation, zooming, shifting, and horizontal flipping were used to improve generalization.

---

## 🏗️ Model Architectures

### 🔹 CNN (Baseline)
- Three convolutional blocks with increasing filters
- Batch normalization and dropout to reduce overfitting
- Performs consistently on static images
- Lightweight and computationally efficient

### 🔹 ResEmoteNet (Advanced)
- Built on top of CNN architecture
- Uses:
  - **Residual connections** for deeper learning
  - **Squeeze-and-Excitation blocks** to focus on important facial features
- Better suited for **real-time and video-based emotion recognition**

---

## ⚡ Real-Time Emotion Detection

- Face detection using **Haar Cascade**
- Each detected face is preprocessed and passed to the trained model
- Supports:
  - Live webcam feed
  - Pre-recorded video files
- Designed for **low-latency predictions** in real-world conditions

---


## 📈 Results Summary

| Model        | Test Accuracy | Key Strength |
|-------------|---------------|--------------|
| CNN         | ~62.6%        | Stable and efficient baseline |
| ResEmoteNet | ~61.9%        | Superior real-time performance |

**Key takeaway:**  
While the CNN performs well on static images, **ResEmoteNet handles dynamic facial expressions better**, making it more suitable for real-time applications.

---

