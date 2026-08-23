# 🏋️ AI Exercise Coach

A real-time AI-powered exercise recognition system that uses **MediaPipe Pose** and **Machine Learning** to recognize exercises from webcam input and track workout performance.

## Features

- Real-time exercise recognition
- 8 supported exercises
- Pose detection using MediaPipe
- 32 pose-angle features
- Machine-learning-based classification
- Real-time confidence score
- Automatic repetition counting
- Plank hold-duration tracking
- Start/Stop workout controls
- FastAPI backend with web frontend

## Supported Exercises

- Bicep Curl
- Forward Lunges
- Lateral Raises
- Leg Raises
- Planks
- Pushups
- Shoulder Press
- Squats

## Tech Stack

**Frontend:** HTML, CSS, JavaScript, MediaPipe  
**Backend:** Python, FastAPI, Uvicorn  
**ML/CV:** Scikit-learn, MediaPipe, OpenCV, NumPy, Pandas

## How It Works

```text
Webcam
   ↓
MediaPipe Pose Detection
   ↓
Pose Landmarks
   ↓
32 Angle Features
   ↓
Machine Learning Model
   ↓
Exercise Prediction
   ↓
Confidence + Rep/Duration Tracking
```

## Project Goal

The goal of this project is to build a real-time AI fitness assistant that can recognize exercises, provide prediction confidence, count repetitions, and track exercise duration using computer vision and machine learning.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/tvishal11/exersice_recognition_project.git
cd exersice_recognition_project
```

### 2. Create virtual environment

```bash
python -m venv venv
```

### 3. Activate it

**Windows:**

```powershell
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the backend

```bash
uvicorn backend.main:app --reload
```

### 6. Start the frontend

Open `frontend/index.html` using VS Code Live Server or another local web server.

Allow webcam access and click **Start Workout**.

## Project Structure

```text
├── backend/
│   ├── main.py
│   ├── exercise_model.pkl
│   ├── label_encoder.pkl
│   ├── scaler.pkl
│   └── scaler_cnn.pkl
│
├── frontend/
│   └── index.html
│
├── data/
├── notebooks/
├── detection.py
├── app.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Author

**Vishal Tiwari**

B.Tech Computer Science & Engineering

## License

This project is developed for educational and portfolio purposes.
