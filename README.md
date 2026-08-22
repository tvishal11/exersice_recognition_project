# Exercise Recognition & Workout Tracking System

A real-time computer vision application that detects and recognizes exercises using human pose estimation and machine learning.

The application uses a webcam to continuously analyze body posture, identify the exercise being performed, count repetitions, and generate a final workout report.

## Features

- Real-time webcam-based exercise detection
- Human pose estimation using MediaPipe
- Machine-learning-based exercise classification
- Confidence score for exercise predictions
- Automatic repetition counting
- Workout duration tracking
- Final workout summary after stopping the workout
- Interactive Streamlit web interface
- Real-time video processing using WebRTC

## Supported Exercises

The current model recognizes 8 exercises:

1. Bicep Curl
2. Forward Lunges
3. Lateral Raises
4. Leg Raises
5. Planks
6. Push-ups
7. Shoulder Press
8. Squats

## How It Works

The application follows this pipeline:

````text
Webcam
  ↓
Video Frame
  ↓
MediaPipe Pose Estimation
  ↓
Body Landmark Detection
  ↓
32 Angle Features
  ↓
Feature Scaling
  ↓
Random Forest Classifier
  ↓
Exercise Prediction
  ↓
Rep Counting + Workout Tracking
  ↓
Final Workout Report

### 1. Pose Detection

MediaPipe detects human body landmarks from each webcam frame.

### 2. Feature Extraction

Body landmark coordinates are used to calculate body joint angles.

The current model uses 32 angle-based features.

### 3. Exercise Classification

The extracted features are scaled using the trained scaler and passed to a Random Forest classifier.

### 4. Repetition Counting

Exercise-specific movement logic is used to detect completed repetitions.

### 5. Workout Tracking

During the workout, the application tracks:

- Exercise name
- Repetition count
- Prediction confidence
- Workout duration

After the workout is stopped, a final workout report is displayed.

## Tech Stack

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- Random Forest

### Computer Vision

- OpenCV
- MediaPipe

### Web Application

- Streamlit
- Streamlit-WebRTC

### Data Processing

- NumPy
- Pandas

### Model Persistence

- Joblib

## Project Structure

```text
exercise_recognition_project/
│
├── app.py
├── detection.py
├── exercise_model.pkl
├── scaler.pkl
├── label_encoder.pkl
├── cnn_exercise_model.h5
├── scaler_cnn.pkl
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── exercise_32angles_dataset.csv
│
└── notebooks/
    ├── dataset.ipynb
    ├── detection.ipynb
    ├── model.ipynb
    └── 1dcnnmodel.ipynb

## Installation

### 1. Clone the repository

git clone https://github.com/tvishal11/exersice_recognition_project.git

### 2. Navigate to the project

cd exersice_recognition_project

### 3. Create a virtual environment

python -m venv venv

### 4. Activate the virtual environment

For Windows PowerShell:

venv\Scripts\Activate.ps1

### 5. Install dependencies

pip install -r requirements.txt

## Run the Application

Start the Streamlit application using:

streamlit run app.py

The application will open in your browser.

Allow camera access when prompted and start the workout.

## Workout Report

During a workout, the application performs real-time exercise recognition.

After stopping the workout, it provides a final report containing information such as:

- Detected exercise
- Number of repetitions
- Workout duration
- Prediction confidence

This allows the user to review the completed workout session.

## Machine Learning Model

The primary exercise recognition model is a Random Forest Classifier trained using pose-derived angle features.

The trained model and preprocessing objects are stored using Joblib:

- exercise_model.pkl
- scaler.pkl
- label_encoder.pkl

The project also contains a CNN-based model and its associated scaler for experimentation:

- cnn_exercise_model.h5
- scaler_cnn.pkl

## Dataset

The project uses a custom dataset containing pose-derived angle features for exercise recognition.

The dataset is available in:

data/exercise_32angles_dataset.csv

Each sample contains angle-based features extracted from human pose landmarks along with the corresponding exercise label.

## Why Pose-Based Exercise Recognition?

Instead of relying directly on raw images, the system uses body pose information and joint angles.

This makes the model focus on body movement and posture rather than image appearance.

The approach can therefore be useful for real-time fitness applications where the goal is to recognize exercises from human movement.

## Future Improvements

Possible future improvements include:

- Support for more exercises
- Improved repetition counting for all exercises
- More robust detection under different camera angles
- Workout history and user profiles
- Progress tracking
- Exercise-specific feedback
- Improved confidence stabilization
- Cloud deployment
- Mobile application integration

## Author

Vishal Tiwari

B.Tech Computer Science & Engineering

[GitHub Repository](https://github.com/tvishal11/exersice_recognition_project)

## Project Goal

The goal of this project is to build a practical computer-vision-based fitness assistant capable of recognizing exercises and tracking workout performance in real time.
````
