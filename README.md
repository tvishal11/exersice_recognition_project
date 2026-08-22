# Exercise Recognition Project (LUSIP Internship)

This project was developed during my **LUSIP Internship** and focuses on **real-time exercise recognition using computer vision and machine learning**.  
The system automatically detects and recognizes different physical exercises **live using a webcam** based on human pose estimation.

---

## Project Overview

The goal of this project is to perform **real-time exercise recognition** using a webcam.  
When the detection script is executed, the webcam opens automatically, and the system identifies the exercise being performed in front of the camera using pose landmarks and trained ML/DL models.

### Supported Exercises
- Bicep Curls  
- Push-ups  
- Squats  
- Planks  
- Forward Lunges   
- Shoulder Press  
- Lateral Raises  
- Leg Raises  

---

## Technologies Used

- Python  
- MediaPipe (Pose Estimation)  
- OpenCV  
- NumPy, Pandas  
- Scikit-learn  
- TensorFlow / Keras  
- Jupyter Notebook  
- Git & GitHub  

---

## Project Structure
```
exercise_recognition_project/
│
├── detection.ipynb # Real-time exercise detection (webcam)
├── model.ipynb # Model training
├── dataset.ipynb # Dataset creation & preprocessing
├── 1dcnnmodel.ipynb # CNN-based model
│
├── cnn_exercise_model.h5 # Trained CNN model
├── exercise_model.pkl # Trained ML model
├── label_encoder.pkl # Label encoder
├── scaler.pkl # Feature scaler
├── scaler_cnn.pkl # CNN scaler
│
├── exercise_32angles_dataset.csv # Pose-angle dataset
├── Results/ # Output videos/images (ignored in repo)
├── exercise_videos/ # Input videos (ignored in repo)
└── README.md
```

---

## Demo

Click the image below to watch the real-time demo on YouTube:

[![Exercise Recognition Demo](https://img.youtube.com/vi/U5kGF8z4Hj8/0.jpg)](https://www.youtube.com/watch?v=U5kGF8z4Hj8)

---

## How It Works

1. The webcam captures live video frames
2. **MediaPipe Pose** extracts body landmarks in real time
3. Joint angles are calculated from the pose landmarks
4. Extracted features are scaled appropriately
5. Trained Machine Learning / CNN models predict the exercise
6. The detected exercise name is displayed as **live feedback on screen**

---

## How to Run the Project

### Requirements

- Python 3.11
- Webcam (for real-time detection)

### 1. Install required libraries
```bash
pip install -r requirements.txt
```

### 2. Run this for real-time exercise detection
```bash
python detection.py
```

Alternatively, run `detection.ipynb` in Jupyter Notebook.

The webcam will open automatically and display the detected exercise in real time.

---

## Results

- The system successfully recognizes multiple exercises in real time

- Works using both:

  - Classical Machine Learning models

  - CNN-based Deep Learning model

- Provides live feedback while the exercise is being performed

(Result videos and images are generated locally and not pushed to GitHub.)

---

## Internship Details

- Internship Program: LUSIP

- Institution: LNMIIT, Jaipur

- Domain: Machine Learning & Computer Vision

- Project Type: Real-time Pose-based Exercise Recognition

---

## Author

Vishal Tiwari, B.Tech (CSE), 
GLA University, Mathura

---

## Future Improvements

- Exercise repetition counting

- Confidence score display

- Performance optimization for low-end devices

- Web or mobile application deployment

- Adding more exercise categories

---
