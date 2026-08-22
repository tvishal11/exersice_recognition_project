import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("exercise_model.pkl")
scaler = joblib.load("scaler.pkl")

# Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Same 32 triplets
angle_triplets = [
    (11, 13, 15), (13, 11, 23), (11, 23, 25), (23, 25, 27), (25, 27, 31),
    (15, 13, 11), (27, 25, 23), (23, 11, 13), (13, 15, 17), (11, 13, 15),
    (12, 14, 16), (14, 12, 24), (12, 24, 26), (24, 26, 28), (26, 28, 32),
    (16, 14, 12), (28, 26, 24), (24, 12, 14), (14, 16, 18), (12, 14, 16),
    (11, 12, 0), (23, 24, 0), (0, 11, 23), (0, 12, 24),
    (15, 17, 19), (16, 18, 20),
    (0, 23, 24), (0, 11, 12), (0, 13, 14),
    (27, 31, 29), (28, 32, 30),
    (23, 24, 26)
]


def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def get_32_angles(landmarks):
    angles = []
    for a, b, c in angle_triplets:
        try:    
            a_coord = [landmarks[a].x, landmarks[a].y]
            b_coord = [landmarks[b].x, landmarks[b].y]
            c_coord = [landmarks[c].x, landmarks[c].y]
            angle = calculate_angle(a_coord, b_coord, c_coord)
        except:
            angle = 0.0
        angles.append(angle)
    return angles

# Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        angles = get_32_angles(lm)
        input_scaled = scaler.transform([angles])
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled).max()

        cv2.putText(image, f"{prediction} ({prob*100:.1f}%)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Exercise Detection", image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





