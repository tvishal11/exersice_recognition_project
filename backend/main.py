from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Same 32 angle triplets used during training
angle_triplets = [

    (11, 13, 15),
    (13, 11, 23),
    (11, 23, 25),
    (23, 25, 27),
    (25, 27, 31),

    (15, 13, 11),
    (27, 25, 23),
    (23, 11, 13),
    (13, 15, 17),
    (11, 13, 15),

    (12, 14, 16),
    (14, 12, 24),
    (12, 24, 26),
    (24, 26, 28),
    (26, 28, 32),

    (16, 14, 12),
    (28, 26, 24),
    (24, 12, 14),
    (14, 16, 18),
    (12, 14, 16),

    (11, 12, 0),
    (23, 24, 0),
    (0, 11, 23),
    (0, 12, 24),

    (15, 17, 19),
    (16, 18, 20),

    (0, 23, 24),
    (0, 11, 12),
    (0, 13, 14),

    (27, 31, 29),
    (28, 32, 30),

    (23, 24, 26)
]


def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = (
        np.arctan2(c[1] - b[1], c[0] - b[0])
        -
        np.arctan2(a[1] - b[1], a[0] - b[0])
    )

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


def get_32_angles(landmarks):

    angles = []

    for a, b, c in angle_triplets:

        try:
            a_coord = [
                landmarks[a]["x"],
                landmarks[a]["y"]
            ]

            b_coord = [
                landmarks[b]["x"],
                landmarks[b]["y"]
            ]

            c_coord = [
                landmarks[c]["x"],
                landmarks[c]["y"]
            ]

            angle = calculate_angle(
                a_coord,
                b_coord,
                c_coord
            )

        except Exception:
            angle = 0.0

        angles.append(angle)

    return angles


app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("backend/exercise_model.pkl")
scaler = joblib.load("backend/scaler.pkl")
label_encoder = joblib.load("backend/label_encoder.pkl")


@app.get("/")
def home():
    return {"message": "Exercise Recognition API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


class PredictionRequest(BaseModel):
    landmarks: list[dict[str, float]]


@app.post("/predict")
def predict(request: PredictionRequest):

    angles = get_32_angles(request.landmarks)

    scaled_angles = scaler.transform([angles])

    prediction = model.predict(scaled_angles)[0]

    exercise = prediction

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled_angles)[0]
        confidence = float(max(probabilities)) * 100

    return {
        "exercise": str(exercise),
        "confidence": confidence,
        "angles": angles
    }
