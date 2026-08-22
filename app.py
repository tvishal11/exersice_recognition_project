import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from collections import Counter, deque
import threading
import uuid


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Exercise Coach",
    page_icon="🏋️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #777;
    font-size: 18px;
    margin-bottom: 30px;
}

.metric-card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #ddd;
    text-align: center;
    margin-bottom: 10px;
}

.metric-title {
    font-size: 16px;
    color: #777;
}

.metric-value {
    font-size: 32px;
    font-weight: 700;
}

.result-box {
    padding: 30px;
    border-radius: 20px;
    border: 1px solid #ddd;
    text-align: center;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🏋️ AI Exercise Coach</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Exercise Recognition & Rep Tracking</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load("exercise_model.pkl")
    scaler = joblib.load("scaler.pkl")

    return model, scaler


model, scaler = load_model()


# ============================================================
# MEDIAPIPE
# ============================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# ============================================================
# SAME 32 ANGLE TRIPLETS
# ============================================================

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


# ============================================================
# ANGLE CALCULATION
# ============================================================

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


# ============================================================
# EXTRACT 32 ANGLES
# ============================================================

def get_32_angles(landmarks):

    angles = []

    for a, b, c in angle_triplets:

        try:

            a_coord = [
                landmarks[a].x,
                landmarks[a].y
            ]

            b_coord = [
                landmarks[b].x,
                landmarks[b].y
            ]

            c_coord = [
                landmarks[c].x,
                landmarks[c].y
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


# ============================================================
# SHARED WORKOUT STATE
# ============================================================

class WorkoutState:

    def __init__(self):

        self.lock = threading.Lock()

        self.active = False

        self.start_time = None

        # Stores the duration after the workout is stopped
        self.final_duration = 0.0

        self.exercise_history = deque(maxlen=30)

        self.confidence_history = deque(maxlen=30)

        # Full-workout prediction statistics (not just the last 30 frames)
        self.exercise_counts = Counter()
        self.exercise_confidence_sum = Counter()

        self.current_exercise = "Waiting..."

        self.current_confidence = 0.0

        self.reps = 0

        # Rep-tracking state
        self.stage = None
        self.last_angle = None
        self.angle_history = deque(maxlen=5)
        self.last_count_time = 0.0
        self.last_exercise_for_reps = None

        self.plank_start = None

        self.plank_time = 0

        self.frames = 0


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "workout_requested" not in st.session_state:
    st.session_state.workout_requested = False


@st.cache_resource
def get_workout_state(session_id):
    return WorkoutState()


state = get_workout_state(st.session_state.session_id)


# ============================================================
# REP COUNTING
# ============================================================

def landmark_angle(landmarks, a, b, c):
    """Return an angle using MediaPipe landmark indices."""
    return calculate_angle(
        [landmarks[a].x, landmarks[a].y],
        [landmarks[b].x, landmarks[b].y],
        [landmarks[c].x, landmarks[c].y]
    )


def visible(landmarks, indices, threshold=0.45):
    """Check that the landmarks needed for a rep calculation are visible."""
    return all(landmarks[i].visibility >= threshold for i in indices)


def smooth_angle(new_angle):
    """Smooth small frame-to-frame angle noise."""
    state.angle_history.append(new_angle)
    return float(np.median(list(state.angle_history)))


def count_repetition(exercise, landmarks):
    """Exercise-specific rep counting using smoothed joint angles.

    A rep is counted only after the body reaches the start position and
    then reaches the opposite position. This prevents duplicate counts
    from noisy frames.
    """
    now = time.time()

    if state.last_exercise_for_reps != exercise:
        state.stage = None
        state.last_angle = None
        state.angle_history.clear()
        state.last_count_time = 0.0
        state.last_exercise_for_reps = exercise

    def count_once(next_stage):
        if now - state.last_count_time >= 0.55:
            state.reps += 1
            state.last_count_time = now
            state.stage = next_stage

    # BICEP CURL: either elbow can drive the rep.
    if exercise == "bicepcurl":
        if not visible(landmarks, [12, 14, 16, 11, 13, 15]):
            return
        right = landmark_angle(landmarks, 12, 14, 16)
        left = landmark_angle(landmarks, 11, 13, 15)
        angle = smooth_angle(min(right, left))
        if angle > 150:
            state.stage = "down"
        elif angle < 65 and state.stage == "down":
            count_once("up")

    # SQUAT: average knee angle, with wider thresholds for different users.
    elif exercise == "squats":
        if not visible(landmarks, [23, 25, 27, 24, 26, 28]):
            return
        right = landmark_angle(landmarks, 24, 26, 28)
        left = landmark_angle(landmarks, 23, 25, 27)
        angle = smooth_angle((right + left) / 2)
        if angle > 150:
            state.stage = "up"
        elif angle < 120 and state.stage == "up":
            count_once("down")

    # PUSH-UP: elbow angle.
    elif exercise == "pushups":
        if not visible(landmarks, [11, 13, 15, 12, 14, 16]):
            return
        right = landmark_angle(landmarks, 12, 14, 16)
        left = landmark_angle(landmarks, 11, 13, 15)
        angle = smooth_angle((right + left) / 2)
        if angle > 150:
            state.stage = "up"
        elif angle < 115 and state.stage == "up":
            count_once("down")

    # LATERAL RAISE: shoulder-to-arm angle.
    elif exercise == "lateralraises":
        if not visible(landmarks, [23, 11, 13, 24, 12, 14]):
            return
        right = landmark_angle(landmarks, 23, 11, 13)
        left = landmark_angle(landmarks, 24, 12, 14)
        angle = smooth_angle((right + left) / 2)
        if angle > 135:
            state.stage = "down"
        elif angle < 110 and state.stage == "down":
            count_once("up")

    # SHOULDER PRESS: arms move from bent/down to overhead.
    elif exercise == "shoulderpress":
        if not visible(landmarks, [23, 11, 13, 24, 12, 14]):
            return
        right = landmark_angle(landmarks, 23, 11, 13)
        left = landmark_angle(landmarks, 24, 12, 14)
        angle = smooth_angle((right + left) / 2)
        if angle < 95:
            state.stage = "down"
        elif angle > 150 and state.stage == "down":
            count_once("up")

    # FORWARD LUNGE: use the more-bent knee rather than averaging both.
    elif exercise == "forward lunges":
        if not visible(landmarks, [23, 25, 27, 24, 26, 28]):
            return
        right = landmark_angle(landmarks, 24, 26, 28)
        left = landmark_angle(landmarks, 23, 25, 27)
        angle = smooth_angle(min(right, left))
        if angle > 155:
            state.stage = "up"
        elif angle < 125 and state.stage == "up":
            count_once("down")

    # LEG RAISE: hip angle decreases when the leg is raised.
    elif exercise == "legraises":
        if not visible(landmarks, [11, 23, 25, 12, 24, 26]):
            return
        right = landmark_angle(landmarks, 12, 24, 26)
        left = landmark_angle(landmarks, 11, 23, 25)
        angle = smooth_angle(min(right, left))
        if angle > 145:
            state.stage = "down"
        elif angle < 105 and state.stage == "down":
            count_once("up")



# ============================================================
# VIDEO PROCESSOR
# ============================================================

class VideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )


    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        # ----------------------------------------------------
        # Convert to RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        results = self.pose.process(rgb)

        # ----------------------------------------------------
        # Pose detected
        # ----------------------------------------------------

        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            # -----------------------------------------------
            # Extract 32 angles
            # -----------------------------------------------

            angles = get_32_angles(landmarks)

            # -----------------------------------------------
            # Scale
            # -----------------------------------------------

            input_scaled = scaler.transform(
                [angles]
            )

            # -----------------------------------------------
            # Prediction
            # -----------------------------------------------

            prediction = model.predict(
                input_scaled
            )[0]

            probability = model.predict_proba(
                input_scaled
            ).max()

            confidence = probability * 100

            # -----------------------------------------------
            # Workout active
            # -----------------------------------------------

            with state.lock:

                if state.active:

                    state.frames += 1

                    state.exercise_history.append(
                        prediction
                    )

                    state.confidence_history.append(
                        confidence
                    )

                    # Keep statistics for the ENTIRE workout. The final
                    # result will use these instead of the last frame.
                    state.exercise_counts[prediction] += 1
                    state.exercise_confidence_sum[prediction] += confidence

                    # Stable prediction
                    counts = Counter(
                        state.exercise_history
                    )

                    stable_exercise = counts.most_common(1)[0][0]

                    state.current_exercise = stable_exercise

                    # Average confidence
                    matching_confidences = [
                        c
                        for e, c in zip(
                            state.exercise_history,
                            state.confidence_history
                        )
                        if e == stable_exercise
                    ]

                    if matching_confidences:

                        state.current_confidence = (
                            sum(matching_confidences)
                            /
                            len(matching_confidences)
                        )

                    # ---------------------------------------
                    # Plank
                    # ---------------------------------------

                    if stable_exercise == "planks":

                        if state.plank_start is None:

                            state.plank_start = time.time()

                        state.plank_time = (
                            time.time()
                            -
                            state.plank_start
                        )

                    else:

                        state.plank_start = None

                        # -----------------------------------
                        # Rep counting
                        # -----------------------------------

                        count_repetition(
                            stable_exercise,
                            landmarks
                        )

            # ------------------------------------------------
            # Draw skeleton
            # ------------------------------------------------

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # ------------------------------------------------
            # Display exercise
            # ------------------------------------------------

            if state.active:

                text = (
                    f"{state.current_exercise.upper()} "
                    f"{state.current_confidence:.1f}%"
                )

                cv2.putText(
                    image,
                    text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

                # --------------------------------------------
                # Display reps
                # --------------------------------------------

                if state.current_exercise == "planks":

                    plank_text = (
                        f"PLANK: "
                        f"{state.plank_time:.1f}s"
                    )

                    cv2.putText(
                        image,
                        plank_text,
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 255),
                        2
                    )

                else:

                    rep_text = (
                        f"REPS: {state.reps}"
                    )

                    cv2.putText(
                        image,
                        rep_text,
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 255),
                        2
                    )

        else:

            cv2.putText(
                image,
                "NO POSE DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

        return av.VideoFrame.from_ndarray(image, format="bgr24")


# ============================================================
# START / STOP CONTROLS
# ============================================================

col1, col2 = st.columns(2)

with col1:
    if st.button(
        "▶️ START WORKOUT",
        use_container_width=True,
        disabled=st.session_state.workout_requested
    ):
        with state.lock:
            state.active = True
            state.start_time = time.time()
            state.final_duration = 0.0
            state.exercise_history.clear()
            state.confidence_history.clear()
            state.exercise_counts.clear()
            state.exercise_confidence_sum.clear()
            state.current_exercise = "Detecting..."
            state.current_confidence = 0.0
            state.reps = 0
            state.stage = None
            state.last_angle = None
            state.angle_history.clear()
            state.last_count_time = 0.0
            state.last_exercise_for_reps = None
            state.plank_start = None
            state.plank_time = 0.0
            state.frames = 0

        st.session_state.workout_requested = True
        st.rerun()

with col2:
    if st.button(
        "🛑 STOP WORKOUT",
        use_container_width=True,
        disabled=not st.session_state.workout_requested
    ):
        with state.lock:
            # Save the final workout duration before marking it inactive
            if state.start_time is not None:
                state.final_duration = time.time() - state.start_time
            state.active = False

        st.session_state.workout_requested = False
        st.rerun()


# ============================================================
# LIVE CAMERA
# ============================================================

webrtc_ctx = webrtc_streamer(
    key="exercise-recognition",
    video_processor_factory=VideoProcessor,
    desired_playing_state=st.session_state.workout_requested,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "facingMode": "user"
        },
        "audio": False
    },
    async_processing=True
)


# ============================================================
# LIVE DASHBOARD
# ============================================================

run_every = 0.5 if st.session_state.workout_requested else None


@st.fragment(run_every=run_every)
def render_dashboard():

    with state.lock:
        current_exercise = state.current_exercise
        confidence = state.current_confidence
        reps = state.reps
        frames = state.frames
        active = state.active
        start_time = state.start_time
        final_duration = state.final_duration
        plank_time = state.plank_time
        exercise_counts = dict(state.exercise_counts)
        exercise_confidence_sum = dict(state.exercise_confidence_sum)

    # Final exercise = dominant exercise across the ENTIRE workout.
    final_exercise = current_exercise
    final_confidence = confidence
    if exercise_counts:
        final_exercise = max(exercise_counts, key=exercise_counts.get)
        total = exercise_counts[final_exercise]
        if total:
            final_confidence = exercise_confidence_sum[final_exercise] / total

    # Live timer while working; saved final duration after STOP
    if active and start_time:
        duration = time.time() - start_time
    else:
        duration = final_duration

    # ------------------------------------------------------------
    # LIVE STATUS
    # ------------------------------------------------------------
    # Show the live dashboard ONLY while the workout is running.
    # After STOP, the user sees only the final workout report below.
    if active:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("🏋️ Exercise", current_exercise)

        with col2:
            st.metric("🎯 Confidence", f"{confidence:.1f}%")

        with col3:
            if current_exercise == "planks":
                st.metric("⏱️ Plank", f"{plank_time:.1f}s")
            else:
                st.metric("🔥 Reps", reps)

        with col4:
            st.metric("⏱️ Duration", f"{duration:.0f}s")

    # ------------------------------------------------------------
    # FINAL RESULT
    # ------------------------------------------------------------

    if not active and frames > 0:
        st.divider()

        st.markdown(
            """
            <div class="result-box">
            <h2>🎉 Workout Complete!</h2>
            <p>Your workout has been recorded successfully.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # One final report containing all important workout information.
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("🏋️ Exercise", final_exercise.upper())

        with col2:
            if final_exercise == "planks":
                st.metric("⏱️ Hold Time", f"{plank_time:.1f}s")
            else:
                st.metric("🔥 Repetitions", reps)

        with col3:
            st.metric("🎯 Confidence", f"{final_confidence:.1f}%")

        with col4:
            st.metric("⏱️ Duration", f"{duration:.0f}s")

        st.caption(f"Frames analyzed: {frames}")


render_dashboard()


# ============================================================
# SUPPORTED EXERCISES
# ============================================================

with st.expander("🏋️ Supported Exercises"):

    exercises = [
        "Bicep Curl",
        "Forward Lunges",
        "Lateral Raises",
        "Leg Raises",
        "Planks",
        "Pushups",
        "Shoulder Press",
        "Squats"
    ]

    for exercise in exercises:

        st.write(f"• {exercise}")