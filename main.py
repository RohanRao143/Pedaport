"""
main.py

Adaptive Learning Observable Metrics Pipeline

Pipeline:

Camera
 |
MediaPipe Face Landmarker
 |
FaceFeatureExtractor
 |
BehaviorFeatureExtractor
 |
AnalyticsEngine
 |
CSV Logger


Output:

logs/metrics.csv

"""

import os
import cv2
import time
import numpy as np


from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp


from pretrained_integration import CognitiveInference


from feature_extraction.face_features import (
    FaceFeatureExtractor
)

from feature_extraction.behavior_features import (
    BehaviorFeatureExtractor
)


from analytics.analytics_engine import (
    AnalyticsEngine,
    CSVLogger
)



# ============================================================
# Model setup
# ============================================================


MODEL_PATH = (
    "face_landmarker.task"
)



def create_detector():


    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )


    options = vision.FaceLandmarkerOptions(

        base_options=base_options,

        running_mode=
        vision.RunningMode.VIDEO,

        num_faces=1,

        output_face_blendshapes=True,

        output_facial_transformation_matrixes=True

    )


    return vision.FaceLandmarker.create_from_options(
        options
    )

def get_derived_scores(summary: dict) -> dict:
    # 1. Normalize attention metric (converts 0.2 to 20.0)
    attention = summary.get("attention_score", 0) * 100
    
    # 2. Extract existing metrics safely
    focus = summary.get("focus_score", 0)
    fatigue = summary.get("fatigue_score", 0)
    gaze_stability = summary.get("gaze_stability", 0)
    interaction = summary.get("interaction_score", 0)
    face_visibility = summary.get("face_visible_pct", 0)
    
    # 3. Create a composite 'head_motion' metric from pose variance
    # Uses the average standard deviation of pitch, yaw, and roll
    p_std = summary.get("pitch_std", 0)
    y_std = summary.get("yaw_std", 0)
    r_std = summary.get("roll_std", 0)
    head_motion = (p_std + y_std + r_std) / 3

    # 4. Compute derived states based on adjusted criteria
    derived = {
        "Focused": (attention > 80 and focus > 75 and fatigue < 30),
        
        "Reading": (attention > 70 and gaze_stability > 80 and interaction < 20),
        
        "Distracted": (attention < 40 and head_motion > 60) or (summary.get("screen_attention_pct", 0) == 0),
        
        "Fatigued": (fatigue > 70) or (summary.get("yawn_count", 0) > 0),
        
        "Needs Break": (fatigue > 80 and attention < 30),
        
        "Highly Engaged": (attention > 90 and focus > 85 and interaction > 40),
        
        "Low Confidence": (face_visibility < 40)
    }
    
    return derived


def get_derived_scores_v2(summary: dict) -> dict:

    print(summary)
    # Scale normalization for attention (0.2 -> 20.0)
    attention = summary.get("attention_score", 0.0) * 100
    
    # Exact metric mapping from schema
    focus = summary.get("focus_score", 0.0)
    fatigue = summary.get("fatigue_score", 0.0)
    gaze_stability = summary.get("gaze_stability", 0.0)
    reading = summary.get("reading_score", 0.0)
    interaction = summary.get("interaction_score", 0.0)
    screen_attention = summary.get("screen_attention_pct", 0.0)
    yaw_std = summary.get("yaw_std", 0.0)
    yawn_count = summary.get("yawn_count", 0)
    face_visibility = summary.get("face_visible_pct", 0.0)

    # Derived states strictly matching the adjusted table criteria
    derived = {
        "Focused": (attention > 80 and focus > 75 and fatigue < 30),
        
        "Reading": (attention > 70 and (gaze_stability > 80 or reading > 70) and interaction < 20),
        
        # Uses explicit schema metrics instead of a fallback string
        "Distracted": (attention < 40 or screen_attention < 20 or yaw_std > 30),
        
        "Fatigued": (fatigue > 70 or yawn_count > 2),
        
        "Needs Break": (fatigue > 80 and attention < 30),
        
        "Highly Engaged": (attention > 90 and focus > 85 and interaction > 40),
        
        "Low Confidence": (face_visibility < 40.0)
    }
    
    return derived

def extract_face_crop(frame, landmarks, margin=0.25):
    """
    Crop face region from MediaPipe landmarks.

    Returns:
        BGR face image for FER+
    """

    h, w, _ = frame.shape

    xs = [
        int(lm.x * w)
        for lm in landmarks
    ]

    ys = [
        int(lm.y * h)
        for lm in landmarks
    ]

    x_min = max(min(xs), 0)
    x_max = min(max(xs), w)

    y_min = max(min(ys), 0)
    y_max = min(max(ys), h)


    face_w = x_max - x_min
    face_h = y_max - y_min


    # add margin around face
    x_pad = int(face_w * margin)
    y_pad = int(face_h * margin)


    x1 = max(x_min - x_pad, 0)
    y1 = max(y_min - y_pad, 0)

    x2 = min(x_max + x_pad, w)
    y2 = min(y_max + y_pad, h)


    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop

# ============================================================
# Main
# ============================================================


def main():

    print(
        "Starting Adaptive Learning Analytics..."
    )

    #2 from pretrained
    predictor = CognitiveInference()

    detector = create_detector()

    face_extractor = (
        FaceFeatureExtractor()
    )

    behavior_extractor = (
        BehaviorFeatureExtractor()
    )

    logger = CSVLogger()
    analytics = AnalyticsEngine(
        logger=logger
    )
    cap = cv2.VideoCapture(0)

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )
    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    frame_timestamp = 0

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(
            frame,
            1
        )

        height,width,_ = frame.shape

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        frame_timestamp += 33

        result = detector.detect_for_video(
            mp_image,
            frame_timestamp
        )

        if result.face_landmarks:
            landmarks = (
                result.face_landmarks[0]
            )

            # ------------------------------
            # FER+ face crop
            # ------------------------------
            face_crop = extract_face_crop(
                frame,
                landmarks
            )

            # ------------------------------
            # Face features
            # ------------------------------
            face_features = (
                face_extractor.extract(
                    landmarks,
                    width,
                    height
                )
            )

            # ------------------------------
            # Behavior features
            # ------------------------------
            behavior = (
                behavior_extractor.update(
                    face_features,
                    landmarks,
                    width,
                    height
                )
            )

            # ------------------------------
            # Combine
            # ------------------------------
            metrics = {
                **face_features.__dict__,
                **behavior
            }



            # ------------------------------
            # Analytics
            # ------------------------------

# 1 - Rule based

            # summary = (
            #     analytics.update(
            #         metrics
            #     )
            # )


            # # ------------------------------
            # # When 60 sec completed
            # # ------------------------------

            # if summary:

            #     derived_scores = get_derived_scores_v2(summary)

#2 - from pretrained

            # prediction = predictor.predict(
            #     face_features,
            #     behavior,
            #     landmarks,
            #     result.face_blendshapes[0] if result.face_blendshapes else None,
            # )

            prediction = predictor.predict(
                face_crop,
                face_features,
                behavior,
                landmarks,
                result.face_blendshapes[0]
                if result.face_blendshapes else None,
            )

# ==========================================================================================================================

            if prediction is not None:

                summary = analytics.update(prediction)


                # logger.write(
                #     summary
                # )

                if summary:


                    print(
                        "\n===== 60 SECOND SUMMARY ====="
                    )

                # print("\n===== AI Prediction =====")

                # for name, value in prediction.items():
                #     print(f"{name:15s}: {value:.3f}")


                    for k,v in summary.items():

                        print(
                            k,
                            ":",
                            round(v,3)
                            if isinstance(v,float)
                            else v
                        )
                    
                    # Print results
                    # for state, active in derived_scores.items():
                    #     print(f"{state}: {active}")



            # ------------------------------
            # Live overlay
            # ------------------------------


            cv2.putText(

                frame,

                f"EAR: {face_features.ear:.2f}",

                (30,40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0,255,0),

                2

            )



            cv2.putText(

                frame,

                f"MAR: {face_features.mar:.2f}",

                (30,70),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0,255,0),

                2

            )


            cv2.putText(

                frame,

                f"Looking: {behavior['looking_at_screen']}",

                (30,100),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255,255,0),

                2

            )


            cv2.putText(

                frame,

                f"Blink: {behavior['blink_count']}",

                (30,130),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255,255,255),

                2

            )



        else:


            cv2.putText(

                frame,

                "Face not detected",

                (30,50),

                cv2.FONT_HERSHEY_SIMPLEX,

                1,

                (0,0,255),

                2

            )



        cv2.imshow(

            "Adaptive Learning Analytics",

            frame

        )



        if cv2.waitKey(1) & 0xff == ord("q"):

            break



    cap.release()

    cv2.destroyAllWindows()





if __name__ == "__main__":

    main()






























































































































































































# import cv2
# import numpy as np
# import mediapipe as mp
# from scipy.spatial import distance as dist
# from sklearn.ensemble import RandomForestClassifier

# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# # --- CONSTANT LANDMARK INDEXES (MediaPipe Face Mesh) ---
# LEFT_EYE = [362, 385, 387, 263, 373, 380]
# RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]
# LEFT_BROW_INNER = 55
# RIGHT_BROW_INNER = 285

# import os
# import urllib.request

# # 1. Standardized path setup using the script's directory
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(SCRIPT_DIR, "face_landmarker.task")

# # 2. Official model URL path from Google's samples
# MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# # 3. Download handler with identity headers
# if not os.path.exists(MODEL_PATH):
#     print("Downloading face_landmarker.task model file... Please wait...")
#     try:
#         # Create a request object mimicking a standard web browser
#         req = urllib.request.Request(
#             MODEL_URL, 
#             headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
#         )
#         with urllib.request.urlopen(req) as response, open(MODEL_PATH, 'wb') as out_file:
#             out_file.write(response.read())
#         print("Download complete and file saved successfully!")
#     except Exception as e:
#         print(f"Failed to auto-download model. Error details: {e}")


# def calculate_ear(eye_landmarks):
#     """Calculate Eye Aspect Ratio to detect squinting/fatigue."""
#     # Vertical distances
#     v1 = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
#     v2 = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
#     # Horizontal distance
#     h = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
#     return (v1 + v2) / (2.0 * h) if h != 0 else 0

# def calculate_mar(mouth_landmarks):
#     """Calculate Mouth Aspect Ratio to detect yawning/lip pursing."""
#     # Vertical distances average
#     v1 = dist.euclidean(mouth_landmarks[1], mouth_landmarks[7])
#     v2 = dist.euclidean(mouth_landmarks[2], mouth_landmarks[6])
#     v3 = dist.euclidean(mouth_landmarks[3], mouth_landmarks[5])
#     # Horizontal distance
#     h = dist.euclidean(mouth_landmarks[0], mouth_landmarks[4])
#     return (v1 + v2 + v3) / (3.0 * h) if h != 0 else 0



# base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

# options = vision.FaceLandmarkerOptions(
#     base_options=base_options,
#     output_face_blendshapes=False,
#     output_facial_transformation_matrixes=False,
#     num_faces=1
# )
# detector = vision.FaceLandmarker.create_from_options(options)

# # --- INITIALIZE MEDIAPIPE & CLASSIFIER ---
# # mp_face_mesh = mp.solutions.face_mesh
# # face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# # Mock Classifier: In production, load your trained model using joblib
# # classes = ['Engaged', 'Confused', 'Frustrated', 'Bored']
# classifier = RandomForestClassifier()
# # Fitting with dummy historical landmark ratio data for initialization
# X_dummy = np.random.rand(10, 3)  # Features: [EAR, MAR, Brow_Distance]
# y_dummy = np.array(['Engaged', 'Confused', 'Frustrated', 'Bored', 'Engaged', 'Bored', 'Frustrated', 'Confused', 'Engaged', 'Engaged'])
# classifier.fit(X_dummy, y_dummy)

# # --- VIDEO CAPTURE LOOP ---
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         break

#     # Flip horizontally for natural mirror display, convert to RGB
#     frame = cv2.flip(frame, 1)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     h, w, _ = frame.shape


#     # UPDATED V2
#     # results = face_mesh.process(rgb_frame)
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

#     detection_result = detector.detect(mp_image)


#     if detection_result.face_landmarks:
#         for face_landmarks in detection_result.face_landmarks:
#             # Extract raw coordinate points
#             points = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks])
            
#             # Compute structural ratios
#             ear = (calculate_ear(points[LEFT_EYE]) + calculate_ear(points[RIGHT_EYE])) / 2.0
#             mar = calculate_mar(points[MOUTH])
            
#             # Brow Furrow: normalized distance between inner eyebrows
#             brow_dist = dist.euclidean(points[LEFT_BROW_INNER], points[RIGHT_BROW_INNER]) / float(w)

#             # Predict affective state
#             # features = np.array([[ear, mar, brow_dist]])
#             # predicted_state = classifier.predict(features)[0]
#             # prediction_proba = classifier.predict_proba(features)[0]
#             # max_confidence = round(np.max(prediction_proba) * 100, 1)

#             # # Render output overlay on stream
#             # cv2.putText(frame, f"State: {predicted_state} ({max_confidence}%)", (30, 50),
#             #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             # cv2.putText(frame, f"EAR: {round(ear,2)} | MAR: {round(mar,2)}", (30, 90),
#             #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

#             # Format feature array for machine learning model prediction
#             features = np.array([[ear, mar, brow_dist]])
#             prediction = classifier.predict(features)[0]
            
#             # Display tracking feedback onto video interface stream
#             cv2.putText(frame, f"State: {prediction}", (30, 50), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     cv2.imshow('Adaptive Learning System - Affective Input', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()







































































































































# import cv2
# import numpy as np
# import mediapipe as mp
# from scipy.spatial import distance as dist
# from sklearn.ensemble import RandomForestClassifier

# # Import the new Tasks API components
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# # --- CONSTANT LANDMARK INDEXES (MediaPipe Face Mesh / Landmarker) ---
# LEFT_EYE = [362, 385, 387, 263, 373, 380]
# RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]
# LEFT_BROW_INNER = 55
# RIGHT_BROW_INNER = 285

# def calculate_ear(eye_landmarks):
#     v1 = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
#     v2 = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
#     h = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
#     return (v1 + v2) / (2.0 * h) if h != 0 else 0

# def calculate_mar(mouth_landmarks):
#     v1 = dist.euclidean(mouth_landmarks[1], mouth_landmarks[7])
#     v2 = dist.euclidean(mouth_landmarks[2], mouth_landmarks[6])
#     v3 = dist.euclidean(mouth_landmarks[3], mouth_landmarks[5])
#     h = dist.euclidean(mouth_landmarks[0], mouth_landmarks[4])
#     return (v1 + v2 + v3) / (3.0 * h) if h != 0 else 0

# # --- INITIALIZE NEW MEDIAPIPE FACE LANDMARKER ---
# # Ensure 'face_landmarker.task' is downloaded and in your project folder
# # base_options = python.BaseOptions(model_asset_path='face_landmarker.task')


# # import os

# # # Get the directory where main.py lives, then build an absolute file path
# # SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# # MODEL_PATH = os.path.join(SCRIPT_DIR, "face_landmarker.task")

# # # Initialize your pipeline safely using the absolute path string

# base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

# options = vision.FaceLandmarkerOptions(
#     base_options=base_options,
#     output_face_blendshapes=False,
#     output_facial_transformation_matrixes=False,
#     num_faces=1
# )
# detector = vision.FaceLandmarker.create_from_options(options)


# # --- INITIALIZE MOCK CLASSIFIER ---
# classifier = RandomForestClassifier()
# X_dummy = np.random.rand(10, 3)  
# y_dummy = np.array(['Engaged', 'Confused', 'Frustrated', 'Bored', 'Engaged', 'Bored', 'Frustrated', 'Confused', 'Engaged', 'Engaged'])
# classifier.fit(X_dummy, y_dummy)

# # --- VIDEO CAPTURE LOOP ---
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         break

#     frame = cv2.flip(frame, 1)
#     h, w, _ = frame.shape
    
#     # New MediaPipe Tasks expects an mp.Image object format
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
#     # Run object inference
#     detection_result = detector.detect(mp_image)

#     # Access coordinates using the new structure
#     if detection_result.face_landmarks:
#         for face_landmarks in detection_result.face_landmarks:
#             # Extract raw coordinate points scaled to your frame size
#             points = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks])
            
#             # Use advanced list-indexing on the NumPy array 
#             ear = (calculate_ear(points[LEFT_EYE]) + calculate_ear(points[RIGHT_EYE])) / 2.0
#             mar = calculate_mar(points[MOUTH])
            
#             # Compute normalized eyebrow furrow distance
#             brow_dist = dist.euclidean(points[LEFT_BROW_INNER], points[RIGHT_BROW_INNER]) / float(w)
            
#             # Format feature array for machine learning model prediction
#             features = np.array([[ear, mar, brow_dist]])
#             prediction = classifier.predict(features)[0]
            
#             # Display tracking feedback onto video interface stream
#             cv2.putText(frame, f"State: {prediction}", (30, 50), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     cv2.imshow('Emotion & Engagement Detector', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
