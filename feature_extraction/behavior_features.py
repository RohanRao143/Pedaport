"""
behavior_features.py

Observable behavior feature extraction.

Includes:
    - Blink detection
    - Yawn detection
    - Head pose estimation
    - Gaze stability
    - Face movement tracking

No cognitive/emotional inference is performed.
Only measurable facial behaviors are calculated.
"""

from collections import deque
from dataclasses import dataclass
import time
import math

import numpy as np
import cv2


# ============================================================
# Configuration
# ============================================================

EAR_THRESHOLD = 0.22

MAR_THRESHOLD = 0.60

MIN_BLINK_FRAMES = 2

MIN_YAWN_FRAMES = 8

GAZE_HISTORY = 60



# ============================================================
# Output container
# ============================================================

@dataclass
class BehaviorFeatures:

    blink_event: bool

    blink_count: int

    blink_rate: float

    avg_blink_duration: float


    yawn_event: bool

    yawn_count: int


    head_pitch: float

    head_yaw: float

    head_roll: float


    gaze_x: float

    gaze_y: float

    gaze_stability: float

    looking_at_screen: bool


    face_velocity: float

    movement_variance: float



# ============================================================
# Blink Detector
# ============================================================


class BlinkDetector:

    def __init__(self):

        self.closed_frames = 0

        self.total_blinks = 0

        self.duration_history = deque(
            maxlen=100
        )

        self.timestamps = deque(
            maxlen=200
        )


    def update(self, ear):

        blink = False


        if ear < EAR_THRESHOLD:

            self.closed_frames += 1


        else:

            if self.closed_frames >= MIN_BLINK_FRAMES:

                blink = True

                self.total_blinks += 1

                self.duration_history.append(
                    self.closed_frames
                )

                self.timestamps.append(
                    time.time()
                )


            self.closed_frames = 0



        return blink



    def metrics(self):

        now = time.time()


        recent = [
            t for t in self.timestamps
            if now - t < 60
        ]


        blink_rate = len(recent)


        avg_duration = (

            np.mean(
                self.duration_history
            )

            if self.duration_history

            else 0

        )


        return {

            "blink_count":
                self.total_blinks,

            "blink_rate":
                blink_rate,

            "avg_blink_duration":
                float(avg_duration)

        }



# ============================================================
# Yawn Detector
# ============================================================


class YawnDetector:


    def __init__(self):

        self.open_frames = 0

        self.total_yawns = 0



    def update(self, mar):

        yawn = False


        if mar > MAR_THRESHOLD:

            self.open_frames += 1


        else:

            if self.open_frames >= MIN_YAWN_FRAMES:

                self.total_yawns += 1

                yawn = True


            self.open_frames = 0



        return yawn



    def metrics(self):

        return {

            "yawn_count":
                self.total_yawns

        }




# ============================================================
# Head Pose Estimator
# ============================================================


class HeadPoseEstimator:


    def __init__(self):

        self.model_points = np.array(
            [

                (0.0,0.0,0.0),

                (0.0,-330,-65),

                (-225,170,-135),

                (225,170,-135),

                (-150,-150,-125),

                (150,-150,-125)

            ],
            dtype=np.float64
        )



        self.landmark_indexes = [

            1,
            152,
            33,
            263,
            61,
            291

        ]



    def estimate(
        self,
        landmarks,
        width,
        height
    ):


        image_points = np.array(
            [

                (
                    landmarks[i].x * width,
                    landmarks[i].y * height

                )

                for i in self.landmark_indexes

            ],
            dtype=np.float64
        )


        focal_length = width


        camera_matrix=np.array(
            [

                [
                    focal_length,
                    0,
                    width/2
                ],

                [
                    0,
                    focal_length,
                    height/2
                ],

                [
                    0,
                    0,
                    1
                ]

            ],
            dtype=np.float64
        )


        dist_coeffs=np.zeros(
            (4,1)
        )


        success, rotation_vector, _ = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs
        )


        if not success:

            return {
                "pitch":0,
                "yaw":0,
                "roll":0
            }


        rotation_matrix,_ = cv2.Rodrigues(
            rotation_vector
        )


        angles = cv2.RQDecomp3x3(
            rotation_matrix
        )[0]


        return {

            "pitch":
                float(
                    angles[0]
                ),

            "yaw":
                float(
                    angles[1]
                ),

            "roll":
                float(
                    angles[2]
                )

        }




# ============================================================
# Gaze Tracker
# ============================================================


class GazeTracker:


    def __init__(self):

        self.history = deque(
            maxlen=GAZE_HISTORY
        )


    def update(
        self,
        yaw,
        pitch
    ):


        self.history.append(
            (
                yaw,
                pitch
            )
        )


        if len(self.history) < 2:

            stability = 1.0

        else:

            variance=np.var(
                self.history
            )


            stability=max(
                0,
                1 -
                variance/100
            )



        looking = (

            abs(yaw) < 20

            and

            abs(pitch) < 15

        )


        return {

            "gaze_x":
                yaw,

            "gaze_y":
                pitch,

            "gaze_stability":
                float(
                    stability
                ),

            "looking_at_screen":
                looking

        }




# ============================================================
# Movement Tracker
# ============================================================


class MovementTracker:


    def __init__(self):

        self.previous=None

        self.history=deque(
            maxlen=120
        )



    def update(
        self,
        x,
        y
    ):


        velocity=0


        if self.previous:


            velocity=math.sqrt(

                (x-self.previous[0])**2

                +

                (y-self.previous[1])**2

            )


        self.previous=(x,y)


        self.history.append(
            velocity
        )


        variance=0


        if len(self.history)>2:

            variance=float(
                np.var(
                    self.history
                )
            )


        return {

            "face_velocity":
                float(velocity),

            "movement_variance":
                variance

        }




# ============================================================
# Unified Processor
# ============================================================


class BehaviorFeatureExtractor:


    def __init__(self):

        self.blink = BlinkDetector()

        self.yawn = YawnDetector()

        self.head = HeadPoseEstimator()

        self.gaze = GazeTracker()

        self.motion = MovementTracker()



    def update(
        self,
        face_features,
        landmarks,
        width,
        height
    ):


        blink_event = self.blink.update(
            face_features.ear
        )


        yawn_event = self.yawn.update(
            face_features.mar
        )


        head = self.head.estimate(
            landmarks,
            width,
            height
        )


        gaze = self.gaze.update(
            head["yaw"],
            head["pitch"]
        )


        movement=self.motion.update(
            face_features.face_center_x,
            face_features.face_center_y
        )


        blink=self.blink.metrics()

        yawn=self.yawn.metrics()



        return {

            **blink,

            **yawn,

            **head,

            **gaze,

            **movement,

            "blink_event":
                blink_event,

            "yawn_event":
                yawn_event

        }