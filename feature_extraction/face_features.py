"""
Face feature extraction module.

Input:
    MediaPipe FaceLandmarker face landmarks

Output:
    Observable facial metrics

This module does NOT infer emotions.
It extracts measurable facial signals.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.spatial import distance
import cv2

# ============================================================
# Landmark indexes (MediaPipe Face Mesh)
# ============================================================

LEFT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]

RIGHT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]


MOUTH = [
    78,
    81,
    13,
    311,
    308,
    402,
    14,
    178
]


# Inner eyebrows
LEFT_BROW = 55
RIGHT_BROW = 285


# ============================================================
# Data object
# ============================================================

# @dataclass
# class FaceFeatures:

#     ear: float

#     mar: float

#     brow_distance: float

#     face_center_x: float

#     face_center_y: float

#     face_area: float

#     landmark_count: int

#     expression_energy: float

#     confidence: float


@dataclass
class FaceFeatures:

    ear: float
    mar: float
    brow_distance: float

    face_center_x: float
    face_center_y: float
    face_area: float

    landmark_count: int
    expression_energy: float

    pitch: float
    yaw: float
    roll: float

    confidence: float



# ============================================================
# Mathematical features
# ============================================================

def calculate_ear(points):

    """
    Eye Aspect Ratio

    Higher:
        eyes open

    Lower:
        eyes closed
    """

    vertical1 = distance.euclidean(
        points[1],
        points[5]
    )

    vertical2 = distance.euclidean(
        points[2],
        points[4]
    )

    horizontal = distance.euclidean(
        points[0],
        points[3]
    )

    if horizontal == 0:
        return 0.0


    return (
        vertical1 + vertical2
    ) / (2.0 * horizontal)



def calculate_mar(points):

    """
    Mouth Aspect Ratio

    Higher:
        mouth open
    """

    vertical1 = distance.euclidean(
        points[1],
        points[7]
    )

    vertical2 = distance.euclidean(
        points[2],
        points[6]
    )

    vertical3 = distance.euclidean(
        points[3],
        points[5]
    )


    horizontal = distance.euclidean(
        points[0],
        points[4]
    )


    if horizontal == 0:
        return 0.0


    return (
        vertical1 +
        vertical2 +
        vertical3
    ) / (
        3.0 *
        horizontal
    )



def calculate_expression_energy(points):

    """
    Estimate facial movement energy.

    This is NOT emotion.
    It represents amount of landmark displacement.
    """

    centered = points - np.mean(
        points,
        axis=0
    )


    magnitude = np.linalg.norm(
        centered,
        axis=1
    )


    return float(
        np.mean(magnitude)
    )

def calculate_head_pose(
    landmarks,
    frame_width,
    frame_height
):
    """
    Estimate head pose using MediaPipe landmarks.

    Returns:
        pitch, yaw, roll in degrees
    """

    image_points = np.array(
        [
            (
                landmarks[1].x * frame_width,
                landmarks[1].y * frame_height
            ),      # nose tip

            (
                landmarks[152].x * frame_width,
                landmarks[152].y * frame_height
            ),      # chin

            (
                landmarks[33].x * frame_width,
                landmarks[33].y * frame_height
            ),      # left eye

            (
                landmarks[263].x * frame_width,
                landmarks[263].y * frame_height
            ),      # right eye

            (
                landmarks[61].x * frame_width,
                landmarks[61].y * frame_height
            ),      # left mouth

            (
                landmarks[291].x * frame_width,
                landmarks[291].y * frame_height
            ),      # right mouth
        ],
        dtype=np.float64
    )


    model_points = np.array(
        [
            (0.0, 0.0, 0.0),          # nose
            (0.0, -63.6, -12.5),      # chin
            (-43.3, 32.7, -26.0),     # left eye
            (43.3, 32.7, -26.0),      # right eye
            (-28.9, -28.9, -24.1),    # left mouth
            (28.9, -28.9, -24.1),    # right mouth
        ],
        dtype=np.float64
    )


    focal_length = frame_width

    center = (
        frame_width / 2,
        frame_height / 2
    )


    camera_matrix = np.array(
        [
            [
                focal_length,
                0,
                center[0]
            ],
            [
                0,
                focal_length,
                center[1]
            ],
            [
                0,
                0,
                1
            ],
        ],
        dtype=np.float64
    )


    dist_coeffs = np.zeros(
        (4,1)
    )


    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )


    if not success:
        return 0.0, 0.0, 0.0


    rotation_matrix, _ = cv2.Rodrigues(
        rotation_vector
    )


    pose_matrix = np.hstack(
        (
            rotation_matrix,
            translation_vector
        )
    )


    # _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(
    #     pose_matrix
    # )


    # pitch = float(euler[0])
    # yaw = float(euler[1])
    # roll = float(euler[2])


    # return pitch, yaw, roll

    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(
        pose_matrix
    )

    pitch = float(euler[0, 0])
    yaw = float(euler[1, 0])
    roll = float(euler[2, 0])

    return pitch, yaw, roll





# ============================================================
# Extractor
# ============================================================


class FaceFeatureExtractor:


    def __init__(self):

        self.previous_landmarks = None



    def extract(
        self,
        face_landmarks,
        frame_width,
        frame_height
    ):


        points = np.array(
            [
                [
                    landmark.x * frame_width,
                    landmark.y * frame_height
                ]

                for landmark in face_landmarks
            ]
        )



        left_eye_points = points[
            LEFT_EYE
        ]

        right_eye_points = points[
            RIGHT_EYE
        ]


        mouth_points = points[
            MOUTH
        ]



        ear = (
            calculate_ear(left_eye_points)
            +
            calculate_ear(right_eye_points)
        ) / 2



        mar = calculate_mar(
            mouth_points
        )



        brow_distance = (
            distance.euclidean(
                points[LEFT_BROW],
                points[RIGHT_BROW]
            )
            /
            frame_width
        )



        center = np.mean(
            points,
            axis=0
        )



        min_xy = np.min(
            points,
            axis=0
        )

        max_xy = np.max(
            points,
            axis=0
        )


        face_area = (
            (max_xy[0]-min_xy[0])
            *
            (max_xy[1]-min_xy[1])
        ) / (
            frame_width *
            frame_height
        )



        expression_energy = (
            calculate_expression_energy(
                points
            )
        )


        pitch, yaw, roll = calculate_head_pose(
            face_landmarks,
            frame_width,
            frame_height
        )



        return FaceFeatures(

            ear=float(ear),

            mar=float(mar),

            brow_distance=float(
                brow_distance
            ),

            face_center_x=float(
                center[0]/frame_width
            ),

            face_center_y=float(
                center[1]/frame_height
            ),

            face_area=float(
                face_area
            ),

            landmark_count=len(
                points
            ),

            expression_energy=float(
                expression_energy
            ),

            confidence=1.0,

            pitch=pitch,

            yaw=yaw,

            roll=roll
        )