# import numpy as np
# from collections import deque
# import onnxruntime as ort


# class CognitiveInference:
#     """
#     Drop-in replacement for rule-based derived scores.

#     Expected ONNX input:
#         (1, sequence_length, feature_dim)

#     Output:
#         [attention,
#          fatigue,
#          engagement,
#          reading,
#          distraction,
#          confidence,
#          confusion]
#     """

#     def __init__(
#         self,
#         model_path="models/cognitive_net.onnx",
#         sequence_length=64,
#         alpha=0.25,
#     ):

#         self.seq = deque(maxlen=sequence_length)

#         self.session = ort.InferenceSession(
#             model_path,
#             providers=["CPUExecutionProvider"]
#         )

#         self.input_name = self.session.get_inputs()[0].name

#         self.feature_dim = None

#         self.alpha = alpha
#         self.smoothed = None

#     def _build_feature_vector(
#         self,
#         face_features,
#         behavior,
#         landmarks,
#         blendshapes=None,
#     ):
#         """
#         Converts MediaPipe output into one flat vector.
#         """

#         lm = np.array(
#             [[p.x, p.y, p.z] for p in landmarks],
#             dtype=np.float32,
#         ).flatten()

#         if blendshapes is not None:
#             bs = np.array(
#                 [b.score for b in blendshapes],
#                 dtype=np.float32,
#             )
#         else:
#             bs = np.zeros(52, dtype=np.float32)

#         engineered = np.array(
#             [
#                 face_features.ear,
#                 face_features.mar,
#                 face_features.pitch,
#                 face_features.yaw,
#                 face_features.roll,

#                 float(behavior.get("blink_count", 0)),
#                 float(behavior.get("yawn_count", 0)),
#                 float(behavior.get("looking_at_screen", 0)),
#                 float(behavior.get("gaze_stability", 0)),
#             ],
#             dtype=np.float32,
#         )

#         return np.concatenate([lm, bs, engineered])

#     def predict(
#         self,
#         face_features,
#         behavior,
#         landmarks,
#         blendshapes=None,
#     ):

#         feature = self._build_feature_vector(
#             face_features,
#             behavior,
#             landmarks,
#             blendshapes,
#         )

#         if self.feature_dim is None:
#             self.feature_dim = feature.size

#         self.seq.append(feature)

#         if len(self.seq) < self.seq.maxlen:
#             return None

#         x = np.stack(self.seq).astype(np.float32)
#         x = np.expand_dims(x, 0)

#         output = self.session.run(
#             None,
#             {self.input_name: x},
#         )[0][0]

#         if self.smoothed is None:
#             self.smoothed = output
#         else:
#             self.smoothed = (
#                 self.alpha * output
#                 + (1 - self.alpha) * self.smoothed
#             )

#         scores = self.smoothed

#         return {
#             "attention": float(scores[0]),
#             "fatigue": float(scores[1]),
#             "engagement": float(scores[2]),
#             "reading": float(scores[3]),
#             "distraction": float(scores[4]),
#             "confidence": float(scores[5]),
#             "confusion": float(scores[6]),
#         }

# Camera frame
#       |
#       v
# MediaPipe Face Detector
#       |
#       +---- landmarks
#       |
#       +---- face_crop --------------+
#       |                             |
#       v                             v
# FaceFeatureExtractor          FER+ ONNX
#       |                             |
#       |                             |
#       +-------------+---------------+
#                     |
#                     v
#           CognitiveInference fusion
#                     |
#                     v
#  attention
#  fatigue
#  engagement
#  reading
#  distraction
#  confidence
#  confusion


# https://chatgpt.com/s/t_6a671df113b88191b54985fd5fcfe84b
# https://chatgpt.com/s/t_6a6721158db881918546e8877cd50f8e

# Challenge #2 :: Inaccurate Cognitive states ? How to determine cognitive state if not with heuristic basis ? learning ?

# https://chatgpt.com/s/t_6a686c1e84b8819188936350c4d82f25


import cv2
import numpy as np
from collections import deque
import onnxruntime as ort


class CognitiveInference:
    """
    FER+ powered cognitive estimation layer.

    Requires:
        emotion-ferplus-8.onnx

    Output:
        attention
        fatigue
        engagement
        reading
        distraction
        confidence
        confusion
    """

    EMOTIONS = [
        "neutral",
        "happiness",
        "surprise",
        "sadness",
        "anger",
        "disgust",
        "fear",
        "contempt",
    ]

    def __init__(
        self,
        model_path="models/emotion-ferplus-8.onnx",
        sequence_length=32,
        alpha=0.25,
    ):

        self.seq = deque(maxlen=sequence_length)

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name

        self.alpha = alpha
        self.smoothed = None


    def _emotion_features(self, face_image):
        """
        FER+ preprocessing.
        Expected:
            grayscale face crop
        """

        gray = cv2.cvtColor(
            face_image,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.resize(
            gray,
            (64,64)
        )

        x = gray.astype(np.float32)

        x = np.expand_dims(x, 0)
        x = np.expand_dims(x, 0)

        return x


    def _run_emotion(self, face_image):

        x = self._emotion_features(face_image)

        output = self.session.run(
            None,
            {
                self.input_name:x
            }
        )[0][0]


        # convert logits to probabilities
        exp = np.exp(output - np.max(output))
        probs = exp / np.sum(exp)

        return {
            name: float(probs[i])
            for i,name in enumerate(self.EMOTIONS)
        }

# 1 and # 2
    # def predict(
    #     self,
    #     face_image,
    #     face_features,
    #     behavior,
    #     landmarks=None,
    #     blendshapes=None,
    # ):

    #     emotions = self._run_emotion(face_image)


    #     ear = float(face_features.ear)
    #     mar = float(face_features.mar)

    #     pitch = float(face_features.pitch)
    #     yaw = abs(float(face_features.yaw))


    #     looking = float(
    #         behavior.get(
    #             "looking_at_screen",
    #             0
    #         )
    #     )

    #     gaze = float(
    #         behavior.get(
    #             "gaze_stability",
    #             0
    #         )
    #     )


    #     blink = float(
    #         behavior.get(
    #             "blink_count",
    #             0
    #         )
    #     )

    #     yawn = float(
    #         behavior.get(
    #             "yawn_count",
    #             0
    #         )
    #     )

    #     # print(yawn, blink, ear, emotions["sadness"], emotions,            0.35 * yawn +
    #     #     0.25 * blink +
    #     #     0.20 * (1-ear) +
    #     #     0.20 * emotions["sadness"])


    #     #
    #     # Fusion model
    #     #

    #     attention = np.clip(
    #         0.45 * looking +
    #         0.30 * gaze +
    #         0.15 * emotions["neutral"] +
    #         0.10 * ear,
    #         0,
    #         1
    #     )


    #     fatigue = np.clip(
    #         0.35 * yawn +
    #         0.25 * blink +
    #         0.20 * (1-ear) +
    #         0.20 * emotions["sadness"],
    #         0,
    #         1
    #     )


    #     engagement = np.clip(
    #         0.45 * attention +
    #         0.25 * emotions["happiness"] +
    #         0.30 * (1-fatigue),
    #         0,
    #         1
    #     )


    #     reading = np.clip(
    #         0.45 * looking +
    #         0.35 * gaze +
    #         0.20 * (1-yaw/45),
    #         0,
    #         1
    #     )


    #     distraction = np.clip(
    #         1-attention,
    #         0,
    #         1
    #     )


    #     confidence = np.clip(
    #         0.45 * emotions["neutral"] +
    #         0.25 * emotions["happiness"] +
    #         0.20 * gaze +
    #         0.10 * looking,
    #         0,
    #         1
    #     )


    #     confusion = np.clip(
    #         0.35 * emotions["surprise"] +
    #         0.25 * emotions["fear"] +
    #         0.20 * emotions["contempt"] +
    #         0.20 * distraction,
    #         0,
    #         1
    #     )


    #     output = np.array(
    #         [
    #             attention,
    #             fatigue,
    #             engagement,
    #             reading,
    #             distraction,
    #             confidence,
    #             confusion,
    #         ],
    #         dtype=np.float32
    #     )


    #     self.seq.append(output)

    #     smooth = np.mean(
    #         np.stack(self.seq),
    #         axis=0
    #     )


    #     if self.smoothed is None:
    #         self.smoothed = smooth
    #     else:
    #         self.smoothed = (
    #             self.alpha * smooth +
    #             (1-self.alpha) * self.smoothed
    #         )


    #     s = self.smoothed


    #     return {
    #         "attention": float(s[0]),
    #         "fatigue": float(s[1]),
    #         "engagement": float(s[2]),
    #         "reading": float(s[3]),
    #         "distraction": float(s[4]),
    #         "confidence": float(s[5]),
    #         "confusion": float(s[6]),
    #     }


# 3
    def predict(
        self,
        face_image,
        face_features,
        behavior,
        landmarks=None,
        blendshapes=None,
    ):

        emotions = self._run_emotion(face_image)

        #
        # -----------------------------
        # Normalize features
        # -----------------------------
        #

        ear = np.clip(
            (float(face_features.ear) - 0.15) / (0.35 - 0.15),
            0.0,
            1.0,
        )

        mar = np.clip(
            float(face_features.mar) / 0.80,
            0.0,
            1.0,
        )

        yaw = np.clip(
            abs(float(face_features.yaw)) / 45.0,
            0.0,
            1.0,
        )

        pitch = np.clip(
            abs(float(face_features.pitch)) / 30.0,
            0.0,
            1.0,
        )

        looking = float(
            behavior.get(
                "looking_at_screen",
                False
            )
        )

        gaze = np.clip(
            float(
                behavior.get(
                    "gaze_stability",
                    0.0
                )
            ),
            0,
            1,
        )

        #
        # Normalize event rates
        #

        blink_rate = np.clip(
            float(
                behavior.get(
                    "blink_count",
                    0
                )
            ) / 30.0,
            0,
            1,
        )

        yawn_rate = np.clip(
            float(
                behavior.get(
                    "yawn_count",
                    0
                )
            ) / 5.0,
            0,
            1,
        )

        #
        # -----------------------------
        # Emotion shortcuts
        # -----------------------------
        #

        neutral = emotions.get("neutral", 0)
        happy = emotions.get("happiness", 0)
        sad = emotions.get("sadness", 0)
        surprise = emotions.get("surprise", 0)
        fear = emotions.get("fear", 0)
        contempt = emotions.get("contempt", 0)

        #
        # -----------------------------
        # Cognitive heuristics
        # -----------------------------
        #

        attention = np.clip(
            0.40 * looking +
            0.30 * gaze +
            0.20 * (1 - yaw) +
            0.10 * neutral,
            0,
            1,
        )

        fatigue = np.clip(
            0.35 * yawn_rate +
            0.25 * blink_rate +
            0.20 * (1 - ear) +
            0.20 * sad,
            0,
            1,
        )

        engagement = np.clip(
            0.50 * attention +
            0.25 * happy +
            0.25 * (1 - fatigue),
            0,
            1,
        )

        reading = np.clip(
            0.40 * looking +
            0.30 * gaze +
            0.20 * (1 - yaw) +
            0.10 * (1 - pitch),
            0,
            1,
        )

        distraction = np.clip(
            0.50 * (1 - looking) +
            0.30 * yaw +
            0.20 * (1 - gaze),
            0,
            1,
        )

        confidence = np.clip(
            0.35 * neutral +
            0.25 * happy +
            0.20 * gaze +
            0.20 * (1 - fear),
            0,
            1,
        )

        confusion = np.clip(
            0.35 * surprise +
            0.25 * fear +
            0.20 * contempt +
            0.20 * distraction,
            0,
            1,
        )

        #
        # -----------------------------
        # Temporal smoothing
        # -----------------------------
        #

        output = np.array(
            [
                attention,
                fatigue,
                engagement,
                reading,
                distraction,
                confidence,
                confusion,
            ],
            dtype=np.float32,
        )

        self.seq.append(output)

        smooth = np.mean(
            np.stack(self.seq),
            axis=0,
        )

        if self.smoothed is None:
            self.smoothed = smooth
        else:
            self.smoothed = (
                self.alpha * smooth
                + (1 - self.alpha) * self.smoothed
            )

        s = self.smoothed

        return {
            "attention": float(s[0]),
            "fatigue": float(s[1]),
            "engagement": float(s[2]),
            "reading": float(s[3]),
            "distraction": float(s[4]),
            "confidence": float(s[5]),
            "confusion": float(s[6]),
        }


# frames : 861
# attention_mean : 0.274 bit stricter
# attention_std : 0.024

# engagement_mean : 0.362 ok
# engagement_std : 0.015

# fatigue_mean : 0.1 ok
# fatigue_std : 0.03

# reading_mean : 0.175 adjust more
# reading_std : 0.024

# confidence_mean : 0.547 diluted
# confidence_std : 0.013

# confusion_mean : 0.149 adjust more
# confusion_std : 0.005