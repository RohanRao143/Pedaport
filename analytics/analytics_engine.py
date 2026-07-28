"""
analytics_engine.py

Adaptive Learning Analytics Layer

Responsibilities:

1. Maintain rolling frame history
2. Aggregate every N seconds
3. Generate derived behavioral scores
4. Persist summaries

The generated scores are behavioral indicators
from observable signals.

They are not direct measurements of cognition.
"""


from collections import deque
from dataclasses import dataclass, asdict

from datetime import datetime

import csv
import os
import statistics
import time



# ============================================================
# Configuration
# ============================================================


WINDOW_SECONDS = 60

CSV_FILE = "logs/metrics.csv"

os.makedirs(
    "logs",
    exist_ok=True
)



# ============================================================
# Rolling Window
# ============================================================


class RollingWindow:


    def __init__(
        self,
        seconds=WINDOW_SECONDS
    ):

        self.seconds = seconds

        self.frames = deque()



    def add(
        self,
        data
    ):

        self.frames.append(
            {
                "time":time.time(),
                **data
            }
        )


        self.cleanup()



    def cleanup(self):

        now=time.time()


        while self.frames:

            if now - self.frames[0]["time"] > self.seconds:

                self.frames.popleft()

            else:

                break



    def values(
        self
    ):

        return list(
            self.frames
        )



    def clear(self):

        self.frames.clear()



# ============================================================
# Statistical helpers
# ============================================================


def mean(values):

    if not values:
        return 0

    return statistics.mean(values)



def std(values):

    if len(values)<2:

        return 0

    return statistics.stdev(values)



def clamp(value):

    return max(
        0,
        min(
            100,
            value
        )
    )



# ============================================================
# Derived Scores
# ============================================================


class DerivedScores:


    """
    Converts observable measurements
    into normalized behavioral indicators.
    """



    def calculate(
        self,
        summary
    ):



        # ----------------------------------
        # Attention
        # ----------------------------------

        attention = (

            summary["screen_attention_pct"]
            *
            0.5

            +

            summary["gaze_stability"]
            *
            100
            *
            0.3

            +

            summary["tracking_quality"]
            *
            0.2

        )


        # ----------------------------------
        # Fatigue
        # ----------------------------------

        blink_factor=min(
            summary["blink_rate"]/30,
            1
        )


        yawn_factor=min(
            summary["yawn_count"]/3,
            1
        )


        eye_factor=max(
            0,
            1-summary["mean_ear"]
        )


        fatigue=(

            blink_factor*40

            +

            yawn_factor*40

            +

            eye_factor*20

        )


        # ----------------------------------
        # Focus Stability
        # ----------------------------------

        movement_penalty=min(
            summary["face_velocity"]*10,
            50
        )


        focus=(

            attention*0.6

            +

            (100-movement_penalty)*0.4

        )



        # ----------------------------------
        # Interaction
        # ----------------------------------

        interaction=(

            min(
                summary["expression_energy"]*100,
                50
            )

            +

            min(
                summary["yawn_count"]*0,
                0
            )

        )



        # ----------------------------------
        # Reading Stability
        # ----------------------------------

        reading=(

            summary["gaze_stability"]
            *
            100
            *
            0.7

            +

            max(
                0,
                100-summary["movement_variance"]*100
            )
            *
            0.3

        )



        return {


            "attention_score":
                round(
                    clamp(attention),
                    2
                ),


            "fatigue_score":
                round(
                    clamp(fatigue),
                    2
                ),


            "focus_score":
                round(
                    clamp(focus),
                    2
                ),


            "interaction_score":
                round(
                    clamp(interaction),
                    2
                ),


            "reading_score":
                round(
                    clamp(reading),
                    2
                )

        }





# ============================================================
# Aggregator
# ============================================================


class AnalyticsEngine:



    def __init__(self, logger=None):

        self.window = RollingWindow()

        self.scorer = DerivedScores()

        self.last_log=time.time()


        self.logger = logger



    def update(
        self,
        features
    ):


        self.window.add(
            features
        )


        if time.time()-self.last_log >= WINDOW_SECONDS:

# 1 
            # result=self.generate_summary()
# 2

            result=self.generate_averages()

            # self.write_csv(
            #     result
            # )

            if self.logger:
                self.logger.write(result)


            self.window.clear()

            self.last_log=time.time()



            return result


        return None




# 1
    # def generate_summary(self):


    #     frames=self.window.values()



    #     if not frames:

    #         return {}



    #     summary={}


    #     summary["timestamp"]=datetime.now().isoformat()


    #     summary["frames"]=len(
    #         frames
    #     )



    #     # ----------------------------------
    #     # Tracking
    #     # ----------------------------------


    #     summary["face_visible_pct"]=round(

    #         sum(
    #             1
    #             for x in frames
    #             if x.get("confidence",0)>0
    #         )

    #         /

    #         len(frames)

    #         *

    #         100,

    #         2
    #     )



    #     summary["screen_attention_pct"]=round(

    #         sum(
    #             1
    #             for x in frames
    #             if x.get(
    #                 "looking_at_screen",
    #                 False
    #             )
    #         )

    #         /

    #         len(frames)

    #         *

    #         100,

    #         2
    #     )



    #     # ----------------------------------
    #     # Face metrics
    #     # ----------------------------------


    #     summary["mean_ear"]=mean(
    #         [
    #             x.get("ear",0)
    #             for x in frames
    #         ]
    #     )


    #     summary["std_ear"]=std(
    #         [
    #             x.get("ear",0)
    #             for x in frames
    #         ]
    #     )


    #     summary["mean_mar"]=mean(
    #         [
    #             x.get("mar",0)
    #             for x in frames
    #         ]
    #     )


    #     summary["std_mar"]=std(
    #         [
    #             x.get("mar",0)
    #             for x in frames
    #         ]
    #     )



    #     # ----------------------------------
    #     # Events
    #     # ----------------------------------


    #     summary["blink_count"]=max(

    #         [
    #             x.get(
    #                 "blink_count",
    #                 0
    #             )

    #             for x in frames

    #         ]

    #         or [0]

    #     )


    #     summary["blink_rate"]=summary["blink_count"]



    #     summary["avg_blink_duration"]=mean(

    #         [
    #             x.get(
    #                 "avg_blink_duration",
    #                 0
    #             )

    #             for x in frames
    #         ]

    #     )


    #     summary["yawn_count"]=max(

    #         [
    #             x.get(
    #                 "yawn_count",
    #                 0
    #             )

    #             for x in frames
    #         ]

    #         or [0]

    #     )



    #     # ----------------------------------
    #     # Head
    #     # ----------------------------------


    #     for key in [
    #         "pitch",
    #         "yaw",
    #         "roll"
    #     ]:


    #         values=[

    #             x.get(
    #                 key,
    #                 0
    #             )

    #             for x in frames

    #         ]


    #         summary[key+"_mean"]=mean(
    #             values
    #         )

    #         summary[key+"_std"]=std(
    #             values
    #         )



    #     summary["face_velocity"]=mean(

    #         [
    #             x.get(
    #                 "face_velocity",
    #                 0
    #             )

    #             for x in frames

    #         ]

    #     )



    #     summary["movement_variance"]=mean(

    #         [
    #             x.get(
    #                 "movement_variance",
    #                 0
    #             )

    #             for x in frames

    #         ]

    #     )



    #     summary["gaze_stability"]=mean(

    #         [
    #             x.get(
    #                 "gaze_stability",
    #                 0
    #             )

    #             for x in frames

    #         ]

    #     )


    #     summary["expression_energy"]=mean(

    #         [
    #             x.get(
    #                 "expression_energy",
    #                 0
    #             )

    #             for x in frames

    #         ]

    #     )



    #     summary["tracking_quality"]=mean(

    #         [
    #             x.get(
    #                 "confidence",
    #                 0
    #             )

    #             for x in frames

    #         ]

    #     )



    #     # Add derived scores

    #     summary.update(
    #         self.scorer.calculate(
    #             summary
    #         )
    #     )


    #     return summary


# 2
    def generate_averages(self):
        frames = self.window.values()

        if not frames:
            return {}

        summary = {
            "timestamp": datetime.now().isoformat(),
            "frames": len(frames),
        }

        for key in [
            "attention",
            "engagement",
            "fatigue",
            "reading",
            "confidence",
            "confusion",
        ]:
            values = [x.get(key, 0.0) for x in frames]
            summary[f"{key}_mean"] = mean(values)
            summary[f"{key}_std"] = std(values)

        return summary



# ============================================================
# CSV Logger
# ============================================================


class CSVLogger:



    def __init__(
        self,
        filename=CSV_FILE
    ):

        self.filename=filename


    def write(
        self,
        data
    ):


        exists=os.path.exists(
            self.filename
        )


        with open(
            self.filename,
            "a",
            newline=""
        ) as f:


            writer=csv.DictWriter(
                f,
                fieldnames=data.keys()
            )


            if not exists:

                writer.writeheader()


            writer.writerow(
                data
            )


