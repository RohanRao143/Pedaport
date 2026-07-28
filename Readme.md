Frame metrics
      |
      v
Rolling window
      |
      v
Summary generation
      |
      v
Derived scores
      |
      v
CSVLogger.write()
      |
      v
logs/metrics.csv





Criteria Rules for derived scores

Focused
    attention > 80
    focus > 75
    fatigue < 30

Reading
    attention > 70
    gaze_stability > 80
    interaction < 20

Distracted
    attention < 40
    head_motion > 60

Fatigued
    fatigue > 70

Needs Break
    fatigue > 80
    attention < 30

Highly Engaged
    attention > 90
    focus > 85
    interaction > 40

Low Confidence
    face_visibility < 40