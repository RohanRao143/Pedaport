# Pedaport — Adaptive Learning Intelligence

Pedaport explores how browser-based inference and temporal interaction metrics can support adaptive learning experiences. Its processing pipeline turns frame-level observations into rolling-window summaries and derived metrics that can be logged for analysis and used by an ONNX-compatible inference component.

The objective is to provide signals that may help an educational application adapt its next step—not to diagnose a learner or treat webcam measurements as definitive evidence of an internal mental state.

## Processing pipeline

```text
Frame metrics
    ↓
Rolling window
    ↓
Summary generation
    ↓
Derived scores
    ↓
CSVLogger.write()
    ↓
logs/metrics.csv
```

## Key concepts

- **Frame metrics:** Collect raw or per-frame measurements from the browser-based observation pipeline.
- **Rolling windows:** Aggregate measurements over a moving time interval to reduce dependence on a single frame.
- **Summary generation:** Calculate temporal summaries from the window, such as counts, averages, or stability measures supported by the implementation.
- **Derived scores:** Transform available measurements into higher-level indicators for analysis.
- **CSV logging:** Persist structured metrics to `logs/metrics.csv` for inspection, experiments, and offline evaluation.
- **ONNX inference:** Use a compatible model in the browser when a model and inference runtime are configured.
- **Adaptive learning direction:** Use validated indicators to inform possible learning recommendations or next actions.

## Example workflow

1. Start the application in an environment with the required browser permissions.
2. Allow camera access only if the feature being tested requires it.
3. Collect frame-level measurements through the configured pipeline.
4. Aggregate observations over a rolling window.
5. Calculate summary features and derived scores.
6. Write the available metrics to CSV.
7. Inspect the logs to assess data quality and behavior before using the metrics in an adaptive policy.

## Technology

- JavaScript and browser APIs
- ONNX Runtime Web, where configured
- Computer-vision or face-landmark inference components, where configured
- Rolling-window aggregation
- Feature engineering and derived metrics
- CSV-based experiment logging

Update this list to match the exact dependencies and models used in the current implementation.

## Data and interpretation

Attention, fatigue, focus stability, and reading engagement are difficult to infer reliably from visual observations alone. These metrics should be treated as uncertain proxies, not ground truth. Camera conditions, lighting, individual differences, accessibility needs, and normal behavior can affect measurements.

Recommended safeguards:

- Request clear permission before camera use.
- Prefer local processing where possible and explain any data storage.
- Avoid collecting or retaining raw frames unless necessary and explicitly disclosed.
- Do not use derived scores as the sole basis for high-impact educational decisions.
- Evaluate model behavior across varied conditions and users.
- Keep the inference pipeline separate from recommendation policy so each can be evaluated independently.

## Logging

The intended CSV output is:

```text
logs/metrics.csv
```

The precise columns, timestamp format, window size, and score definitions should be documented from the implementation. Treat the sample path as a description of the stated workflow, not a guarantee that the file is created before the application runs.

## Evaluation

Useful evaluation questions include:

- Are frame measurements stable under normal lighting and camera movement?
- How do rolling-window size and missing observations affect derived scores?
- Are logged values reproducible and correctly timestamped?
- Do scores correlate with independently collected, consented evaluation data?
- Do recommendations improve learning outcomes without unfairly penalizing different behavior patterns?

## Potential improvements

- Add explicit schemas and versioning for CSV output.
- Handle missing frames and invalid measurements consistently.
- Provide configurable window sizes and feature definitions.
- Add privacy controls and clear data-retention behavior.
- Evaluate calibration and uncertainty before generating recommendations.
- Separate raw observations, derived features, model output, and recommendation decisions.

## Project status

This README describes the processing workflow and design intent. Add the exact setup commands, model filenames, feature definitions, and inference entry points from the repository before publishing.





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
