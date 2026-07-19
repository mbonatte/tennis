# Contributing to Tennis Analyzer

Thank you for contributing to the Tennis Analyzer project!

## Documentation Maintenance Requirement

To prevent the technical documentation from silently becoming obsolete, any code change affecting any of the following items **MUST** update the relevant model or pipeline documentation in the same pull request:

- **Model Architecture** (e.g. layers, channels, backbone structure);
- **Model Checkpoint** (e.g. weights filename, size, hashes, source/licensing);
- **Preprocessing** (e.g. resolution changes, color space shifts, normalizations);
- **Output Conversion** (e.g. coordinate transformations, heatmap extraction steps, Hough Circles parameters);
- **Threshold** (e.g. confidence settings, bounce regression score filters, keypoint detection limits);
- **Post-processing** (e.g. linear/spline interpolation, continuity gates, outlier removals);
- **Heuristic** (e.g. bounce-to-contact reclassifications, serve recovery criteria, player role constraints);
- **Assumption** (e.g. singles vs doubles court boundaries, fixed camera perspective limits);
- **Result Field** (e.g. schemas, `result.json` structures, artifact formats);
- **Stage Ordering** (e.g. pipeline stages sequence).

The main technical references are located at:
- [Analysis Pipeline Document](docs/ANALYSIS_PIPELINE.md)
- [Models Directory](docs/models/README.md)

---

## Local Development Guidelines

Before opening a pull request, please ensure all code formatting, linting, and tests pass:

```bash
# Run Ruff lint and format checks
python -m ruff check app tennis_analyzer tests ball.py
python -m ruff format --check app tennis_analyzer tests ball.py

# Run the test suite
python -m pytest -q
```
