# Assessment deliverables

The repository contains the implementation, trained model, evaluation evidence
and deployment configuration listed below.

| Requirement | Repository location |
|---|---|
| Setup and usage | `README.md` |
| Training and dataset preparation | `casting_inspector/train.py` |
| Trained model and metadata | `artifacts/` |
| Image upload interface and inference API | `casting_inspector/api.py`, `casting_inspector/web/` |
| Architecture diagram | `docs/architecture.md` |
| Model choices and trade-offs | `docs/technical_decisions.md` |
| Dataset audit and split assignments | `reports/dataset_analysis.json`, `reports/split_manifest.csv` |
| Evaluation results and confusion matrix | `reports/RESULTS.md`, `reports/metrics.json`, `reports/figures/` |
| False-positive / false-negative analysis | `docs/error_analysis.md`, `reports/errors.json` |
| Example images and predictions | `reports/examples/`, `reports/example_predictions.json` |
| Docker deployment | `Dockerfile`, `compose.yaml`, `docs/docker_guide.md` |
| Inference integration tests | `tests/test_service.py` |
| Repeatable API and browser verification | `scripts/validate_and_report.py`, `scripts/verify_ui.py` |
| Checks performed and limits | `docs/verification.md` |
| Recorded demonstration | [Video download](https://github.com/Abdullah-Sarwar-MIrza/inspection-assessment/releases/download/assessment-demo/Technical-Skills-Assessment-Demo.mp4); [details](demo.md) |
