# Assessment deliverables

| PDF requirement | Delivered location |
|---|---|
| Dataset analysis/preparation | `reports/dataset_analysis.json`, `casting_inspector/train.py` |
| Normal / Defective classification | `artifacts/model.pt`, shared `model.py` |
| Preprocessing / augmentation | Full-image resize and normalization; train-only mirror and small rotation |
| Class imbalance | Balanced training weights; stratified train/validation and per-class metrics |
| Train/validation/test split | `reports/split_manifest.csv`, exact hash-isolation check |
| Architecture choice / transfer learning | `docs/technical_decisions.md` |
| Precision / recall / F1 / confusion matrix | `reports/RESULTS.md`, `reports/metrics.json`, `reports/figures/` |
| False-positive / false-negative analysis | `docs/error_analysis.md`, `reports/errors.json` |
| Working image inference API | `casting_inspector/api.py`, actual responses in `reports/http_evidence.json` |
| Predicted class / confidence | API and CLI outputs; `reports/example_predictions.json` |
| Validation / logging / errors | Upload bounds, strict decoding, exception responses, request IDs, lifecycle probes |
| GitHub repository | Public `Abdullah-Sarwar-MIrza/inspection-assessment` after publishing |
| Setup / approach / strategy / results / limitations | `README.md` |
| Architecture diagram | Mermaid in `docs/architecture.md` |
| Trained model and reproduction | `artifacts/`, training CLI and pinned dependencies |
| Example predictions | Four source images, recorded JSON and sample figure |
| Docker / deployment | `Dockerfile`, `compose.yaml`, README deployment commands |
| 2-3 minute video | `docs/demo.mp4`, narration in `docs/demo_script.md` |
| Interview explanation | `docs/technical_decisions.md` and concrete error analysis |

The demo is explicitly an automatically narrated walkthrough using actual
recorded HTTP responses. The user should review the code and technical decisions
and be prepared to explain and modify them during the interview, as the PDF
requires. AI assistance was used to prepare this submission.
