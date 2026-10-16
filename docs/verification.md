# Verification record

Verified on 16 October 2026 with the supplied dataset and the environment
recorded in `reports/http_evidence.json`.

| Check | Result and evidence |
|---|---|
| Full PDF review | Both pages of `Technical Skills Assessment.pdf` were extracted and read. |
| Dataset decode and audit | All 7,348 images decoded; all 300x300 RGB files containing grayscale imagery. `reports/dataset_analysis.json`. |
| Split isolation | Train/validation/test pixel-hash sets are disjoint; 64 train duplicates excluded. Immutable assignments in `reports/split_manifest.csv`. |
| Actual training | ImageNet ResNet-18 features plus balanced logistic head trained locally; wall time 272.3 seconds including initialization/download and evaluation. |
| Threshold/model selection | Validation only; C=0.1 and threshold=0.36. Full candidate metrics saved. |
| Held-out evaluation | 715 images, accuracy 98.60%, defect precision/recall/F1 98.90%; five FP and five FN. |
| Saved head parity | Training asserts Torch output matches scikit-learn probability within numerical tolerance. |
| Actual HTTP service | Uvicorn on localhost, four valid sample requests and one malformed image, readiness and OpenAPI endpoints verified. |
| Latency | 50 measured sequential warm HTTP requests after five warmups; median 28.83 ms, p95 79.02 ms on this CPU. |
| Integration suite | **16 passed**. `reports/test-results.xml`; includes a chunked oversized body, image dimension limits, sanitized 500, model checksum rejection and hash isolation. |
| Error analysis | All ten errors visually reviewed and documented with scores and operational consequences in `docs/error_analysis.md`. |
| Container execution | **Not run**: Docker CLI/daemon is unavailable here. Dockerfile and Compose configuration are supplied for reproduction. |

The environment emits a Starlette deprecation warning about future replacement
of its httpx test adapter. The current pinned adapter passed all tests.

The first test run exposed a test-harness issue: pytest expanded an 8 MiB bytes
parameter into its generated test ID, exceeding Windows environment-variable
limits. Explicit short IDs fixed this; the final suite above passes. No model
or API behavior was changed to address it.

The training console log contains a PowerShell native-stderr error record from
Torchvision's download progress output. The Python process completed training,
saved its final metrics and artifacts, and the artifacts were independently
loaded and verified through the real service and integration suite.

This is an assessment implementation with practical serving controls. It has
not undergone factory acceptance, adversarial security/load testing or a
production deployment. The dataset's pre-augmentation and missing part IDs
limit the independence of the evaluation. See the README and error analysis.

Demo video verified: 168.2 seconds, 1280x720 at 24 fps; frames decoded across all scenes and narration checked for non-silent audio. `reports/video_verification.json`.
