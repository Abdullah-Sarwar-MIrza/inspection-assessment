# Verification record

Measured with the supplied dataset on 16 October 2026. Detailed evidence is
stored under `reports/`.

| Check | Result and evidence |
|---|---|
| Dataset audit | All 7,348 images decoded; `reports/dataset_analysis.json`. |
| Split isolation | Train/validation/test pixel hashes are disjoint; 64 train duplicates excluded; `reports/split_manifest.csv`. |
| Training | Frozen ImageNet ResNet-18 features and balanced logistic head; 272.3 seconds including initialization/download and evaluation. |
| Model/threshold selection | Validation only; C=0.1, threshold=0.36; `reports/metrics.json`. |
| Held-out evaluation | 715 images; accuracy 98.60%; defect precision/recall/F1 98.90%; five FP and five FN. |
| Saved model parity | Training asserts Torch head output matches scikit-learn probabilities within numerical tolerance. |
| Actual HTTP service | Four valid image requests, malformed-image rejection, readiness and OpenAPI checked; `reports/http_evidence.json`. |
| CPU latency | 50 sequential warm HTTP requests; median 28.83 ms and p95 79.02 ms; benchmark settings in HTTP evidence. |
| Integration suite | 16 passed; `reports/test-results.xml`. |
| Browser interface | 12 checks passed in local Chrome: desktop/mobile, real predictions, uploads, drag and drop, downloads, reset, malformed files, errors, offline recovery and help; `reports/ui-verification.json`. |
| Error analysis | All ten errors visually reviewed; `docs/error_analysis.md`. |
| Docker execution | Optional: image build, healthy startup, readiness response and one real JPEG prediction passed; `reports/docker_verification.json`. |

The current pinned Starlette test adapter emits a deprecation warning; its
integration checks passed. Factory acceptance, adversarial security testing and
production load testing have not been performed. The dataset was already
augmented and lacks physical-part IDs, which limits evaluation independence.
