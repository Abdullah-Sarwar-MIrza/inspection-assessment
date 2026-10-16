# Assessment results

All values below are measured from the saved model and split manifest.

| Test metric | Value |
|---|---:|
| Accuracy | 98.6014% |
| Defect precision | 98.8962% |
| Defect recall | 98.8962% |
| Defect F1 | 0.988962 |
| Defect F2 | 0.988962 |
| ROC AUC | 0.998964 |
| Average precision | 0.999411 |
| Majority-class accuracy baseline | 63.3566% |

Confusion matrix rows are true Normal/Defective; columns are predicted Normal/Defective:
`[[257, 5], [5, 448]]`. True negatives: 257; false positives: 5; false negatives: 5; true positives: 448.

Validation selected C=0.1 and threshold=0.36.
The objective maximizes defect F2 subject to at least 95% precision on validation.
The constraint was met: True.
The threshold was locked before test evaluation. The final model is not refit on validation.

Training wall time: 272.3 seconds.
Bootstrap 95% intervals (conditional on fixed predictions and image-level sampling):
`{"accuracy": [0.9762237762237762, 0.9944055944055944], "recall_defective": [0.9781181619256017, 0.997773198060587]}`.

## Dataset audit

Split counts (0=Normal, 1=Defective): `{"train": {"1": 3006, "0": 2249}, "validation": {"1": 752, "0": 562}, "test": {"0": 262, "1": 453}}`.
Unreadable images: 0; excluded exact duplicate train images: 64.
Unique pixel hashes in the test set: 715.
Potential test/train near matches (dHash <=4): 566.
These are screening candidates; dHash similarity is not a reliable physical-part identifier.
The 512px collection is excluded to avoid mixing unverified overlapping sources.
The original 300px dataset was already augmented by its author. Exact deduplication
does not prove independent parts across splits. A new part/batch/time-separated
prospective dataset is required before a factory rollout.

## Live HTTP verification

Four real sample images returned HTTP 200 and matched local inference. A corrupt
image returned HTTP 422. Full responses and request IDs: `http_evidence.json`.

Warm CPU inference service, two Torch threads, 50 measured requests after five warmups:
HTTP p50 28.8 ms,
p95 79.0 ms.
Decode/preprocess/inference p50 26.2 ms,
p95 75.2 ms.
Hardware/platform: Intel64 Family 6 Model 158 Stepping 10, GenuineIntel, Windows-10-10.0.19045-SP0.
This is a sequential localhost benchmark, not a throughput or SLA claim.
