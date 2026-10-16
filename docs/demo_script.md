# Demo narration

This is an automatically narrated walkthrough using actual recorded HTTP responses, not a human screen recording.

## Scene 1

This assessment implements a visual inspection system for casting products. It takes one image and returns Normal or Defective, together with a confidence score and model version. The submission contains reproducible training code, actual trained weights, held out evaluation, error analysis, and a working HTTP API. This demonstration uses recorded responses from real requests to that service, with automated narration.

## Scene 2

I preserved the official test set and split the remaining training pool into training and validation with stratified classes and a fixed seed. Every image is decoded and hashed. Exact training duplicates and exact test overlaps are excluded. Augmentation happens only after splitting. The separate higher resolution source collection is excluded because its overlap is uncertain. The publisher already augmented this dataset, so these image level results cannot prove generalization to independent physical parts.

## Scene 3

I chose a frozen ImageNet ResNet eighteen feature extractor to keep training practical on this CPU and limit overfitting. Images are resized without cropping, so boundary defects remain visible. A balanced logistic head handles the modest class imbalance. Standardization sees training features only. Validation selects regularization and the decision threshold by maximizing defect F two, subject to a ninety five percent precision floor. This favors catching defective parts while controlling false rejects.

## Scene 4

On the locked test set, accuracy is 98.60 percent, defect precision is 98.90 percent, and recall is 98.90 percent. There are 5 false positives and 5 false negatives. A false positive causes unnecessary rejection or review. A false negative lets a defective part pass. The report lists every prediction and observed error. Tiny defects, ambiguous labels, and camera changes remain weaknesses; test feedback was not used to retune the model.

## Scene 5

Here is a real uploaded defective image and the response captured from the running API. The response includes its class, confidence, defect score, threshold, version, request identifier, and elapsed time. The confidence is an uncalibrated model score, not a guarantee. A corrupt image returns HTTP four twenty two. The service also validates media type, upload size, dimensions, and decoded content. Shared preprocessing keeps command line and API predictions consistent.

## Scene 6

Deployment uses a non root container, a read only filesystem, bounded resources, startup integrity checks, and readiness probes. Request logs record identifiers, status, and latency without retaining uploaded images. The report measures warm sequential CPU latency rather than claiming production throughput. Before a factory rollout, collect independent batches, validate lighting and motion conditions, add authenticated transport and operator review, and agree acceptance criteria. The README explains setup, trade offs, reproduction, limitations, and maintenance.


Duration: 168.1 seconds. Rebuild on Windows with `python scripts/build_demo.py` after `python scripts/validate_and_report.py`.
