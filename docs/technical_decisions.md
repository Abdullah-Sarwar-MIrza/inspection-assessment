# Technical decisions and interview guide

**Why transfer learning?** Small labeled datasets benefit from existing visual
features. ResNet-18 is a straightforward, maintainable CPU baseline. A frozen
backbone avoids backpropagation cost and constrains overfitting; regularized
logistic fitting is convex and easy to inspect. No claim is made that it beats
a fine-tuned network. Consider fine-tuning the last block, a larger resolution,
or MobileNet in a new validation comparison when compute and representative
data permit it.

**Why no object detector?** The labels are image-level Normal/Defective, without
bounding boxes or masks. Classification directly matches the requirement.
Localization would need reviewed defect annotations. A class-activation map is
not a verified defect segmentation.

**Why resize the whole image?** Edge burrs matter. A center crop can remove
them. 160px reduces CPU work, but may lose small holes/cracks. Images outside
the square-camera domain are accepted technically but not validated; aspect
ratio distortion is a known limitation.

**What is trained?** Only a 512-weight linear head and bias are optimized;
the mean/scale buffers are fit on original and augmented training features.
ImageNet weights and batch-normalization statistics stay fixed. Logistic
regression minimizes balanced, regularized binary log loss; C controls inverse
regularization strength. A low C shrinks coefficients more strongly.

**How is imbalance handled?** Defective examples modestly outnumber normal
ones. Balanced class weights give each class equal total fitting weight.
Per-class support and recall expose the minority-class behavior; accuracy alone
can obscure failures. Threshold tuning additionally models the operational
preference for avoiding missed defects.

**False positive versus false negative?** A false positive rejects or reviews
a good part, creating rework and inspection cost. A false negative passes a bad
part and can create downstream failures or returns. Neither cost is supplied,
so validation F2 with a 95% precision floor is an explicit provisional policy.
Production requires a cost model and operator review for uncertain decisions.

**What prevents leakage?** Exact pixel duplicates are removed from training
before splitting, and hashes are compared against test. Augmentation and
feature standardization use training only; model and threshold selection use
validation only. Test is scored after selection. However, the upstream data
was already augmented and lacks physical-part identifiers, so this does not
establish independence of augmented views. Stronger claims need new data.

**What does confidence mean?** The sigmoid head produces an uncalibrated
defect score. The returned confidence is p(defect) for Defective and
1-p(defect) for Normal. Class weighting and threshold selection can change
probabilistic interpretation. Calibrate using a separate representative set
and inspect reliability curves/Brier score if decisions depend on probability.

**Accuracy versus latency?** The measured report establishes this model's
operating point on this CPU. Larger images and fine-tuning may improve small
defect sensitivity, at higher compute cost. Quantization or ONNX could improve
latency but require new numerical parity and accuracy checks. No alternative
architecture speed/accuracy numbers are invented.

**How would I debug a bad prediction?** Retrieve its request ID and model
version, reproduce using the CLI, verify class mapping and shared preprocessing,
inspect the source image and score relative to the threshold, then compare
similar training cases. Have an operator review the label. Add corrected
examples to a future training pool while keeping a new batch-separated test.

**What is needed for a real line?** Camera triggering and synchronization,
part tracking, blur/illumination checks, a fail-safe reject/manual review
policy, authenticated transport, device-specific load testing, prospective
acceptance criteria, drift monitoring and a controlled rollback process.
