"""Reproducible frozen-ResNet transfer learning with validation-only selection."""
import argparse
import csv
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             classification_report, confusion_matrix, fbeta_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .model import IMAGE_SIZE, Inspector, preprocess

LABELS = {"ok_front": 0, "def_front": 1}


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def inventory(root):
    rows, invalid = [], []
    for split in ("train", "test"):
        for folder, label in LABELS.items():
            files = sorted((root / split / folder).glob("*"))
            if not files:
                raise ValueError(f"No data: {root / split / folder}")
            for path in files:
                try:
                    with Image.open(path) as image:
                        image.load()
                        rgb = image.convert("RGB")
                        digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                        gray = np.asarray(rgb.convert("L").resize((9, 8)), dtype=np.int16)
                        bits = gray[:, 1:] > gray[:, :-1]
                        dhash = int.from_bytes(np.packbits(bits).tobytes(), "big")
                        rows.append({"path": str(path.as_posix()), "source_split": split,
                                     "label": label, "sha256_pixels": digest,
                                     "dhash": f"{dhash:016x}", "width": image.width,
                                     "height": image.height, "mode": image.mode})
                except (OSError, ValueError) as exc:
                    invalid.append({"path": str(path), "error": str(exc)})
    if invalid:
        raise ValueError(f"Unreadable images: {invalid}")
    return rows


class Images(Dataset):
    def __init__(self, rows, augmented=False):
        self.rows, self.augmented = rows, augmented
        self.transform = preprocess()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.rows[index]["path"]) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
        if self.augmented:
            # Deterministic second training view; split BEFORE augmentation.
            image = ImageOps.mirror(image)
            image = image.rotate(8 if index % 2 else -8, resample=Image.Resampling.BILINEAR,
                                 fillcolor=(128, 128, 128))
        return self.transform(image)


def extract(model, rows, batch_size, workers, augmented=False):
    key = hashlib.sha256(json.dumps({"rows": [(r["path"], r["sha256_pixels"]) for r in rows],
                                    "augmented": augmented, "size": IMAGE_SIZE,
                                    "recipe": "resnet18-v1-mirror-rotate8-v1"}).encode()).hexdigest()
    cache = Path("reports/cache") / f"{key}.npy"
    if cache.exists():
        print(f"Using verified recipe/manifest feature cache {cache.name}", flush=True)
        return np.load(cache, allow_pickle=False)
    loader = DataLoader(Images(rows, augmented), batch_size=batch_size,
                        num_workers=workers, shuffle=False)
    features = []
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            features.append(model.backbone(batch).numpy())
            if index % 20 == 0:
                print(f"features augmentation={augmented} {min((index+1)*batch_size, len(rows))}/{len(rows)}", flush=True)
    result = np.concatenate(features)
    cache.parent.mkdir(exist_ok=True)
    np.save(cache, result, allow_pickle=False)
    return result


def metrics(y, probability, threshold):
    predicted = (probability >= threshold).astype(int)
    return {"accuracy": float(accuracy_score(y, predicted)),
            "precision_defective": float(precision_score(y, predicted, zero_division=0)),
            "recall_defective": float(recall_score(y, predicted, zero_division=0)),
            "f1_defective": float(fbeta_score(y, predicted, beta=1)),
            "f2_defective": float(fbeta_score(y, predicted, beta=2)),
            "roc_auc": float(roc_auc_score(y, probability)),
            "average_precision": float(average_precision_score(y, probability)),
            "confusion_matrix_normal_defective": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
            "classification_report": classification_report(y, predicted, labels=[0, 1],
                target_names=["Normal", "Defective"], output_dict=True, zero_division=0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("casting_data/casting_data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    for directory in ("artifacts", "reports", "reports/figures", "reports/examples"):
        Path(directory).mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    rows = inventory(args.data)
    test = [r for r in rows if r["source_split"] == "test"]
    test_hashes = {r["sha256_pixels"] for r in test}
    seen, clean, excluded = {}, [], []
    all_labels = {}
    for row in rows:
        digest = row["sha256_pixels"]
        if digest in all_labels and all_labels[digest] != row["label"]:
            raise ValueError(f"Contradictory labels: {row['path']}")
        all_labels[digest] = row["label"]
        if row["source_split"] == "test":
            continue
        if digest in test_hashes or digest in seen:
            excluded.append({**row, "reason": "exact_duplicate"})
        else:
            seen[digest] = row["label"]
            clean.append(row)
    training, validation = train_test_split(clean, test_size=0.2, random_state=args.seed,
                                          stratify=[r["label"] for r in clean])
    training, validation = sorted(training, key=lambda r: r["path"]), sorted(validation, key=lambda r: r["path"])
    for split, items in (("train", training), ("validation", validation), ("test", test)):
        for row in items:
            row["split"] = split
    manifest = training + validation + test
    with open("reports/split_manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    # dHash is a screening signal, not proof of shared product identity.
    train_hashes = [int(r["dhash"], 16) for r in training]
    near_test = []
    for row in test:
        h = int(row["dhash"], 16)
        distance = min((h ^ other).bit_count() for other in train_hashes)
        if distance <= 4:
            near_test.append({"path": row["path"], "minimum_dhash_distance": distance})
    data_report = {"source_counts": dict(Counter(f"{r['source_split']}/{r['label']}" for r in rows)),
                   "split_counts": {s: dict(Counter(str(r["label"]) for r in items)) for s, items in
                                    (("train", training), ("validation", validation), ("test", test))},
                   "label_mapping": {"0": "Normal", "1": "Defective"},
                   "dimensions": dict(Counter(f"{r['width']}x{r['height']}" for r in rows)),
                   "modes": dict(Counter(r["mode"] for r in rows)),
                   "excluded": excluded, "test_unique_pixel_hashes": len(test_hashes),
                   "potential_near_overlap_test": near_test,
                   "unreadable_images": 0,
                   "unused_512_collection": "Excluded: possible source overlap with augmented 300px set; no reliable physical-part IDs."}
    save_json("reports/dataset_analysis.json", data_report)
    print(json.dumps(data_report["split_counts"]), flush=True)
    model = Inspector(pretrained=True).eval()
    x = extract(model, training, args.batch_size, args.workers)
    xa = extract(model, training, args.batch_size, args.workers, augmented=True)
    xv = extract(model, validation, args.batch_size, args.workers)
    y = np.array([r["label"] for r in training])
    yv = np.array([r["label"] for r in validation])
    scaler = StandardScaler().fit(np.concatenate([x, xa]))
    xx = scaler.transform(np.concatenate([x, xa]))
    yy = np.concatenate([y, y])
    candidates = []
    best = None
    # Recall is valuable for defect screening. Select F2 with >=95% validation precision.
    for c in (0.001, 0.01, 0.1, 1.0, 10.0):
        classifier = LogisticRegression(C=c, class_weight="balanced", max_iter=2500,
                                         random_state=args.seed).fit(xx, yy)
        probabilities = classifier.predict_proba(scaler.transform(xv))[:, 1]
        for threshold in np.arange(0.10, 0.901, 0.02):
            pred = probabilities >= threshold
            precision = precision_score(yv, pred, zero_division=0)
            recall = recall_score(yv, pred)
            f2 = fbeta_score(yv, pred, beta=2)
            score = (precision >= 0.95, f2, precision, recall)
            candidates.append({"C": c, "threshold": float(threshold), "precision": precision,
                               "recall": recall, "f2": f2})
            if best is None or score > best[0]:
                best = (score, classifier, float(threshold), c, probabilities)
    _, classifier, threshold, best_c, pv = best
    with torch.no_grad():
        model.feature_mean.copy_(torch.tensor(scaler.mean_, dtype=torch.float32))
        model.feature_scale.copy_(torch.tensor(scaler.scale_, dtype=torch.float32))
        model.head.weight.copy_(torch.tensor(classifier.coef_, dtype=torch.float32))
        model.head.bias.copy_(torch.tensor(classifier.intercept_, dtype=torch.float32))
    # Test is evaluated only after all model/threshold choices are locked.
    xt = extract(model, test, args.batch_size, args.workers)
    yt = np.array([r["label"] for r in test])
    pt = classifier.predict_proba(scaler.transform(xt))[:, 1]
    with torch.inference_mode():
        portable = torch.sigmoid(model.head((torch.tensor(xt) - model.feature_mean) / model.feature_scale)).numpy().ravel()
    np.testing.assert_allclose(portable, pt, atol=2e-6, rtol=2e-5)
    torch.save(model.state_dict(), "artifacts/model.pt")
    model_hash = hashlib.sha256(Path("artifacts/model.pt").read_bytes()).hexdigest()
    metadata = {"model_version": "resnet18-frozen-v1", "sha256": model_hash,
                "image_size": IMAGE_SIZE, "threshold": threshold, "C": best_c,
                "seed": args.seed, "backbone": "ResNet18 IMAGENET1K_V1 frozen",
                "classes": ["Normal", "Defective"], "confidence": "Uncalibrated sigmoid score"}
    save_json("artifacts/metadata.json", metadata)
    result = {"validation": metrics(yv, pv, threshold), "test": metrics(yt, pt, threshold),
              "selection": {"C": best_c, "threshold": threshold,
                            "rule": "Max validation F2 subject to precision >= 0.95; fallback maximum F2",
                            "precision_constraint_met": bool(best[0][0])},
              "candidates": candidates,
              "majority_test_accuracy": float(max(np.mean(yt == 0), np.mean(yt == 1)))}
    # Bootstrap fixed predictions; uncertainty conditional on this split, not physical-part independence.
    rng = np.random.default_rng(args.seed)
    samples = []
    for _ in range(1000):
        idx = rng.integers(0, len(yt), len(yt))
        samples.append([accuracy_score(yt[idx], pt[idx] >= threshold),
                        recall_score(yt[idx], pt[idx] >= threshold, zero_division=0)])
    result["test_bootstrap_95_percent"] = dict(zip(["accuracy", "recall_defective"],
        np.quantile(samples, [0.025, 0.975], axis=0).T.tolist()))
    result["training_wall_seconds"] = time.perf_counter() - start
    save_json("reports/metrics.json", result)
    save_json("artifacts/evaluation.json", {"model_sha256": model_hash,
              "test_accuracy": result["test"]["accuracy"], "test_images": len(test)})
    errors = []
    with open("reports/test_predictions.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "predicted", "defect_probability", "error_type"])
        writer.writeheader()
        for row, probability in zip(test, pt):
            pred = int(probability >= threshold)
            kind = "FN" if row["label"] == 1 and pred == 0 else "FP" if row["label"] == 0 and pred == 1 else "correct"
            item = {"path": row["path"], "label": row["label"], "predicted": pred,
                    "defect_probability": float(probability), "error_type": kind}
            writer.writerow(item)
            if kind != "correct":
                errors.append(item)
    save_json("reports/errors.json", errors)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(yt, pt >= threshold, display_labels=["Normal", "Defective"],
                                           ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Held-out test confusion matrix")
    fig.tight_layout(); fig.savefig("reports/figures/confusion_matrix.png", dpi=160); plt.close(fig)
    precision, recall, _ = precision_recall_curve(yt, pt)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision); ax.set(xlabel="Defect recall", ylabel="Defect precision", title="Held-out test precision-recall")
    ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig("reports/figures/precision_recall.png", dpi=160); plt.close(fig)
    for name, items in (("errors", errors[:16]), ("samples", [test[i] | {"defect_probability": float(pt[i]), "error_type": "sample"} for i in np.linspace(0, len(test)-1, 12).astype(int)])):
        if not items:
            continue
        fig, axes = plt.subplots((len(items)+3)//4, 4, figsize=(12, 3*((len(items)+3)//4)), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, item in zip(axes.flat, items):
            with Image.open(item["path"]) as image:
                ax.imshow(image, cmap="gray")
            ax.set_title(f"{item['error_type']} true={item['label']} p(def)={item['defect_probability']:.3f}", fontsize=9)
        fig.tight_layout(); fig.savefig(f"reports/figures/{name}.png", dpi=130); plt.close(fig)
    print(json.dumps({"test": result["test"], "selection": result["selection"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
