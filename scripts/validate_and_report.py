"""Exercise the real HTTP service, record evidence and render a standalone report."""
import csv
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from casting_inspector.model import Predictor, decode_image
    torch.set_num_threads(2)
    metrics = json.loads(Path("reports/metrics.json").read_text())
    data = json.loads(Path("reports/dataset_analysis.json").read_text())
    with open("reports/test_predictions.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    predictor = Predictor()
    examples = []
    for label in ("0", "1"):
        selected = [r for r in rows if r["label"] == label and r["error_type"] == "correct"][:2]
        for index, row in enumerate(selected):
            destination = Path(f"reports/examples/{'normal' if label == '0' else 'defective'}_{index+1}.jpg")
            shutil.copyfile(row["path"], destination)
            examples.append({"image": destination.as_posix(), "source_image": row["path"],
                             "ground_truth": "Normal" if label == "0" else "Defective",
                             "prediction": predictor.predict(decode_image(destination.read_bytes()))})
    Path("reports/example_predictions.json").write_text(json.dumps(examples, indent=2))
    log = open("reports/server.log", "w", encoding="utf-8")
    command = [sys.executable, "-m", "uvicorn", "casting_inspector.api:app", "--host", "127.0.0.1",
               "--port", "8765", "--limit-concurrency", "16"]
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                              creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    evidence = {}
    try:
        with httpx.Client(base_url="http://127.0.0.1:8765", timeout=30) as client:
            for _ in range(120):
                try:
                    if client.get("/health/ready").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                if server.poll() is not None:
                    raise RuntimeError("Server exited; see reports/server.log")
                time.sleep(1)
            else:
                raise TimeoutError("Server did not become ready")
            evidence["health"] = client.get("/health/ready").json()
            evidence["requests"] = []
            for example in examples:
                with open(example["image"], "rb") as handle:
                    response = client.post("/predict", files={"file": ("example.jpg", handle, "image/jpeg")})
                response.raise_for_status()
                assert response.json()["predicted_class"] == example["prediction"]["predicted_class"]
                evidence["requests"].append({"image": example["image"], "status": response.status_code,
                                             "response": response.json()})
            invalid = client.post("/predict", files={"file": ("broken.jpg", b"invalid image", "image/jpeg")})
            assert invalid.status_code == 422
            evidence["invalid_image"] = {"status": invalid.status_code, "response": invalid.json()}
            timings, service_timings = [], []
            raw = Path(examples[0]["image"]).read_bytes()
            for i in range(55):
                start = time.perf_counter()
                response = client.post("/predict", files={"file": ("example.jpg", raw, "image/jpeg")})
                response.raise_for_status()
                if i >= 5:
                    timings.append((time.perf_counter() - start) * 1000)
                    service_timings.append(response.json()["latency_ms"])
            evidence["latency"] = {"samples": 50, "warmup_requests": 5,
                "http_round_trip_ms": {"p50": float(np.percentile(timings, 50)), "p95": float(np.percentile(timings, 95))},
                "decode_preprocess_infer_ms": {"p50": float(np.percentile(service_timings, 50)), "p95": float(np.percentile(service_timings, 95))},
                "platform": platform.platform(), "processor": platform.processor(),
                "torch_threads": 2, "device": "CPU", "concurrency": 1,
                "note": "Localhost sequential warm service; not production throughput or cold-start latency."}
            Path("reports/http_evidence.json").write_text(json.dumps(evidence, indent=2))
            Path("reports/openapi.json").write_text(json.dumps(client.get("/openapi.json").json(), indent=2))
    finally:
        server.terminate()
        server.wait(timeout=20)
        log.close()
    test = metrics["test"]
    cm = test["confusion_matrix_normal_defective"]
    report = f"""# Assessment results

All values below are measured from the saved model and split manifest.

| Test metric | Value |
|---|---:|
| Accuracy | {test['accuracy']:.4%} |
| Defect precision | {test['precision_defective']:.4%} |
| Defect recall | {test['recall_defective']:.4%} |
| Defect F1 | {test['f1_defective']:.6f} |
| Defect F2 | {test['f2_defective']:.6f} |
| ROC AUC | {test['roc_auc']:.6f} |
| Average precision | {test['average_precision']:.6f} |
| Majority-class accuracy baseline | {metrics['majority_test_accuracy']:.4%} |

Confusion matrix rows are true Normal/Defective; columns are predicted Normal/Defective:
`{cm}`. True negatives: {cm[0][0]}; false positives: {cm[0][1]}; false negatives: {cm[1][0]}; true positives: {cm[1][1]}.

Validation selected C={metrics['selection']['C']} and threshold={metrics['selection']['threshold']:.2f}.
The objective maximizes defect F2 subject to at least 95% precision on validation.
The constraint was met: {metrics['selection']['precision_constraint_met']}.
The threshold was locked before test evaluation. The final model is not refit on validation.

Training wall time: {metrics['training_wall_seconds']:.1f} seconds.
Bootstrap 95% intervals (conditional on fixed predictions and image-level sampling):
`{json.dumps(metrics['test_bootstrap_95_percent'])}`.

## Dataset audit

Split counts (0=Normal, 1=Defective): `{json.dumps(data['split_counts'])}`.
Unreadable images: {data['unreadable_images']}; excluded exact duplicate train images: {len(data['excluded'])}.
Unique pixel hashes in the test set: {data['test_unique_pixel_hashes']}.
Potential test/train near matches (dHash <=4): {len(data['potential_near_overlap_test'])}.
These are screening candidates; dHash similarity is not a reliable physical-part identifier.
The 512px collection is excluded to avoid mixing unverified overlapping sources.
The original 300px dataset was already augmented by its author. Exact deduplication
does not prove independent parts across splits. A new part/batch/time-separated
prospective dataset is required before a factory rollout.

## Live HTTP verification

Four real sample images returned HTTP 200 and matched local inference. A corrupt
image returned HTTP 422. Full responses and request IDs: `http_evidence.json`.

Warm CPU inference service, two Torch threads, 50 measured requests after five warmups:
HTTP p50 {evidence['latency']['http_round_trip_ms']['p50']:.1f} ms,
p95 {evidence['latency']['http_round_trip_ms']['p95']:.1f} ms.
Decode/preprocess/inference p50 {evidence['latency']['decode_preprocess_infer_ms']['p50']:.1f} ms,
p95 {evidence['latency']['decode_preprocess_infer_ms']['p95']:.1f} ms.
Hardware/platform: {platform.processor()}, {platform.platform()}.
This is a sequential localhost benchmark, not a throughput or SLA claim.
"""
    Path("reports/RESULTS.md").write_text(report, encoding="utf-8")
    styles = getSampleStyleSheet()
    story = [Paragraph("Casting Defect Inspection", styles["Title"]),
             Paragraph("Technical assessment - measured results", styles["Heading2"])]
    for paragraph in report.split("\n\n")[1:]:
        if paragraph.startswith("|") or paragraph.startswith("##"):
            continue
        story += [Paragraph(paragraph.replace("\n", " ").replace("`", "").replace("&", "&amp;").replace("<=", "&lt;="), styles["BodyText"]), Spacer(1, 8)]
    table = Table([["Metric", "Held-out test"], ["Accuracy", f"{test['accuracy']:.2%}"],
                   ["Defect precision", f"{test['precision_defective']:.2%}"],
                   ["Defect recall", f"{test['recall_defective']:.2%}"],
                   ["Defect F1", f"{test['f1_defective']:.4f}"]])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                               ("GRID", (0, 0), (-1, -1), 0.4, colors.grey)]))
    story += [table, Spacer(1, 12), Image("reports/figures/confusion_matrix.png", width=4.8*inch, height=4*inch)]
    SimpleDocTemplate("docs/assessment_report.pdf").build(story)
    print(json.dumps(evidence["latency"], indent=2))


if __name__ == "__main__":
    main()
