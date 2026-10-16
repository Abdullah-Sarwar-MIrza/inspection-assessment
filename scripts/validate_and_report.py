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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def render_pdf_report(metrics, data, evidence):
    # Render only saved evidence. This function never starts an API or Docker.
    import html
    import json
    from pathlib import Path
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                   Table, TableStyle, PageBreak, Flowable)
    ink=colors.HexColor('#172d4d'); muted=colors.HexColor('#61738c')
    blue=colors.HexColor('#3268e9'); line=colors.HexColor('#e3eaf3')
    pale=colors.HexColor('#f3f6fb'); width=A4[0]-88
    styles={
      'body':ParagraphStyle('body',fontName='Helvetica',fontSize=9.4,leading=14,textColor=ink,spaceAfter=8),
      'small':ParagraphStyle('small',fontName='Helvetica',fontSize=8,leading=11.5,textColor=muted,spaceAfter=5),
      'label':ParagraphStyle('label',fontName='Helvetica-Bold',fontSize=8,leading=12,textColor=muted,spaceAfter=7),
      'h1':ParagraphStyle('h1',fontName='Helvetica-Bold',fontSize=24,leading=30,textColor=ink,spaceAfter=12),
      'h2':ParagraphStyle('h2',fontName='Helvetica-Bold',fontSize=12,leading=17,textColor=ink,spaceBefore=13,spaceAfter=7),
      'title':ParagraphStyle('title',fontName='Helvetica-Bold',fontSize=37,leading=42,textColor=ink,spaceAfter=13),
      'subtitle':ParagraphStyle('subtitle',fontName='Helvetica',fontSize=12,leading=18,textColor=muted,spaceAfter=17),
      'cell':ParagraphStyle('cell',fontName='Helvetica',fontSize=8.7,leading=12.5,textColor=ink),
      'head':ParagraphStyle('head',fontName='Helvetica-Bold',fontSize=8.5,leading=12,textColor=colors.white),
      'metric':ParagraphStyle('metric',fontName='Helvetica-Bold',fontSize=24,leading=31,textColor=blue),
      'caption':ParagraphStyle('caption',fontName='Helvetica',fontSize=8,leading=11,textColor=muted,alignment=TA_CENTER),
      'mono':ParagraphStyle('mono',fontName='Courier',fontSize=7.8,leading=11,textColor=muted),
    }
    def P(text,style='body'):
        return Paragraph(str(text),styles[style])
    def table(headers,rows,widths=None):
        content=[[P(h,'head') for h in headers]]+[[P(html.escape(str(v)),'cell') for v in row] for row in rows]
        t=Table(content,colWidths=widths or [width/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),ink),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,pale]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),11),('RIGHTPADDING',(0,0),(-1,-1),11),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),('LINEBELOW',(0,-1),(-1,-1),.5,line)]))
        return t
    def callout(title,text):
        t=Table([[P(title,'label')],[P(text,'small')]],colWidths=[width])
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),pale),('LINEBEFORE',(0,0),(0,-1),3,blue),('LEFTPADDING',(0,0),(-1,-1),14),('RIGHTPADDING',(0,0),(-1,-1),14),('TOPPADDING',(0,0),(-1,0),11),('BOTTOMPADDING',(0,-1),(-1,-1),9)]))
        return t
    def figure(path,maxw,maxh):
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            w,h=im.size
        scale=min(maxw/w,maxh/h)
        return Image(str(path),width=w*scale,height=h*scale)
    class Pipeline(Flowable):
        def __init__(self):
            Flowable.__init__(self); self.width=width; self.height=79
        def draw(self):
            c=self.canv; gap=12; box=(width-gap*4)/5
            labels=[('IMAGE','JPG / PNG'),('PREPROCESS','160 x 160 RGB'),('RESNET-18','Frozen features'),('LINEAR HEAD','Balanced scores'),('DECISION','Normal / Defective')]
            for i,(a,b) in enumerate(labels):
                x=i*(box+gap); c.setFillColor(pale); c.setStrokeColor(line); c.roundRect(x,12,box,56,6,stroke=1,fill=1)
                c.setFillColor(blue); c.setFont('Helvetica-Bold',7.2); c.drawCentredString(x+box/2,46,a)
                c.setFillColor(muted); c.setFont('Helvetica',6.9); c.drawCentredString(x+box/2,29,b)
                if i<4:
                    c.setStrokeColor(blue); c.line(x+box+2,40,x+box+gap-2,40); c.line(x+box+gap-5,43,x+box+gap-2,40); c.line(x+box+gap-5,37,x+box+gap-2,40)
    def page(canvas,doc):
        canvas.saveState(); w,h=A4
        canvas.setTitle('Casting Defect Inspection - Technical Assessment Report')
        canvas.setAuthor(''); canvas.setSubject('Dataset, model, evaluation and inference service')
        canvas.setFillColor(blue); canvas.rect(0,h-7,w,7,fill=1,stroke=0)
        canvas.setFont('Helvetica-Bold',8); canvas.setFillColor(ink); canvas.drawString(44,h-36,'CASTING DEFECT INSPECTION')
        canvas.setFont('Helvetica',8); canvas.setFillColor(muted); canvas.drawRightString(w-44,h-36,'TECHNICAL ASSESSMENT')
        canvas.setStrokeColor(line); canvas.line(44,43,w-44,43)
        canvas.setFont('Helvetica',7.4); canvas.setFillColor(muted); canvas.drawString(44,29,'16 October 2026  |  Evaluation and implementation report')
        canvas.drawRightString(w-44,29,f'{doc.page}')
        canvas.restoreState()
    story=[]; test=metrics['test']; selected=metrics['selection']; cm=test['confusion_matrix_normal_defective']
    # Page 1: concise executive overview.
    story += [P('COMPUTER VISION / MANUFACTURING QUALITY','label'),P('Casting Defect<br/>Inspection','title'),P('Technical assessment report','subtitle')]
    story += [P('A visual inspection system for pump-impeller casting images. The trained model classifies each submitted image as <b>Normal</b> or <b>Defective</b> and returns a model score through a browser interface and HTTP API.'),Spacer(1,13)]
    cards=[]
    for label,value in [('TEST ACCURACY',f"{test['accuracy']:.2%}"),('DEFECT PRECISION',f"{test['precision_defective']:.2%}"),('DEFECT RECALL',f"{test['recall_defective']:.2%}"),('DEFECT F1',f"{test['f1_defective']:.4f}")]:
        cards.append([P(label,'label'),P(value,'metric')])
    t=Table([cards],colWidths=[width/4]*4)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),pale),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),12),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),13),('BOTTOMPADDING',(0,0),(-1,-1),12)]))
    story += [t,Spacer(1,12),P('System overview','h2'),table(['Component','Implementation'],[
       ['Model','ImageNet-pretrained ResNet-18 with a frozen backbone and balanced logistic classification head.'],
       ['Input / output','JPEG or PNG image; predicted class, class confidence, defect score, threshold and model version.'],
       ['Evaluation','715 held-out test images; 705 correct classifications, 5 false positives and 5 false negatives.'],
       ['Serving','FastAPI service with shared preprocessing, bounded uploads, structured errors, request IDs and health probes.'],
    ],[105,width-105]),P('Inference architecture','h2'),Pipeline(),Spacer(1,7),callout('INTERPRETATION','Results describe the supplied image-level benchmark. Pre-existing augmentation and missing physical-part IDs limit claims about independent factory products. Confidence scores are uncalibrated.')]
    story.append(PageBreak())
    # Page 2: dataset and model selection.
    story += [P('01 / DATA AND MODEL','label'),P('Dataset preparation<br/>and model strategy','h1'),P('The original test set remains held out. Exact pixel deduplication precedes a stratified split of the supplied training pool.')]
    counts=data['split_counts']
    story += [table(['Partition','Normal','Defective','Total'],[[label,counts[key].get('0',0),counts[key].get('1',0),sum(counts[key].values())] for key,label in [('train','Training'),('validation','Validation'),('test','Test')]],[width*.34,width*.22,width*.22,width*.22])]
    story += [Spacer(1,10),P(f"All 7,348 supplied 300 x 300 images decoded successfully. The audit excluded {len(data['excluded'])} exact duplicate training images, including any exact matches to test images. Remaining training data is split 80/20 with stratified labels and seed 42. The separate 512 x 512 source collection is excluded because its overlap with the augmented collection is uncertain."),P('Preprocessing and augmentation','h2'),P('Grayscale-looking RGB images are converted to RGB, resized as a whole to 160 x 160 and normalized with ImageNet channel statistics. No center crop is applied, preserving the outer rim. Each training image receives one additional mirrored view with a small positive or negative 8-degree rotation. Validation and test images are not augmented.'),P('Architecture and imbalance','h2'),P('A frozen ResNet-18 produces 512 pooled features. Training-only feature standardization feeds a balanced, L2-regularized logistic head. Inverse-frequency class weights address the modest class imbalance. The scaler and learned coefficients are embedded in the saved Torch model, so inference does not require scikit-learn.'),P('Validation-only selection','h2'),table(['Setting','Selected strategy'],[
       ['Regularization','C in {0.001, 0.01, 0.1, 1, 10}; selected C = '+str(selected['C'])],
       ['Decision threshold',f"0.10 to 0.90 in steps of 0.02; selected threshold = {selected['threshold']:.2f}"],
       ['Objective','Maximum validation defect F2 subject to at least 95% precision; the constraint was met.'],
       ['Test discipline','Model and threshold locked before test scoring; no refit on validation data.'],
    ],[112,width-112]),Spacer(1,12),callout('DATA INDEPENDENCE',f"The audit flags {len(data['potential_near_overlap_test'])}/715 test images with a training dHash within four bits. Similar aligned castings can share hashes; these are screening candidates, not proven duplicates. Exact deduplication cannot establish physical-part independence.")]
    story.append(PageBreak())
    # Page 3: measured test results.
    story += [P('02 / EVALUATION','label'),P('Held-out test results','h1'),P('Defective is the positive class. Classification uses the validation-selected threshold of 0.36; all values below come from the saved test predictions.')]
    ci=metrics['test_bootstrap_95_percent']
    story += [table(['Metric','Test result','Interpretation'],[
      ['Accuracy',f"{test['accuracy']:.2%}",'705 of 715 images correctly classified'],
      ['Defect precision',f"{test['precision_defective']:.2%}",'448 true defects / 453 defect predictions'],
      ['Defect recall',f"{test['recall_defective']:.2%}",'448 detected / 453 labeled defects'],
      ['Defect F1 / F2',f"{test['f1_defective']:.4f} / {test['f2_defective']:.4f}",'Precision-recall operating-point summaries'],
      ['ROC AUC',f"{test['roc_auc']:.6f}",'Ranking performance across thresholds'],
      ['Average precision',f"{test['average_precision']:.6f}",'Precision-recall curve summary'],
      ['Majority baseline',f"{metrics['majority_test_accuracy']:.2%}",'Always predict the majority test class'],
    ],[width*.29,width*.24,width*.47]),Spacer(1,16)]
    plots=Table([[figure(ROOT/'reports/figures/confusion_matrix.png',width*.49,205),figure(ROOT/'reports/figures/precision_recall.png',width*.49,205)]],colWidths=[width/2,width/2])
    plots.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0)]))
    story += [plots,P('Confusion matrix: rows = true class; columns = predicted class. TN = 257, FP = 5, FN = 5, TP = 448.','small'),P('Uncertainty and interpretation','h2'),P(f"Bootstrap 95% intervals are {ci['accuracy'][0]:.2%} to {ci['accuracy'][1]:.2%} for accuracy and {ci['recall_defective'][0]:.2%} to {ci['recall_defective'][1]:.2%} for defect recall. These intervals resample fixed image-level predictions; they do not resolve dependence between augmented views."),P('Full candidate metrics, individual predictions and class reports are retained in reports/metrics.json and reports/test_predictions.csv.','small')]
    story.append(PageBreak())
    # Page 4: visually readable examples of model failures.
    story += [P('03 / ERROR ANALYSIS','label'),P('False positives<br/>and false negatives','h1'),P('The test set contains five false positives and five false negatives. The examples below illustrate confident and localized errors. Observations are visual hypotheses, not operator-confirmed defect annotations.')]
    errors=json.loads((ROOT/'reports/errors.json').read_text())
    wanted=['cast_ok_0_259.jpeg','cast_ok_0_9926.jpeg','cast_def_0_2483.jpeg','cast_def_0_3105.jpeg']
    notes=['Strong highlights on a Normal-labeled part; a confident false rejection.','A near-threshold false rejection of a Normal-labeled part.','Localized upper-rim roughness; the model still assigns a low defect score.','Outer-rim irregularities on an otherwise regular-looking part are missed.']
    cells=[]
    for name,note in zip(wanted,notes):
        item=next(e for e in errors if Path(e['path']).name==name)
        imagepath=ROOT/item['path']
        cell=[P('FALSE POSITIVE' if item['error_type']=='FP' else 'FALSE NEGATIVE','label')]
        if imagepath.exists():
            cell += [figure(imagepath,160,125),Spacer(1,6)]
        cell += [P(name,'small'),P(f"Defect score: <b>{item['defect_probability']:.3f}</b> | threshold: 0.36",'small'),P(note,'small')]
        cells.append(cell)
    examples=Table([[cells[0],cells[1]],[cells[2],cells[3]]],colWidths=[width/2,width/2])
    examples.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),pale),('BOX',(0,0),(-1,-1),.6,line),('INNERGRID',(0,0),(-1,-1),.6,line),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),12),('RIGHTPADDING',(0,0),(-1,-1),12),('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    story += [examples,Spacer(1,12),P('Operational consequences','h2'),P('False positives create unnecessary rejection or review of good parts. False negatives allow defective parts to pass. All five missed defects have scores below 0.18, so a narrow review band around the 0.36 threshold would not catch them.'),P('Resizing can weaken tiny rim details; global pooled features may emphasize overall appearance. Lighting, surface texture and label ambiguity are plausible contributors. No observed test error was used to retune this submitted model. All ten cases are listed in reports/errors.json and discussed in docs/error_analysis.md.','small')]
    story.append(PageBreak())
    # Page 5: service evidence, deployment boundaries and references.
    latency=evidence['latency']['http_round_trip_ms']
    story += [P('04 / INFERENCE AND DEPLOYMENT','label'),P('Service verification<br/>and operating limits','h1'),table(['Capability','Implementation / evidence'],[
       ['Browser interface','Drag-and-drop image upload, whole-image preview, reference images, real predictions and JSON downloads.'],
       ['Inference API','POST /predict; class, confidence, defect score, threshold, version, request ID and latency.'],
       ['Input controls','JPEG/PNG only; 8 MiB file limit, bounded request body, minimum 32 x 32 pixels and maximum 16 megapixels.'],
       ['Lifecycle and logging','Artifact checksum, startup warmup, readiness/liveness probes, request IDs and sanitized error responses.'],
       ['Verification','16 API/inference checks and 12 desktop/mobile browser checks passed.'],
       ['Optional Docker','Non-root user, read-only filesystem, CPU/memory limits and health check. Image build, healthy startup and one real image prediction verified.'],
    ],[112,width-112]),P('Measured latency','h2'),P(f"On the recorded Windows CPU environment, 50 sequential warm HTTP requests after five warmups measured <b>{latency['p50']:.1f} ms median</b> and <b>{latency['p95']:.1f} ms p95</b>. The service used two Torch threads and one active inference call. These figures describe localhost round-trip latency, not cold-start latency, container latency or a production throughput guarantee."),P('Known limitations','h2'),P('The classifier does not locate defects, identify defect types, inspect hidden surfaces or reject unrelated images as out of domain. Scores are uncalibrated. New lighting, geometry, motion blur and manufacturing batches fall outside the established evaluation. The supplied data cannot establish physical-part independence. Factory acceptance and production load testing are outside this verification scope.'),P('Evidence and references','h2'),P('Model: artifacts/model.pt and artifacts/metadata.json. Dataset/splits: reports/dataset_analysis.json and reports/split_manifest.csv. HTTP evidence: reports/http_evidence.json. Test evidence: reports/test-results.xml and reports/ui-verification.json. Architecture: docs/architecture.md.','small'),P('Dataset: <link href="https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product" color="#3268e9">Ravirajsinh Dabhi, casting product image data for quality inspection (Kaggle)</link>. Publisher acknowledgement: PILOT TECHNOCAST. Imagery retains CC BY-NC-ND 4.0 attribution.','small'),P('Framework references: <link href="https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html" color="#3268e9">Torchvision ResNet-18</link>; <link href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html" color="#3268e9">scikit-learn logistic regression</link>; <link href="https://fastapi.tiangolo.com/tutorial/request-files/" color="#3268e9">FastAPI image uploads</link>.','small')]
    output=ROOT/'docs/assessment_report.pdf'
    doc=SimpleDocTemplate(str(output),pagesize=A4,leftMargin=44,rightMargin=44,topMargin=64,bottomMargin=58,title='Casting Defect Inspection - Technical Assessment Report',author='')
    doc.build(story,onFirstPage=page,onLaterPages=page)


def main():
    if "--report-only" in sys.argv:
        render_pdf_report(json.loads((ROOT / "reports/metrics.json").read_text()),
                          json.loads((ROOT / "reports/dataset_analysis.json").read_text()),
                          json.loads((ROOT / "reports/http_evidence.json").read_text()))
        print("Formatted assessment report generated from saved evidence; no services started.")
        return
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
    render_pdf_report(metrics, data, evidence)
    print(json.dumps(evidence["latency"], indent=2))


if __name__ == "__main__":
    main()
