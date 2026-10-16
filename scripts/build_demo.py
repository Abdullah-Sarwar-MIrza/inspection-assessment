"""Create a 2-3 minute narrated demo from recorded live HTTP evidence.

Uses local Windows text-to-speech, not a fabricated human recording. Frames
contain actual source images, measured results and recorded service responses.
"""
import json
import subprocess
import textwrap
import wave
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1280, 720
BG, FG, MUTED, ACCENT = "#0b1220", "#f1f5f9", "#aabbd1", "#38bdf8"


def font(size, mono=False):
    windows = Path("C:/Windows/Fonts")
    candidates = [windows / ("consola.ttf" if mono else "segoeui.ttf"),
                  Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default(size=size)


def slide(title, subtitle, index):
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 8), fill=ACCENT)
    draw.text((55, 35), "CASTING DEFECT INSPECTION", font=font(19), fill=ACCENT)
    draw.text((55, 88), title, font=font(39), fill=FG)
    draw.text((55, 150), subtitle, font=font(21), fill=MUTED)
    draw.text((55, 670), "Technical assessment | Automated narration | Recorded API evidence", font=font(17), fill=MUTED)
    draw.text((1170, 670), f"{index}/6", font=font(19), fill=ACCENT)
    return image, draw


def lines(draw, entries, x=60, y=230, spacing=65, size=27):
    for entry in entries:
        draw.text((x, y), entry, font=font(size), fill=FG)
        y += spacing


def main():
    metrics = json.loads(Path("reports/metrics.json").read_text())
    audit = json.loads(Path("reports/dataset_analysis.json").read_text())
    http = json.loads(Path("reports/http_evidence.json").read_text())
    test = metrics["test"]
    cm = test["confusion_matrix_normal_defective"]
    threshold = metrics["selection"]["threshold"]
    scenes = []
    image, draw = slide("A working visual inspection service", "Normal or Defective, from a single uploaded casting image", 1)
    lines(draw, ["Dataset audit + reproducible training", "Transfer learning + held-out evaluation",
                 "Validated HTTP API + deployable container", "Code, trained weights, report and this demonstration"], size=29)
    scenes.append((image, "This assessment implements a visual inspection system for casting products. It takes one image and returns Normal or Defective, together with a confidence score and model version. The submission contains reproducible training code, actual trained weights, held out evaluation, error analysis, and a working HTTP API. This demonstration uses recorded responses from real requests to that service, with automated narration."))
    image, draw = slide("Audit first; keep the test set held out", "Ravirajsinh Dabhi / Kaggle casting dataset | CC BY-NC-ND 4.0", 2)
    counts = audit["split_counts"]
    lines(draw, [f"Train: {sum(counts['train'].values()):,} | Validation: {sum(counts['validation'].values()):,} | Test: 715",
                 f"Exact duplicate training images excluded: {len(audit['excluded'])}",
                 "Stratified split, fixed seed, train-only augmentation",
                 "Limitation: pre-augmented data lacks physical-part IDs"], size=28)
    scenes.append((image, "I preserved the official test set and split the remaining training pool into training and validation with stratified classes and a fixed seed. Every image is decoded and hashed. Exact training duplicates and exact test overlaps are excluded. Augmentation happens only after splitting. The separate higher resolution source collection is excluded because its overlap is uncertain. The publisher already augmented this dataset, so these image level results cannot prove generalization to independent physical parts."))
    image, draw = slide("Transfer learning with an inspectable head", "Frozen ResNet-18 + balanced, regularized logistic classifier", 3)
    lines(draw, ["RGB -> full image resize 160x160 -> normalization",
                 "ImageNet ResNet-18 -> 512 pooled features",
                 "Training-only standardization -> weighted linear head",
                 f"Validation selects regularization and threshold: {threshold:.2f}",
                 "Objective: defect F2, with at least 95% precision"], spacing=60, size=27)
    scenes.append((image, "I chose a frozen ImageNet ResNet eighteen feature extractor to keep training practical on this CPU and limit overfitting. Images are resized without cropping, so boundary defects remain visible. A balanced logistic head handles the modest class imbalance. Standardization sees training features only. Validation selects regularization and the decision threshold by maximizing defect F two, subject to a ninety five percent precision floor. This favors catching defective parts while controlling false rejects."))
    image, draw = slide("Measured held-out performance", "Threshold locked before evaluating the 715 test images", 4)
    lines(draw, [f"Accuracy: {test['accuracy']:.2%}", f"Defect precision: {test['precision_defective']:.2%}",
                 f"Defect recall: {test['recall_defective']:.2%}", f"Defect F1: {test['f1_defective']:.4f}",
                 f"False positives: {cm[0][1]}   |   False negatives: {cm[1][0]}"], spacing=63, size=28)
    figure = Image.open("reports/figures/confusion_matrix.png").convert("RGB")
    figure.thumbnail((550, 440))
    image.paste(figure, (680, 205))
    scenes.append((image, f"On the locked test set, accuracy is {test['accuracy']*100:.2f} percent, defect precision is {test['precision_defective']*100:.2f} percent, and recall is {test['recall_defective']*100:.2f} percent. There are {cm[0][1]} false positives and {cm[1][0]} false negatives. A false positive causes unnecessary rejection or review. A false negative lets a defective part pass. The report lists every prediction and observed error. Tiny defects, ambiguous labels, and camera changes remain weaknesses; test feedback was not used to retune the model."))
    image, draw = slide("Real HTTP request and response", "POST /predict | multipart JPEG upload | HTTP 200", 5)
    request = http["requests"][2]
    photo = Image.open(request["image"]).convert("RGB").resize((350, 350))
    image.paste(photo, (60, 230))
    result = request["response"]
    summary = {key: result[key] for key in ("predicted_class", "confidence", "defect_probability", "threshold", "model_version")}
    y = 230
    for line in json.dumps(summary, indent=2).splitlines():
        draw.text((465, y), line, font=font(23, mono=True), fill=FG)
        y += 37
    draw.text((465, 565), "Corrupt image -> HTTP 422", font=font(26), fill=ACCENT)
    scenes.append((image, "Here is a real uploaded defective image and the response captured from the running API. The response includes its class, confidence, defect score, threshold, version, request identifier, and elapsed time. The confidence is an uncalibrated model score, not a guarantee. A corrupt image returns HTTP four twenty two. The service also validates media type, upload size, dimensions, and decoded content. Shared preprocessing keeps command line and API predictions consistent."))
    image, draw = slide("Deployment, observability and next steps", "docker compose up --build -d", 6)
    lines(draw, ["Non-root container + read-only filesystem",
                 "Artifact checksum, startup warmup, health probes",
                 f"Warm CPU HTTP p95: {http['latency']['http_round_trip_ms']['p95']:.1f} ms (sequential localhost)",
                 "Request IDs, status and timing logs; no stored uploads",
                 "Next: independent batches, review policy and drift monitoring"], spacing=61, size=26)
    scenes.append((image, "Deployment uses a non root container, a read only filesystem, bounded resources, startup integrity checks, and readiness probes. Request logs record identifiers, status, and latency without retaining uploaded images. The report measures warm sequential CPU latency rather than claiming production throughput. Before a factory rollout, collect independent batches, validate lighting and motion conditions, add authenticated transport and operator review, and agree acceptance criteria. The README explains setup, trade offs, reproduction, limitations, and maintenance."))
    frame_dir = Path("reports/demo_frames"); frame_dir.mkdir(exist_ok=True)
    audio_dir = Path("reports/demo_audio"); audio_dir.mkdir(exist_ok=True)
    script = ["# Demo narration\n\nThis is an automatically narrated walkthrough using actual recorded HTTP responses, not a human screen recording.\n"]
    import win32com.client
    voice = win32com.client.Dispatch("SAPI.SpVoice")
    voice.Rate = 2
    for i, (image, narration) in enumerate(scenes):
        image.save(frame_dir / f"{i}.png")
        audio_path = audio_dir / f"{i}.wav"
        stream = win32com.client.Dispatch("SAPI.SpFileStream")
        stream.Open(str(audio_path.resolve()), 3)
        voice.AudioOutputStream = stream
        voice.Speak(narration)
        stream.Close()
        script.append(f"## Scene {i+1}\n\n{narration}\n")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    durations, segments = [], []
    for i in range(len(scenes)):
        audio_path = audio_dir / f"{i}.wav"
        with wave.open(str(audio_path)) as source:
            duration = source.getnframes() / source.getframerate() + 1.0
        durations.append(duration)
        output = frame_dir / f"scene_{i}.mp4"
        subprocess.run([ffmpeg, "-y", "-loop", "1", "-framerate", "1", "-i", str(frame_dir / f"{i}.png"),
            "-i", str(audio_path), "-t", str(duration), "-vf", "fps=24", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "24", "-pix_fmt", "yuv420p", "-c:a", "aac", "-af", "apad", str(output)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        segments.append(f"file '{output.resolve().as_posix()}'")
    concat = frame_dir / "concat.txt"; concat.write_text("\n".join(segments))
    subprocess.run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
                    "-c", "copy", "-movflags", "+faststart", "docs/demo.mp4"],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    duration = sum(durations)
    if not 120 <= duration <= 180:
        raise ValueError(f"Demo duration {duration:.1f}s outside 2-3 minutes; adjust narration rate")
    script.append(f"\nDuration: {duration:.1f} seconds. Rebuild on Windows with `python scripts/build_demo.py` after `python scripts/validate_and_report.py`.\n")
    Path("docs/demo_script.md").write_text("\n".join(script), encoding="utf-8")
    print(f"Created docs/demo.mp4: {duration:.1f} seconds")


if __name__ == "__main__":
    main()
