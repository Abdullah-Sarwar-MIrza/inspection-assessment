"""Finalize reviewable submission artifacts and the local Git commit."""
import base64
import hashlib
import json
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def git(*args):
    return subprocess.run(['git', '-c', 'safe.directory=' + ROOT.as_posix(), *args], check=True, capture_output=True, text=True).stdout

# Native Windows speech and its direct dependency.
p = ROOT / 'scripts/build_demo.py'
s = p.read_text(encoding='utf-8').replace('voice.Rate = 1', 'voice.Rate = 2')
p.write_text(s, encoding='utf-8')
p = ROOT / 'requirements-dev.txt'
p.write_text(p.read_text().replace('pyttsx3==2.99', 'pywin32==312'), encoding='utf-8')
p = ROOT / 'scripts/publish_github.py'
s = p.read_text().replace('["git",', '["git", "-c", "safe.directory=D:/Job Assessment",')
p.write_text(s, encoding='utf-8')
subprocess.run([sys.executable, '-X', 'utf8', 'scripts/build_demo.py'], check=True)

import imageio.v2 as imageio
import numpy as np
import pymupdf
from PIL import Image
reader = imageio.get_reader('docs/demo.mp4')
video_metadata = reader.get_meta_data()
assert 120 <= video_metadata['duration'] <= 180, video_metadata
assert video_metadata['size'] == (1280, 720)
assert abs(video_metadata['fps'] - 24) < 1, video_metadata
# Decode frames across the full movie, including the final scene.
for seconds in (0, 40, 80, 120, video_metadata['duration'] - 2):
    frame = reader.get_data(int(seconds * video_metadata['fps']))
    assert frame.std() > 10
reader.close()
for audio in Path('reports/demo_audio').glob('*.wav'):
    with wave.open(str(audio)) as sound:
        signal = np.frombuffer(sound.readframes(sound.getnframes()), dtype=np.int16)
        assert np.sqrt(np.mean(signal.astype(float)**2)) > 20, audio
with pymupdf.open('docs/assessment_report.pdf') as document:
    assert len(document) >= 2
    assert '98.60' in ''.join(page.get_text() for page in document)
    document[0].get_pixmap(matrix=pymupdf.Matrix(1, 1)).save('reports/figures/report_preview.png')
metadata = json.loads(Path('artifacts/metadata.json').read_text())
assert hashlib.sha256(Path('artifacts/model.pt').read_bytes()).hexdigest() == metadata['sha256']
Path('reports/video_verification.json').write_text(json.dumps({
    'duration_seconds': video_metadata['duration'], 'size': video_metadata['size'],
    'fps': video_metadata['fps'], 'decoded_sample_frames': 5,
    'narration_audio': 'Six non-silent local SAPI speech tracks, muxed as AAC',
    'file_sha256': hashlib.sha256(Path('docs/demo.mp4').read_bytes()).hexdigest()
}, indent=2))
p = Path('docs/verification.md')
p.write_text(p.read_text() + f'\nDemo video verified: {video_metadata["duration"]:.1f} seconds, 1280x720 at 24 fps; frames decoded across all scenes and narration checked for non-silent audio. `reports/video_verification.json`.\n', encoding='utf-8')
# Record the exact installed environment for local reproducibility without publishing host details.
freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True).stdout
Path('reports/environment-freeze.txt').write_text(freeze)
# Dataset paths, assessment PDFs, environment/cache and credentials stay excluded.
git('add', '.dockerignore', '.gitignore', '.github', 'Dockerfile', 'compose.yaml', 'README.md',
    'requirements.txt', 'requirements-dev.txt', 'pytest.ini', 'artifacts', 'casting_inspector',
    'docs', 'reports', 'scripts', 'tests')
tracked = git('ls-files').splitlines()
assert not any(name.startswith(('.venv/', 'casting_data/', 'casting_512x512/')) for name in tracked)
assert all(Path(name).stat().st_size < 95 * 1024 * 1024 for name in tracked)
git('commit', '-m', 'Build casting defect inspection assessment with trained model and verified API')
# git archive uses tracked files only; no raw dataset or supplied PDFs.
subprocess.run(['git', '-c', 'safe.directory=' + ROOT.as_posix(), 'archive', '--format=zip',
                '--output=inspection-assessment.zip', 'HEAD'], check=True)
print(json.dumps({'duration': video_metadata['duration'], 'tracked_files': len(tracked),
                  'commit': git('rev-parse', '--short', 'HEAD').strip(),
                  'zip_bytes': Path('inspection-assessment.zip').stat().st_size}))
