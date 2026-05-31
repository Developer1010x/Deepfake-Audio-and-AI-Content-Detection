# RJPOLICE_HACK_1400_BitwiseOperators_8

# Deepfake Audio and AI Content Detection

An integrated project combining multiple repositories to detect **deepfake audio**,
**AI-generated images/videos**, and **AI-generated text** using deep learning,
signal processing, and face analysis techniques.

> Built for research, awareness, and educational purposes to counteract the
> rising threat of deepfake and synthetic media.

---

## Features

- **Audio Deepfake Detection** — identifies synthetic speech from mel-spectrogram
  features using a trained Keras CNN (`weights/Deepfake_audio.h5`).
- **Image & Video Deepfake Detection** — MesoInception-4 and ResNet CNN models
  detect manipulated faces (`weights/MesoInception_F2F.h5`).
- **AI-Generated Text Detection** — a BERT sequence classifier (served via the
  Flask app) estimates the probability that a passage was written by an AI.
- **Face Extraction** — `extractFaceFromVideo.py` uses MTCNN to crop faces from
  video frames for downstream analysis.
- **GUI Tools** — Tkinter desktop apps and a Flask web app to load media, run
  analysis, and read results.
- **Batch CLI** *(new)* — `detect_audio.py`, a headless command-line tool that
  scores one file or a whole folder of audio and exports JSON/CSV reports.

---

## Directory Structure

```
.
├── Src/
│   ├── app.py              # Flask web app (audio deepfake + AI-text detection)
│   ├── templates/          # index.html, result.html
│   └── uploads/            # runtime upload folder
├── weights/                # Pre-trained model weights (.h5)
│   ├── Deepfake_audio.h5
│   └── MesoInception_F2F.h5
├── ScreenShots/            # Example outputs and GUI visuals
├── detect_audio.py         # NEW: headless batch audio classifier (CLI)
├── final_gui.py            # Tkinter GUI: audio + MesoInception image detection
├── resnetgui.py            # Tkinter GUI: ResNet image classifier
├── partialGUI.py           # Tkinter chat-style detection UI prototype
├── extractFaceFromVideo.py # MTCNN face extraction from video
├── *.ipynb                 # Training/experiment notebooks for each model
└── requirements.txt        # Python dependencies
```

---

## Getting Started

### Installation

```bash
git clone https://github.com/Developer1010x/Deepfake-Audio-and-AI-Content-Detection.git
cd Deepfake-Audio-and-AI-Content-Detection
pip install -r requirements.txt
```

### Run the Flask web app

```bash
cd Src
python app.py
# then open http://127.0.0.1:5000
```

The web app expects `Deepfake_audio.h5` for audio and a BERT `model/` directory
for the text classifier.

### Run a desktop GUI

```bash
python final_gui.py     # audio + image (MesoInception) classifier
python resnetgui.py     # ResNet image classifier
```

> Note: the GUI scripts contain hard-coded Windows-style model paths
> (e.g. `D:\Desktop\hack\...`). Update those paths to point at the files under
> `weights/` before running.

---

## Batch Audio Detection (CLI)

`detect_audio.py` runs the audio deepfake model without any GUI or web server,
reusing the exact same mel-spectrogram preprocessing as the rest of the project.
It accepts a single file or a directory and can emit JSON/CSV reports.

```bash
# Score a single file (uses weights/Deepfake_audio.h5 by default)
python detect_audio.py sample.wav

# Recursively score every audio file in a folder and write a CSV report
python detect_audio.py ./clips --csv results.csv

# Point at a different model and print JSON to stdout
python detect_audio.py sample.mp3 --model weights/Deepfake_audio.h5 --json -

# Quiet mode (suppress per-file lines), JSON to file
python detect_audio.py ./clips --quiet --json results.json
```

Supported formats: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`.

Example output:

```
    Real  ( 92.4%)  clips/genuine_01.wav
Deepfake  ( 88.1%)  clips/synth_02.wav
```

Exit codes: `0` success, `1` bad input / nothing classified, `2` model load
failure.

---

## Models & Weights

| Task                 | Model            | Weights file                   |
|----------------------|------------------|--------------------------------|
| Audio deepfake       | Keras CNN        | `weights/Deepfake_audio.h5`    |
| Image deepfake       | MesoInception-4  | `weights/MesoInception_F2F.h5` |
| Image deepfake (alt) | ResNet           | `resnet.h5` (not bundled)      |
| AI-text detection    | BERT classifier  | `model/` (not bundled)         |

---

## Disclaimer

These models are provided for educational and research use. Predictions are
probabilistic and may be wrong; do not rely on them as sole evidence for any
high-stakes decision.
