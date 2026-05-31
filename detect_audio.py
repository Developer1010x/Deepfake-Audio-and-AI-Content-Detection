#!/usr/bin/env python3
"""
detect_audio.py
================

Headless command-line batch classifier for the audio deepfake model shipped
in this repository (``weights/Deepfake_audio.h5``).

Unlike ``final_gui.py`` (Tkinter) and ``Src/app.py`` (Flask), this script needs
no display and no web server. It scores a single audio file or every supported
audio file inside a directory and prints a human-readable report. Results can
optionally be exported to JSON or CSV, which makes it convenient for scripted
pipelines, batch evaluation, or CI checks.

The mel-spectrogram preprocessing is intentionally identical to the pipeline
used by the GUI/Flask apps so predictions stay consistent across the project.

Usage
-----
    # Score one file
    python detect_audio.py sample.wav

    # Score every audio file in a folder, write a CSV report
    python detect_audio.py ./clips --csv results.csv

    # Use a different model and emit JSON to stdout
    python detect_audio.py sample.mp3 --model weights/Deepfake_audio.h5 --json -

Exit codes
----------
    0  ran successfully
    1  bad arguments / no input files found
    2  model could not be loaded
"""

import argparse
import csv
import json
import os
import sys

# Preprocessing constants -- kept in sync with Src/app.py and final_gui.py.
SAMPLE_RATE = 16000
DURATION = 5
N_MELS = 128
MAX_TIME_STEPS = 109

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "weights", "Deepfake_audio.h5"
)

SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def find_audio_files(path):
    """Return a sorted list of supported audio files for a file or directory."""
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        matches = []
        for root, _dirs, files in os.walk(path):
            for name in files:
                if name.lower().endswith(SUPPORTED_EXTENSIONS):
                    matches.append(os.path.join(root, name))
        return sorted(matches)
    return []


def preprocess_audio(file_path):
    """Load an audio file and return a model-ready mel-spectrogram batch.

    This mirrors the preprocessing in ``Src/app.py`` so the CLI produces the
    same features the rest of the project expects.
    """
    import librosa
    import numpy as np

    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=N_MELS
    )
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    if mel_spectrogram.shape[1] < MAX_TIME_STEPS:
        mel_spectrogram = np.pad(
            mel_spectrogram,
            ((0, 0), (0, MAX_TIME_STEPS - mel_spectrogram.shape[1])),
            mode="constant",
        )
    else:
        mel_spectrogram = mel_spectrogram[:, :MAX_TIME_STEPS]

    return np.expand_dims(mel_spectrogram, axis=0)


def classify_file(model, file_path):
    """Run the model on one file and return a result dict.

    The model emits per-class probabilities. Consistent with ``Src/app.py``,
    class index 1 is treated as "Real" and class index 0 as "Deepfake".
    """
    import numpy as np

    features = preprocess_audio(file_path)
    probabilities = model.predict(features, verbose=0)[0]
    predicted_class = int(np.argmax(probabilities))
    label = "Real" if predicted_class == 1 else "Deepfake"
    confidence = float(np.max(probabilities))

    return {
        "file": file_path,
        "label": label,
        "confidence": round(confidence, 4),
        "probabilities": [round(float(p), 4) for p in probabilities],
    }


def load_audio_model(model_path):
    """Load the Keras audio model, exiting cleanly on failure."""
    if not os.path.exists(model_path):
        print(f"[error] model file not found: {model_path}", file=sys.stderr)
        sys.exit(2)
    try:
        from tensorflow.keras.models import load_model
    except ImportError as exc:  # pragma: no cover - environment dependent
        print(f"[error] TensorFlow/Keras is required: {exc}", file=sys.stderr)
        sys.exit(2)
    try:
        return load_model(model_path)
    except Exception as exc:  # pragma: no cover - model/IO dependent
        print(f"[error] could not load model '{model_path}': {exc}", file=sys.stderr)
        sys.exit(2)


def write_json(results, destination):
    payload = json.dumps(results, indent=2)
    if destination == "-":
        print(payload)
    else:
        with open(destination, "w", encoding="utf-8") as handle:
            handle.write(payload)
        print(f"[info] wrote JSON report to {destination}")


def write_csv(results, destination):
    fieldnames = ["file", "label", "confidence"]

    def rows():
        for item in results:
            yield {k: item[k] for k in fieldnames}

    if destination == "-":
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows())
    else:
        with open(destination, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows())
        print(f"[info] wrote CSV report to {destination}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Batch-classify audio files as Real or Deepfake.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        help="Path to an audio file or a directory of audio files.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to the Keras .h5 audio model.",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="Write results as JSON to PATH (use '-' for stdout).",
    )
    parser.add_argument(
        "--csv",
        metavar="PATH",
        help="Write results as CSV to PATH (use '-' for stdout).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the per-file human-readable summary.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    audio_files = find_audio_files(args.input)
    if not audio_files:
        print(
            f"[error] no supported audio files found at '{args.input}' "
            f"(supported: {', '.join(SUPPORTED_EXTENSIONS)})",
            file=sys.stderr,
        )
        return 1

    model = load_audio_model(args.model)

    results = []
    for file_path in audio_files:
        try:
            result = classify_file(model, file_path)
        except Exception as exc:  # keep processing the remaining files
            print(f"[warn] failed to classify '{file_path}': {exc}", file=sys.stderr)
            continue
        results.append(result)
        if not args.quiet:
            print(
                f"{result['label']:>8}  "
                f"({result['confidence'] * 100:5.1f}%)  "
                f"{result['file']}"
            )

    if not results:
        print("[error] no files were successfully classified.", file=sys.stderr)
        return 1

    if args.json:
        write_json(results, args.json)
    if args.csv:
        write_csv(results, args.csv)

    return 0


if __name__ == "__main__":
    sys.exit(main())
