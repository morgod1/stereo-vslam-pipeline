"""Estimator input: RGB stereo pairs, frame IDs, time and calibration only."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class StereoSample:
    frame: int
    timestamp_seconds: float
    left: np.ndarray
    right: np.ndarray


def read_manifest(run):
    with (Path(run) / "stereo.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_calibration(run):
    return json.loads((Path(run) / "calibration.json").read_text(encoding="utf-8"))


def replay(run):
    """Decode each PNG fully, in recorded order; never open ground-truth files."""
    run = Path(run)
    cal = load_calibration(run)
    previous_frame = None
    previous_time = None
    for row in read_manifest(run):
        frame, timestamp = int(row["frame"]), float(row["timestamp_seconds"])
        if not np.isfinite(timestamp) or int(row["left_frame"]) != frame or int(row["right_frame"]) != frame:
            raise ValueError("Stereo frame mismatch or invalid time")
        if any(abs(float(row[f"{side}_timestamp_seconds"]) - timestamp) > 1e-5 or
               not np.isfinite(float(row[f"{side}_timestamp_seconds"])) for side in ("left", "right")):
            raise ValueError("Stereo timestamp mismatch")
        if previous_frame is not None and (frame != previous_frame + 1 or
                abs(timestamp - previous_time - cal["sample_interval_seconds"]) > 1e-5):
            raise ValueError("Stereo input frames must be consecutive at the calibrated cadence")
        decoded = []
        for side in ("left", "right"):
            path = run / row[side]
            if Path(row[side]) != Path(side) / f"{frame:08d}.png":
                raise ValueError(f"Image filename does not match frame {frame}: {path}")
            with Image.open(path) as image:
                if image.format != "PNG" or image.size != (cal["width"], cal["height"]):
                    raise ValueError(f"Invalid image format or size: {path}")
                decoded.append(np.array(image.convert("RGB")))
        yield StereoSample(frame, timestamp, *decoded)
        previous_frame, previous_time = frame, timestamp
