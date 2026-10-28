"""Validate stereo recording integrity and report measured capture/replay cost."""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from simulator.dataset import load_calibration, read_manifest, replay
from simulator.geometry import calibration


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def matrix(row):
    result = np.array([float(row[f"t{r}{c}"]) for r in range(4) for c in range(4)]).reshape(4, 4)
    require(np.isfinite(result).all() and np.allclose(result[3], [0, 0, 0, 1]), "Invalid homogeneous pose")
    rotation = result[:3, :3]
    require(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) and
            np.isclose(np.linalg.det(rotation), 1, atol=1e-5), "Pose is not a proper right-handed rigid transform")
    return result


def distribution(values):
    return {"median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95, method="linear"))}


def benchmark(run):
    run = Path(run)
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    require(metadata["status"] == "complete", "Recording is incomplete")
    config = metadata["config"]
    rows = read_manifest(run)
    require(len(rows) == config["frames"] and len(rows) > 1, "Unexpected stereo pair count")
    cal = load_calibration(run)
    require((cal["width"], cal["height"]) == (config["width"], config["height"]), "Calibration resolution mismatch")
    require(np.isclose(cal["baseline_m"], config["baseline_m"]), "Calibration baseline mismatch")
    expected = calibration(config)
    for field in ("K", "T_vehicle_left", "T_vehicle_right", "T_right_left", "distortion_coefficients"):
        require(np.allclose(cal[field], expected[field], atol=1e-8, rtol=0), f"Calibration {field} mismatch")
    require(sum(1 for _ in (run / "left").glob("*.png")) == len(rows) and
            sum(1 for _ in (run / "right").glob("*.png")) == len(rows), "Unexpected image count")
    times = np.array([float(row["timestamp_seconds"]) for row in rows])
    require(np.isfinite(times).all() and np.allclose(np.diff(times), config["fixed_delta_seconds"], atol=1e-5, rtol=0), "Invalid timestamp cadence")
    ground_truth = {side: read_csv(run / "ground_truth" / f"{side}.csv") for side in ("left", "right", "vehicle")}
    require(all(len(poses) == len(rows) for poses in ground_truth.values()), "Ground-truth count mismatch")
    timings = read_csv(run / "timings.csv")
    controls = read_csv(run / "controls.csv")
    require(len(timings) == len(rows) and len(controls) == len(rows), "Timing/control count mismatch")
    path = []
    for index, row in enumerate(rows):
        frame, timestamp = int(row["frame"]), float(row["timestamp_seconds"])
        require(int(row["left_frame"]) == frame == int(row["right_frame"]), "Stereo frame mismatch")
        require(abs(float(row["left_timestamp_seconds"]) - timestamp) < 1e-5 and
                abs(float(row["right_timestamp_seconds"]) - timestamp) < 1e-5, "Stereo timestamp mismatch")
        transforms = {}
        for side, poses in ground_truth.items():
            pose = poses[index]
            require(int(pose["frame"]) == frame and abs(float(pose["timestamp_seconds"]) - timestamp) < 1e-5, "Ground-truth association mismatch")
            transforms[side] = matrix(pose)
        for side in ("left", "right"):
            require(np.allclose(transforms[side], transforms["vehicle"] @ cal[f"T_vehicle_{side}"], atol=1e-3, rtol=0), "Camera pose does not match the vehicle mount")
        require(np.allclose(np.linalg.inv(transforms["right"]) @ transforms["left"], cal["T_right_left"], atol=1e-3, rtol=0), "Stereo extrinsic mismatch")
        require(int(timings[index]["frame"]) == frame == int(controls[index]["frame"]), "Timing/control frame mismatch")
        path.append(transforms["left"][:3, 3])
    distance = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    require(distance > 0.1, "Camera did not travel at least 0.1 m")
    durations = []
    fingerprint = hashlib.sha256()
    stream = iter(replay(run))
    for _ in rows:
        started = time.perf_counter()
        sample = next(stream)
        durations.append((time.perf_counter() - started) * 1000)
        fingerprint.update(f"{sample.frame},{sample.timestamp_seconds:.9f}\n".encode())
        fingerprint.update(sample.left.tobytes())
        fingerprint.update(sample.right.tobytes())
    capture = {}
    stage_totals = []
    for field in ("tick_ms", "sensor_wait_ms", "write_ms", "total_ms"):
        values = np.array([float(row[field]) for row in timings])
        require(np.isfinite(values).all() and (values >= 0).all(), "Invalid timing sample")
        capture[field] = distribution(values)
        stage_totals.append(values)
    require(np.allclose(stage_totals[3], stage_totals[0] + stage_totals[1] + stage_totals[2], atol=0.01, rtol=0),
            "Timing stages do not add up to the iteration total")
    wall_seconds = metadata["capture_wall_seconds"]
    require(np.isfinite(wall_seconds) and wall_seconds > 0, "Invalid capture wall duration")
    require(sum(stage_totals[3]) / 1000 <= wall_seconds + 0.01, "Iteration times exceed capture wall duration")
    return {
        "pairs": len(rows), "decoded_pngs": 2 * len(rows),
        "first_frame": int(rows[0]["frame"]), "last_frame": int(rows[-1]["frame"]),
        "timestamp_span_seconds": float(times[-1] - times[0]),
        "advanced_simulation_seconds": len(rows) * config["fixed_delta_seconds"],
        "capture_wall_seconds": wall_seconds, "capture_pairs_per_wall_second": len(rows) / wall_seconds,
        "camera_distance_m": distance, "capture_timings": capture,
        "replay_decode": {**distribution(durations), "pairs_per_decode_second": len(rows) / (sum(durations) / 1000)},
        "decoded_input_sha256": fingerprint.hexdigest(),
        "notes": "Warmup excluded; linear percentile interpolation. Replay decode excludes fingerprint hashing and has filesystem cache effects. No estimator accuracy measured.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = benchmark(args.run)
    path = args.run / "benchmark.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
