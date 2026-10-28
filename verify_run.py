"""Check that a completed capture has images, matched rows, and motion."""

import csv
import json
import math
import struct
import sys
from pathlib import Path


def main(run):
    if (run / "stereo.csv").exists():
        from evaluation.benchmark_pipeline import benchmark
        result = benchmark(run)
        print(f"Verified {result['pairs']} stereo pairs ({result['decoded_pngs']} fully decoded PNGs); "
              f"camera distance: {result['camera_distance_m']:.2f} m")
        return
    metadata = json.loads((run / "metadata.json").read_text())
    with (run / "poses.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 100, f"Expected 100 poses, found {len(rows)}"
    for index, row in enumerate(rows):
        image = run / row["image"]
        with image.open("rb") as stream:
            header = stream.read(24)
        assert header[:8] == b"\x89PNG\r\n\x1a\n", f"Invalid PNG: {image}"
        assert struct.unpack(">II", header[16:24]) == (metadata["camera"]["width"], metadata["camera"]["height"])
        assert image.stem == f"{int(row['frame']):08d}"
        matrix = [float(row[f"t{r}{c}"]) for r in range(4) for c in range(4)]
        assert all(math.isfinite(value) for value in matrix)
        assert all(abs(a - b) < 1e-5 for a, b in zip(matrix[12:], [0, 0, 0, 1]))
        if index:
            assert int(row["frame"]) == int(rows[index - 1]["frame"]) + 1
            dt = float(row["timestamp_seconds"]) - float(rows[index - 1]["timestamp_seconds"])
            assert abs(dt - metadata["fixed_delta_seconds"]) < 1e-5
    displacement = math.dist(
        [float(rows[0][f"t{axis}3"]) for axis in range(3)],
        [float(rows[-1][f"t{axis}3"]) for axis in range(3)],
    )
    assert displacement > 0.1, f"Camera barely moved: {displacement:.3f} m"
    print(f"Verified {len(rows)} PNGs and poses; camera displacement: {displacement:.2f} m")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python verify_run.py runs/<run-directory>")
    main(Path(sys.argv[1]))
