"""Replay an offline recording through the estimator-facing stereo interface."""

import argparse
from pathlib import Path

from simulator.dataset import load_calibration, replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    calibration = load_calibration(args.run)
    count = 0
    for sample in replay(args.run):
        # A future VO frontend receives only sample and calibration here.
        if count == 0:
            print(f"First frame {sample.frame}, timestamp {sample.timestamp_seconds:.9f}, RGB shape {sample.left.shape}")
        count += 1
    print(f"Replayed {count} stereo pairs; baseline {calibration['baseline_m']} m")


if __name__ == "__main__":
    main()
