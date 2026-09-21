"""Isolated native CSF process: permits cancellation without blocking Qt/GIL."""

import argparse
import json
import sys
import traceback

import numpy as np

from .csf_filter import CSFGroundFilter, CSFOptions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("points")
    parser.add_argument("labels")
    parser.add_argument("options")
    parser.add_argument("scale", type=float)
    args = parser.parse_args()
    try:
        points = np.load(args.points, allow_pickle=False, mmap_mode="r")
        options = CSFOptions(**json.loads(args.options))
        labels = CSFGroundFilter(options).process(points, args.scale)
        np.save(args.labels, labels, allow_pickle=False)
        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
