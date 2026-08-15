"""Launch (or resume) the 50M-parameter FPFF-Z2G long run.

Usage:  python scripts/launch.py [--steps 6000]

Resumes automatically from results/fifty-m/latest.pt if it exists.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from zero.methods import fifty_m          # noqa: E402
from zero.train import train               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    args = ap.parse_args()

    cfg = fifty_m(args.steps)
    resume = os.path.join("results", cfg.name, "latest.pt")
    if os.path.exists(resume):
        cfg.resume_from = resume
        print(f"resuming from {resume}")
    train(cfg)


if __name__ == "__main__":
    main()
