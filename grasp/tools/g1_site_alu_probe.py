#!/usr/bin/env python3
"""Offline check of the aluminium finder on still images (e.g. G1 head-camera frames from the Takeei site). **No motion.**

2026-11-22 (Sota): run the perceive server's aluminium words on each image and apply decide.py's rules
(score >= ALU_THR, mask <= 8 % of the image, bar shape: decide.bar_like). Green = taken as an aluminium frame,
orange = scored as aluminium but not bar-shaped (sheet -> obstacle), red = low score.

    ~/work/perceive/.venv/bin/python grasp/tools/g1_site_alu_probe.py <dir of jpg> <out dir>
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decide import ALU_THR, ALU_WORDS, ask, bar_shape, bar_like  # noqa: E402


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    res = {}
    for f in sorted(glob.glob(src + "/*.jpg")):
        img = cv2.imread(f); H, W = img.shape[:2]
        r = ask(5558, {"bgr": img, "prompts": ALU_WORDS}, 120000)
        vis = img.copy(); rows = []
        for c in r.get("cands", []):
            m = np.asarray(c["mask"], bool)
            if m.sum() > 0.08 * H * W:
                continue
            a = float(c["alu"]); asp, fill = bar_shape(m); bar = bar_like(m)
            verdict = "alu" if (a >= ALU_THR and bar) else ("not_bar" if a >= ALU_THR else "low")
            col = {"alu": (0, 255, 0), "not_bar": (0, 165, 255), "low": (0, 0, 255)}[verdict]
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cs, -1, col, 2 if verdict != "low" else 1)
            ys, xs = np.nonzero(m)
            if verdict != "low":
                cv2.putText(vis, f"{a:.2f} a{asp:.1f} f{fill:.2f}", (int(xs.min()), max(14, int(ys.min()) - 4)), 0, 0.5, col, 2)
            rows.append({"label": c["label"], "alu": round(a, 3), "aspect": round(asp, 2), "fill": round(fill, 2),
                         "px": int(m.sum()), "verdict": verdict})
        cv2.imwrite(f"{out}/{os.path.basename(f)}", vis); res[os.path.basename(f)] = rows
        print(os.path.basename(f), [(x["verdict"], x["alu"], x["aspect"], x["fill"]) for x in rows if x["verdict"] != "low"], flush=True)
    json.dump(res, open(f"{out}/result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
