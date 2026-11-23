#!/usr/bin/env python3
"""Fit a kinematic correction for the LEFT arm from wrist-marker views in the chest camera. **Does not move the arm.**

2026-11-22 (Sota: 左手を使えるようにしよ). With the jaws tilted (handeye_near_20261122_161236) the wrist markers sat
21-29 mm away from where the left FK put them, while the 13:52 hand-eye poses fitted to 3 mm. Models tried, each
checked on held-out samples (every 3rd detection):
  joints      - zero offsets of J1..J7 (camera fixed)
  cam+joints  - joint offsets + a small camera correction
The markers' positions on link7 stay fixed (markers_link7_m of T_base_cam.json). Writes the chosen model as JSON
(joint_offsets_deg: added to the measured lerobot angles before FK; camera_delta: rotvec + t applied to T_base_cam).

    GRASP_ARM=left ~/work/GraspGenX/.venv/bin/python grasp/tools/fit_left_arm_corr.py <sample dirs ...> -o out.json
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rr

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.environ["GRASP_LEFT_CORR"] = "none"  # fits use the plain left model (ik/openarm_left_ik.py)
from ik.openarm_left_ik import OpenArmLeftIK  # noqa: E402


def detections(dirs, T):
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    D = []
    for g, d in enumerate(dirs):
        s = np.load(os.path.join(d, "samples.npz")); Q, K = s["q_deg"], s["K"]
        for k in range(len(Q)):
            col = cv2.imread(os.path.join(d, f"color_{k:03d}.jpg"))
            dep = np.load(os.path.join(d, f"depth_{k:03d}.npy")).astype(np.float32)
            cs, ids, _ = det.detectMarkers(cv2.cvtColor(col, cv2.COLOR_BGR2GRAY))
            if ids is None:
                continue
            for c, i in zip(cs, ids.ravel()):
                if i != 0:
                    continue
                c = c.reshape(4, 2); u, v = c.mean(0)
                zs = dep[int(c[:, 1].min()):int(c[:, 1].max()) + 1, int(c[:, 0].min()):int(c[:, 0].max()) + 1]; zs = zs[zs > 0]
                if len(zs) < 10:
                    continue
                z = float(np.median(zs))
                D.append({"g": g, "k": k, "q": Q[k].copy(), "p_cam": np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])})
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--t-base-cam", default=os.path.join(REPO, "grasp/config/T_base_cam.json"))
    a = ap.parse_args()
    ik = OpenArmLeftIK()
    J = json.load(open(a.t_base_cam)); T = np.array(J["T_base_cam"]); M = [np.array(m) for m in J["markers_link7_m"]]
    D = detections(a.dirs, T)
    print(f"[corr] {len(D)} marker detections: " + ", ".join(f"{os.path.basename(d.rstrip('/'))} {sum(x['g'] == g for x in D)}" for g, d in enumerate(a.dirs)))

    def unpack(x, mode):
        off = np.zeros(7); dR = np.eye(3); dt = np.zeros(3)
        if mode in ("joints", "cam+joints"):
            off = x[:7]
        if mode == "cam+joints":
            dR = Rr.from_rotvec(x[7:10]).as_matrix(); dt = x[10:13]
        return off, dR, dt

    def res(x, mode, idx):
        off, dR, dt = unpack(x, mode)
        out = []
        for i in idx:
            d = D[i]; pb = dR @ (T[:3, :3] @ d["p_cam"] + T[:3, 3]) + dt
            F = ik.fk(np.deg2rad(d["q"] + off)); pr = [F.rotation @ m + F.translation for m in M]
            out.append(min((pb - m for m in pr), key=np.linalg.norm))
        return np.concatenate(out)

    n = len(D); test = [i for i in range(n) if i % 3 == 2]; train = [i for i in range(n) if i % 3 != 2]
    nparam = {"none": 0, "joints": 7, "cam+joints": 13}
    results = {}
    for mode, k in nparam.items():
        def fit(idx):
            if not k:
                return np.zeros(0)
            reg = lambda x: np.r_[res(x, mode, idx), 0.002 * x[:7] / 1.0]  # weak pull to 0 (2 mm per deg)
            return least_squares(reg, np.zeros(k), loss="soft_l1", f_scale=0.01).x
        xt = fit(train)
        et = np.linalg.norm(res(xt, mode, test).reshape(-1, 3), axis=1) * 1000
        xa = fit(list(range(n)))
        ea = np.linalg.norm(res(xa, mode, list(range(n))).reshape(-1, 3), axis=1) * 1000
        g = np.array([d["g"] for d in D])
        per = {os.path.basename(d.rstrip('/')): round(float(np.median(ea[g == i])), 1) for i, d in enumerate(a.dirs) if (g == i).any()}
        results[mode] = {"x": xa, "test_median_mm": float(np.median(et)), "test_p90_mm": float(np.percentile(et, 90)),
                         "all_median_mm": float(np.median(ea)), "per_set_median_mm": per}
        off = xa[:7] if k else np.zeros(7)
        print(f"[corr] {mode:10s} held-out median {np.median(et):5.1f} mm  p90 {np.percentile(et, 90):5.1f}  | all {np.median(ea):4.1f}  per set {per}"
              + (f"  offsets deg {np.round(off, 2).tolist()}" if k else ""))
    best = min((m for m in results if m != "none"), key=lambda m: results[m]["test_median_mm"])
    off, dR, dt = unpack(results[best]["x"], best)
    out = {"model": best, "joint_offsets_deg": np.round(off, 3).tolist(),
           "camera_delta": {"rotvec": Rr.from_matrix(dR).as_rotvec().round(6).tolist(), "t_m": dt.round(5).tolist()},
           "held_out_median_mm": round(results[best]["test_median_mm"], 1), "held_out_p90_mm": round(results[best]["test_p90_mm"], 1),
           "uncorrected_held_out_median_mm": round(results["none"]["test_median_mm"], 1),
           "per_set_median_mm": results[best]["per_set_median_mm"], "uncorrected_per_set_median_mm": results["none"]["per_set_median_mm"],
           "detections": n, "sources": [os.path.basename(d.rstrip('/')) for d in a.dirs],
           "note": "actual joint angle used in FK = measured lerobot angle + joint_offsets_deg (left arm). fit_left_arm_corr.py"}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[corr] chose {best} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
