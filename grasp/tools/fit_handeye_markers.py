#!/usr/bin/env python3
"""Fit the chest camera pose (T_base_cam) from handeye_near.py samples with the two id0 ArUco markers on the left arm.
**Does not move the arm.**

2026-11-21: the camera was moved twice in one evening (18:00 lowered, 20:30 raised to Kitakyushu's ~0.7 m). The two
markers stay where they are on link7, so their offsets from the previous fit (markers_link7_m in the old
T_base_cam.json) are kept fixed and only the camera pose is fitted (6 numbers), each detection assigned to the marker
that fits it best (a frame showing both uses both). Marker centre + median depth inside the marker; soft_l1.
--free-markers also refits the two offsets (when the markers were re-stuck).

    python grasp/tools/fit_handeye_markers.py <samples dir> [--old grasp/config/T_base_cam.json] -o T_new.json
"""
from __future__ import annotations

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
if "--left-corr" not in sys.argv:  # default: the plain left model; --left-corr: the corrected one (ik/openarm_left_ik.py)
    os.environ["GRASP_LEFT_CORR"] = "none"
from ik.openarm_left_ik import OpenArmLeftIK  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("samples")
    ap.add_argument("--old", default=os.path.join(REPO, "grasp", "config", "T_base_cam.json"))
    ap.add_argument("--free-markers", action="store_true")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--left-corr", action="store_true", help="use the left-arm correction (left_arm_corr.json) in the FK "
                    "(2026-11-22: after a camera move, so the camera agrees with the corrected arm)")
    ap.add_argument("--floor", action="store_true", help="also put the floor seen in the first frame on config/floor_plane.json "
                    "(fixed in the base frame) after the marker fit")
    a = ap.parse_args(argv)
    ik = OpenArmLeftIK()
    s = np.load(os.path.join(a.samples, "samples.npz")); Q, K = s["q_deg"], s["K"]
    old = json.load(open(a.old)); M = [np.array(m) for m in old["markers_link7_m"]]
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    P, FK, FR = [], [], []
    for k in range(len(Q)):
        col = cv2.imread(os.path.join(a.samples, f"color_{k:03d}.jpg"))
        dep = np.load(os.path.join(a.samples, f"depth_{k:03d}.npy")).astype(np.float32)
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
            P.append([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
            F = ik.fk(np.deg2rad(Q[k])); FK.append((F.rotation, F.translation)); FR.append(k)
    P = np.array(P); n = len(P)
    print(f"[fit] {n} marker detections in {len(set(FR))} of {len(Q)} frames")
    if n < 6:
        raise SystemExit("[fit] too few detections (is the arm in view of the camera?)")

    def unpack(x):
        ms = M if not a.free_markers else [x[6:9], x[9:12]]
        return Rr.from_rotvec(x[:3]).as_matrix(), x[3:6], ms

    def resid(x, asg):
        Rc, t, ms = unpack(x)
        return np.concatenate([Rc @ P[i] + t - (FK[i][0] @ ms[asg[i]] + FK[i][1]) for i in range(n)])

    rng = np.random.default_rng(1); best = None
    T0 = np.array(old["T_base_cam"])
    for trial in range(120):
        rv = Rr.from_matrix(T0[:3, :3]).as_rotvec() if trial == 0 else Rr.random(random_state=int(rng.integers(1e9))).as_rotvec()
        t0 = T0[:3, 3] if trial == 0 else rng.uniform([-0.1, -0.15, 0.4], [0.2, 0.15, 1.1])
        x = np.r_[rv, t0] if not a.free_markers else np.r_[rv, t0, M[0], M[1]]
        asg = rng.integers(0, 2, n)
        for _ in range(6):
            f = least_squares(resid, x, args=(asg,), loss="soft_l1", f_scale=0.01, max_nfev=300); x = f.x
            Rc, t, ms = unpack(x)
            e = np.stack([np.linalg.norm(P @ Rc.T + t - np.array([FK[i][0] @ m + FK[i][1] for i in range(n)]), axis=1) for m in ms], 1)
            new = e.argmin(1)
            for fr in set(FR):  # a frame with two detections sees both markers
                idx = [i for i in range(n) if FR[i] == fr]
                if len(idx) == 2 and new[idx[0]] == new[idx[1]]:
                    c0 = e[idx[0], 0] + e[idx[1], 1]; c1 = e[idx[0], 1] + e[idx[1], 0]
                    new[idx[0]], new[idx[1]] = (0, 1) if c0 < c1 else (1, 0)
            if (new == asg).all():
                break
            asg = new
        r = e[np.arange(n), asg]
        if best is None or np.median(r) < best[0]:
            best = (float(np.median(r)), x.copy(), asg.copy(), r)
    med, x, asg, r = best
    if a.floor:
        f = json.load(open(os.path.join(REPO, "grasp", "config", "floor_plane.json"))); nb = np.array(f["n"], float); db = float(f["d"])
        d0 = np.load(os.path.join(a.samples, "depth_000.npy")).astype(np.float32)
        v, u = np.mgrid[0:d0.shape[0], 0:d0.shape[1]]; mk = (d0 > 0.2) & (d0 < 2.0)
        Pc = np.c_[(u[mk] - K[0, 2]) * d0[mk] / K[0, 0], (v[mk] - K[1, 2]) * d0[mk] / K[1, 1], d0[mk]]
        rng2 = np.random.default_rng(0); bestp = None
        for _ in range(300):  # the dominant plane of the first frame = the floor
            sp = Pc[rng2.choice(len(Pc), 3, replace=False)]; nn = np.cross(sp[1] - sp[0], sp[2] - sp[0])
            if np.linalg.norm(nn) < 1e-9:
                continue
            nn /= np.linalg.norm(nn); inl = np.abs((Pc - sp[0]) @ nn) < 0.008
            if bestp is None or inl.sum() > bestp.sum():
                bestp = inl
        Fl = Pc[bestp][rng2.choice(int(bestp.sum()), min(400, int(bestp.sum())), replace=False)]
        wf = np.sqrt(n * 3 / len(Fl))  # the floor weighs as much as all marker residuals together

        def resid_f(xx):
            Rf, tf, _ = unpack(xx)
            return np.concatenate([resid(xx, asg), wf * (Fl @ Rf.T @ nb + tf @ nb + db)])
        x = least_squares(resid_f, x, loss="soft_l1", f_scale=0.01).x
        Rc_, t_, ms_ = unpack(x)
        r = np.linalg.norm(P @ Rc_.T + t_ - np.array([FK[i][0] @ ms_[asg[i]] + FK[i][1] for i in range(n)]), axis=1)
        med = float(np.median(r)); fr = Fl @ Rc_.T @ nb + t_ @ nb + db
        print(f"[fit] with the floor: floor residual median {np.median(np.abs(fr))*1000:.1f} mm ({len(Fl)} floor points)")
    Rc, t, ms = unpack(x); z = Rc[:, 2]
    print(f"[fit] median {med*1000:.1f} mm  90% {np.percentile(r, 90)*1000:.1f} mm  max {r.max()*1000:.1f} mm  "
          f"<5 mm {(r < 0.005).sum()}/{n}  marker split {np.bincount(asg, minlength=2).tolist()}")
    print(f"[fit] camera {np.round(t, 3).tolist()} (base frame), pitch down {np.degrees(np.arcsin(-z[2])):.1f} deg, "
          f"yaw {np.degrees(np.arctan2(z[1], z[0])):.1f} deg")
    Tn = np.eye(4); Tn[:3, :3] = Rc; Tn[:3, 3] = t
    json.dump({"T_base_cam": Tn.tolist(), "markers_link7_m": [m.tolist() for m in ms], "n": int(n), "median_m": med,
               "p90_m": float(np.percentile(r, 90)), "max_m": float(r.max()),
               "method": f"fit_handeye_markers.py on {os.path.basename(os.path.normpath(a.samples))}: two id0 markers "
                         + ("(offsets refitted)" if a.free_markers else "(offsets fixed from " + os.path.basename(a.old) + ")")
                         + ", per-detection marker assignment, soft_l1" + (", left-arm correction in the FK" if a.left_corr else "")
                         + (", + floor constraint (config/floor_plane.json)" if a.floor else "")}, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
