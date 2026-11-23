#!/usr/bin/env python3
"""ENPIRE-style auto-research loop for the OpenArm top-down grasp task (no motion yet).

Loop (ENPIRE autoresearch contract): hypothesis -> smallest policy change -> run the trial budget
-> frozen verifier (contract.py) -> keep the change only if it beats the current best and passes gates.
Every iteration is recorded under grasp/runs/enpire/iter_NNN/ (hypothesis, params, per-trial
result.json + overlay, summary, decision). The dashboard (dashboard.py) reads the same files.

Sources (each trial = one scene):
    --source offline:<glob>   re-run saved captures (grasp/runs/cap_*), no hardware at all
    --source thor:N           take N fresh chest-camera frames on Thor (camera only, arm does NOT move)
    --source live:N           execute on the arm  -> REFUSED until hand-eye + IK + motion step exist

Usage (GraspGenX env, server on :5556):
    PY="env PYTHONPATH=$HOME/work/GraspGenX:$HOME/work/openarm-orboh $HOME/work/GraspGenX/.venv/bin/python"
    $PY -m grasp.enpire.loop run  --hypothesis "baseline" --source 'offline:grasp/runs/cap_*'
    $PY -m grasp.enpire.loop auto --source 'offline:grasp/runs/cap_*' --max-iters 8
    $PY -m grasp.enpire.loop run  --hypothesis "fresh frames" --source thor:5 --params grasp/runs/enpire/best_params.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import socket
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
from grasp import remote  # noqa: E402
from grasp.enpire import contract, policy  # noqa: E402

ROOT = os.path.join(REPO, "grasp", "runs", "enpire")
EVENTS = os.path.join(ROOT, "events.jsonl")
THOR = remote.HOST  # 既定は東京の Thor。腕とカメラがこの PC にあるなら GRASP_ROBOT_HOST=local（grasp/remote.py）


# ---------------------------------------------------------------- records
def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def event(kind, **kw):
    os.makedirs(ROOT, exist_ok=True)
    rec = {"t": now(), "kind": kind, **kw}
    with open(EVENTS, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    print(f"[enpire] {kind} " + " ".join(f"{k}={v}" for k, v in kw.items() if k != "params"), flush=True)


def wjson(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    os.replace(tmp, path)


def next_iter_dir():
    os.makedirs(ROOT, exist_ok=True)
    n = len(glob.glob(os.path.join(ROOT, "iter_*")))
    d = os.path.join(ROOT, f"iter_{n:03d}")
    os.makedirs(os.path.join(d, "trials"))
    return d


def track_key(source, t_base_cam_path):
    """Iterations are compared only within one track: same scenes, same hand-eye, same verifier."""
    return f"{source}|{t_base_cam_path or 'no-handeye'}|{contract.VERIFIER_VERSION}"


def best_record(source=None, t_base_cam_path=None):
    """Best of one track (best.json = {"tracks": {key: rec}, "latest": key}); no source -> latest track."""
    p = os.path.join(ROOT, "best.json")
    b = json.load(open(p)) if os.path.exists(p) else {}
    if "tracks" not in b:  # pre-track format
        b = {"tracks": {track_key(b["source"], b.get("t_base_cam")): b}, "latest": None} if b else {"tracks": {}}
    key = track_key(source, t_base_cam_path) if source else b.get("latest")
    return b["tracks"].get(key) if key else None


def save_best(rec):
    p = os.path.join(ROOT, "best.json")
    b = json.load(open(p)) if os.path.exists(p) else {}
    if "tracks" not in b:
        b = {"tracks": {track_key(b["source"], b.get("t_base_cam")): b} if b else {}}
    key = track_key(rec["source"], rec.get("t_base_cam"))
    b["tracks"][key] = rec
    b["latest"] = key
    wjson(p, b)


# ---------------------------------------------------------------- health (ENPIRE: stop when readiness fails)
def health(source: str, port: int = 5556) -> dict:
    h = {}
    s = socket.socket(); s.settimeout(1.0)
    h["graspgenx"] = s.connect_ex(("localhost", port)) == 0
    s.close()
    if source.startswith(("thor", "live")):
        h["thor_ssh"] = remote.reachable()
    return h


# ---------------------------------------------------------------- scenes
def scenes(source: str):
    kind, _, arg = source.partition(":")
    if kind == "offline":
        dirs = sorted(d for d in glob.glob(os.path.join(REPO, arg) if not os.path.isabs(arg) else arg)
                      if os.path.exists(os.path.join(d, "depth.npy")))
        for d in dirs:
            yield os.path.basename(d), d
    elif kind == "thor":
        runs = os.path.join(REPO, "grasp", "runs")
        for i in range(int(arg or 1)):
            r = remote.run("cd ~/grasp_work && o=~/grasp_runs/cap_$(date +%Y%m%d_%H%M%S); "
                           f"{remote.CAMERA_PY} grasp/capture.py --out $o >/dev/null && basename $o", timeout=None)
            if r.returncode != 0:
                raise subprocess.CalledProcessError(r.returncode, r.args, r.stdout, r.stderr)
            name = r.stdout.strip()
            remote.get(f"grasp_runs/{name}", runs)
            yield name, os.path.join(runs, name)
            if i + 1 < int(arg or 1):
                event("operator_reset", note="re-place the object; next frame in 15 s")
                time.sleep(15)
    elif kind == "live":
        raise SystemExit(
            "[enpire] live: refused. Motion needs (1) a validated hand-eye T_base_cam (current ICP fit: "
            "only ~20% of points within 1 cm), (2) the IK/plan step and (3) run_grasp.py, plus an "
            "operator at the robot who asks for it (AGENTS.md §3-8). Use offline:/thor: for now.")
    else:
        raise SystemExit(f"unknown source {source}")


def load_scene(d):
    depth = np.load(os.path.join(d, "depth.npy")).astype(np.float32)
    if os.path.exists(os.path.join(d, "intrinsics.json")):
        K = np.array(json.load(open(os.path.join(d, "intrinsics.json")))["K"])
    else:
        K = np.array(json.load(open(os.path.join(d, "meta_data.json")))["intrinsics"])
    if np.nanmax(depth) > 20:
        depth = depth / 1000.0
    color = None
    for n in ("color.png", "color.jpg", "rgb.png"):
        if os.path.exists(os.path.join(d, n)):
            import cv2
            color = cv2.imread(os.path.join(d, n))
            break
    return depth, K, color


# ---------------------------------------------------------------- plan level (optional, offline IK)
def ik_check(G_cam, T_base_cam, side=None):
    """Grasp (camera frame, base=flange) -> link7 target in arm base -> numeric IK on urdf_fk.
    Returns {"converged", "pos_err_m", "rot_err_deg"}. Pure numpy/scipy; no motion.
    腕（2026-11-22 Munenori）: side が無ければ GRASP_ARM（both なら左右とも解いて誤差の小さい方。結果に "arm" を入れる）。"""
    if side is None:
        env_arm = os.environ.get("GRASP_ARM", "left")
        sides = ["left", "right"] if env_arm == "both" else [env_arm]
        res = [ik_check(G_cam, T_base_cam, sd) for sd in sides]
        return min(res, key=lambda r: r["pos_err_m"] + r["rot_err_deg"] / 1000)
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as R
    from grasp.urdf_fk import LeftArmFK

    cache = getattr(ik_check, "_fks", {})
    fk = cache.get(side) or LeftArmFK(side=side)
    cache[side] = fk
    ik_check._fks = cache
    lim = []
    for j in fk.chain:
        if j.get("type") == "revolute":
            l = j.find("limit")
            lim.append((float(l.get("lower")), float(l.get("upper"))))
    lo, hi = np.array(lim).T
    flange = np.eye(4); flange[2, 3] = 0.0895          # hand mount face in link7 (config/openarm_left_gripper.json)
    best = None
    for yaw in (0.0, np.pi):                          # parallel jaw: closing axis is symmetric
        Rz = np.eye(4); Rz[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
        T = T_base_cam @ G_cam @ Rz @ np.linalg.inv(flange)

        def res(q):
            P = fk.link_poses(q)[f"openarm_{side}_link7"]
            dp = P[:3, 3] - T[:3, 3]
            dr = R.from_matrix(T[:3, :3].T @ P[:3, :3]).as_rotvec() * 0.1  # 10 cm per rad
            return np.r_[dp, dr]

        seed2 = np.array([-30, 0, 30, 60, 0, 0, 0], float)
        if side == "right":  # 左の種を鏡に写す（grasp/arm.py）
            from grasp.arm import mirror_q
            seed2 = mirror_q(seed2)
        for seed in (np.zeros(7), np.deg2rad(seed2)):
            r = least_squares(res, np.clip(seed, lo + 1e-3, hi - 1e-3), bounds=(lo, hi), max_nfev=300)
            pe = float(np.linalg.norm(r.fun[:3])); re = float(np.degrees(np.linalg.norm(r.fun[3:]) / 0.1))
            if best is None or pe + re / 1000 < best["pos_err_m"] + best["rot_err_deg"] / 1000:
                best = {"pos_err_m": pe, "rot_err_deg": re, "q_deg": np.degrees(r.x).round(1).tolist()}
    best["converged"] = best["pos_err_m"] < 0.005 and best["rot_err_deg"] < 3.0
    best["arm"] = side
    return best


# ---------------------------------------------------------------- one trial
def run_trial(name, d, p, tdir, T_base_cam=None):
    from grasp.capture import depth_to_points
    from grasp.segment import SegParams, segment_clusters
    from grasp import objects
    from grasp.infer import infer_grasps, load_sweep

    os.makedirs(tdir, exist_ok=True)
    t0 = time.time()
    obs, info = {}, {"scene": name, "scene_dir": d}
    depth, K, color = load_scene(d)
    (v0, v1), (u0, u1) = p["rows"], p["cols"]
    dm = np.zeros_like(depth); dm[v0:v1, u0:u1] = depth[v0:v1, u0:u1]
    spec = objects.load(p["target"]) if p.get("target") else None
    if spec is not None and color is None:
        raise SystemExit(f"[loop] --target {spec['name']} には色の画像が要る（{d}）")
    if spec is not None and spec["separate_from"]:
        dm[objects.separate_mask(color, spec)] = 0
    pts, _ = depth_to_points(dm, K, zmin=p["zmin"], zmax=p["zmax"])
    sp = SegParams(voxel=p["voxel"], zmax=p["zmax"], plane_dist=p["plane_dist"],
                   above_plane_min=p["above_plane_min"], above_plane_max=p["above_plane_max"],
                   dbscan_eps=p["dbscan_eps"], dbscan_min_pts=p["dbscan_min_pts"],
                   min_cluster_pts=p["min_cluster_pts"])
    try:
        clusters, dbg = segment_clusters(pts, sp)
        best = 0
        if spec is not None:
            best, table = objects.pick(clusters, K, color, spec, plane=dbg.get("plane"))
            info["target"] = {"name": spec["name"], "picked": best, "clusters": table}
            if best is None:
                raise RuntimeError(objects.not_found(spec, table))
        obj = clusters[best]
        obs["obj"], obs["plane"] = obj, dbg["plane"]
        info["seg"] = {"pts": len(obj), "extent_m": np.ptp(obj, 0).round(3).tolist(), **dbg}
    except RuntimeError as e:
        obs["obj"], obs["seg_error"] = None, str(e)
    if obs.get("obj") is not None:
        try:
            sweep = load_sweep(p["sweep_json"]) if p.get("sweep_json") else None
            g, s = infer_grasps(obs["obj"], gripper_name=p["gripper_name"], sweep=sweep,
                                num_grasps=p["num_grasps"], topk=p["topk"], max_points=p["max_points"])
            i, sel = policy.select_grasp(g, s, obs["plane"], p)
            info["select"] = {**sel, "n_grasps": int(len(s))}
            if i is not None:
                obs["grasp"], obs["score"] = g[i], float(s[i])
                if T_base_cam is not None:
                    order = [i] + [j for j in np.argsort(-s) if j != i]
                    nt = int(p.get("select_ik_tries", 1))
                    n_ok = np.asarray(obs["plane"][:3], float)
                    tried = 0
                    for j in order:
                        if tried >= nt:
                            break
                        tilt = np.degrees(np.arccos(np.clip(-(g[j][:3, 2] @ n_ok), -1, 1)))
                        if j != i and (tilt > p["select_max_tilt_deg"] or s[j] < p["select_min_score"]):
                            continue
                        tried += 1
                        ik = ik_check(np.asarray(g[j], float), T_base_cam)
                        if ik["converged"] or j == i:
                            obs["grasp"], obs["score"], obs["ik"] = g[j], float(s[j]), ik
                        if ik["converged"]:
                            break
                    info["select"]["ik_tries"] = tried
        except Exception as e:  # server down / wire error -> the trial is a failure, recorded as such
            obs["infer_error"] = f"{type(e).__name__}: {e}"
    res = contract.evaluate(obs)
    res.update(info=info, seconds=round(time.time() - t0, 2),
               grasp=None if obs.get("grasp") is None else np.asarray(obs["grasp"]).tolist(),
               score=obs.get("score"), ik=obs.get("ik"))
    if color is not None:
        overlay(color, K, obs, res, os.path.join(tdir, "overlay.jpg"), (u0, v0, u1, v1))
    wjson(os.path.join(tdir, "result.json"), res)
    return res


def overlay(img, K, obs, res, path, roi):
    import cv2
    img = img.copy()
    if obs.get("obj") is not None:
        uv = obs["obj"] @ K.T; uv = (uv[:, :2] / uv[:, 2:]).astype(int)
        for u, v in uv[:: max(1, len(uv) // 3000)]:
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (int(u), int(v)), 1, (0, 0, 255), -1)
    G = obs.get("grasp")
    if G is not None:
        G = np.asarray(G)
        tip = G[:3, 3] + contract.FINGERTIP_DEPTH_M * G[:3, 2]
        def px(x):
            q = K @ x; return int(q[0] / q[2]), int(q[1] / q[2])
        cv2.arrowedLine(img, px(G[:3, 3]), px(tip), (255, 120, 0), 2, tipLength=0.2)
        w = contract.MAX_OPENING_M / 2 * G[:3, 0]
        cv2.line(img, px(tip - w), px(tip + w), (0, 255, 0), 2)
    cv2.rectangle(img, roi[:2], (roi[2] - 1, roi[3] - 1), (255, 200, 0), 1)
    col = (0, 200, 0) if res["success"] else (0, 0, 255)
    cv2.putText(img, ("SUCCESS" if res["success"] else res["cause"])[:70], (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    cv2.imwrite(path, img)


# ---------------------------------------------------------------- one iteration
def run_iteration(hypothesis, p, source, T_base_cam=None, t_base_cam_path=None, parent=None):
    d = next_iter_dir()
    it = os.path.basename(d)
    h = health(source)
    meta = {"iter": it, "hypothesis": hypothesis, "source": source, "parent": parent,
            "params": p, "t_base_cam": t_base_cam_path, "verifier_version": contract.VERIFIER_VERSION,
            "gate": contract.SUCCESS_GATE, "health": h, "started": now(), "status": "running"}
    wjson(os.path.join(d, "iteration.json"), meta)
    event("iter_start", iter=it, hypothesis=hypothesis, source=source, health=h)
    if not all(h.values()):
        meta.update(status="aborted", reason=f"readiness failed: {h}", finished=now())
        wjson(os.path.join(d, "iteration.json"), meta)
        event("iter_abort", iter=it, reason=meta["reason"])
        return meta
    results = []
    for k, (name, sd) in enumerate(scenes(source)):
        try:
            r = run_trial(name, sd, p, os.path.join(d, "trials", f"{k:02d}_{name}"), T_base_cam)
        except Exception as e:
            r = {"success": False, "cause": f"crash: {type(e).__name__}: {e}", "level": "perception",
                 "info": {"scene": name}, "traceback": traceback.format_exc()[-1500:]}
            os.makedirs(os.path.join(d, "trials", f"{k:02d}_{name}"), exist_ok=True)
            wjson(os.path.join(d, "trials", f"{k:02d}_{name}", "result.json"), r)
        results.append(r)
        event("trial", iter=it, trial=k, scene=name, success=r["success"], cause=r["cause"])
    n = len(results)
    sr = sum(r["success"] for r in results) / n if n else 0.0
    hist = {}
    for r in results:
        hist[r["cause"]] = hist.get(r["cause"], 0) + 1
    best = best_record(source, t_base_cam_path)
    if not n:
        beats, decision = False, "invalid (no trials)"
    elif best is None:
        beats, decision = True, "keep (first baseline of this track)"
    else:
        beats = sr > best["success_rate"]
        decision = "keep (new best)" if beats else "revert (no improvement)"
    meta.update(status="done", finished=now(), n_trials=n, success_rate=sr, failure_histogram=hist,
                passes_gate=sr >= contract.SUCCESS_GATE, decision=decision,
                best_before=None if best is None else {"iter": best["iter"], "success_rate": best["success_rate"]})
    wjson(os.path.join(d, "iteration.json"), meta)
    if beats:
        save_best({"iter": it, "success_rate": sr, "source": source, "params": p,
                   "t_base_cam": t_base_cam_path, "verifier_version": contract.VERIFIER_VERSION})
        wjson(os.path.join(ROOT, "best_params.json"), p)
    event("iter_done", iter=it, success_rate=round(sr, 3), decision=decision, n=n)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("run", "auto"):
        s = sub.add_parser(c)
        s.add_argument("--source", default="offline:grasp/runs/cap_*")
        s.add_argument("--params", default=None, help="policy params JSON (default: best so far, else defaults)")
        s.add_argument("--t-base-cam", default=None, help="hand-eye JSON with T_base_cam -> enables the plan level (IK)")
        if c == "run":
            s.add_argument("--hypothesis", required=True)
            s.add_argument("--set", nargs="*", default=[], help="override params k=v (JSON values)")
        else:
            s.add_argument("--max-iters", type=int, default=10)
    a = ap.parse_args()

    T, tp = None, None
    if a.t_base_cam:
        tp = os.path.abspath(a.t_base_cam)
        T = np.array(json.load(open(tp))["T_base_cam"], float)
    best = best_record(a.source, tp)
    base = policy.load(a.params) if a.params else (best["params"] if best else policy.load(None))

    if a.cmd == "run":
        p = dict(base)
        for kv in a.set:
            k, v = kv.split("=", 1)
            p[k] = json.loads(v)
        run_iteration(a.hypothesis, p, a.source, T, tp, parent=best and best["iter"])
        return
    # auto: baseline (if no best for this source) then one-knob proposals around the moving best
    if not best:
        run_iteration("baseline (default policy params)", base, a.source, T, tp)
    tried = set()
    for _ in range(a.max_iters):
        best = best_record(a.source, tp)
        prop = next(((h, q) for h, q in policy.proposals(best["params"])
                     if json.dumps(q, sort_keys=True) not in tried), None)
        if prop is None:
            event("auto_done", reason="search space exhausted"); break
        h, q = prop
        tried.add(json.dumps(q, sort_keys=True))
        m = run_iteration(h, q, a.source, T, tp, parent=best["iter"])
        if m.get("status") == "aborted":
            break
    b = best_record(a.source, tp)
    event("auto_done", best=b["iter"], success_rate=b["success_rate"])


if __name__ == "__main__":
    main()
