#!/usr/bin/env python3
"""Decide what to take first from a cluttered pile, and plan it. **Does not move the arm.**

2026-11-21 (Sota): the old chain only looked for aluminium and tried bars in a fixed order (crowd, then height).
This file is the "which first" step:

  1. objects: perceive_server twice (aluminium words, then other words: wood / cardboard / bag ...). One mask per
     object, its points (fixed floor, config/floor_plane.json), height, size, whether it is in reach.
  2. relations: which objects touch, which one rests on which (depth along the shared border).
  3. advice: the image-language model (vlm_server.py, Qwen3-VL) gets the picture with numbers and the list, and
     returns the order to take things: pick an aluminium piece, or move an obstacle (non-aluminium) out of the way.
     Without the model the order is: aluminium by crowd then height, then non-aluminium that touch an aluminium piece.
  4. check: for each proposed action, the real planner (plan.py; bars: bar grasps + GraspGenX, other things:
     GraspGenX) on that object, with all other rigid objects as obstacles. The first action that plans is chosen;
     plan.json / object.npy in the capture belong to it.
  5. confidence: low when the aluminium score is borderline, few grasps passed, or the model's first choice did not
     plan. Then the panel asks a person before GO.
  -> decision.json + decision.jpg in the capture folder.

    python grasp/tools/decide.py grasp/runs/<cap> --q <home deg> --plan-args "<alu plan args>"
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import os
import re
import sys
import time

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

ALU_WORDS = ["aluminum frame", "metal bar", "aluminum extrusion", "silver metal rod", "white metal bar", "long thin bar"]
OTHER_WORDS = ["wooden block", "cardboard box", "plastic bag", "white sack", "paper", "plastic bottle", "cloth"]
SOFT = ("bag", "sack", "paper", "cloth")          # soft: never an obstacle to the jaws, never a target
ALU_THR, ALU_SURE = 0.55, 0.75
# 2026-11-22 (Sota: 細長さの条件): on the G1 site frames (11/17, floor) grey metal sheets scored 0.60-0.75 as
# "aluminium". A frame is long and thin; a sheet is not. Shape of the mask's largest blob (min-area rectangle):
BAR_ASPECT, BAR_FILL = 3.5, 0.55   # long/short >= 3.5, blob fills >= 55 % of its rectangle. 0.75 dropped 5 real bars (occluded / cut by the image edge, fill 0.59-0.74) in our own captures
OTHER_ARGS = ("--max-tilt 45 --rounds 8 --min-score 0.5 --max-try 400 --center-tol 0.06 --avoid-obstacles --soft-below=-1 "
              "--clear-gap --gap-soft-below 0.01 --floor-clear 0.006 --max-branch-deg 60")


def ask(port, req, timeout_ms):
    import msgpack
    import msgpack_numpy as mn
    import zmq
    s = zmq.Context.instance().socket(zmq.REQ); s.setsockopt(zmq.RCVTIMEO, timeout_ms); s.setsockopt(zmq.LINGER, 0)
    s.connect(f"tcp://localhost:{port}")
    try:
        s.send(msgpack.packb(req, default=mn.encode, use_bin_type=True))
        return msgpack.unpackb(s.recv(), object_hook=mn.decode, raw=False)
    finally:
        s.close()


def bar_shape(mask):
    """(aspect, fill) of the largest blob: long/short side of its min-area rectangle, and blob area / rectangle area."""
    cs, _ = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return 0.0, 0.0
    c = max(cs, key=cv2.contourArea)
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    lo, hi = min(w, h), max(w, h)
    return hi / max(lo, 1.0), cv2.contourArea(c) / max(w * h, 1.0)


def bar_like(mask):
    a, f = bar_shape(mask)
    return a >= BAR_ASPECT and f >= BAR_FILL


def iou(a, b):
    return (a & b).sum() / max(1, (a | b).sum())


def objects(img, depth, K, T, n_b, d_b, band, perceive_port):
    """All objects in reach: list of dicts with mask, kind ('aluminium' | 'obstacle' | 'soft'), name, alu score, points."""
    H, W = depth.shape
    reps = []
    for words in (ALU_WORDS, OTHER_WORDS):
        r = ask(perceive_port, {"bgr": img, "prompts": words}, 60000)
        if "error" in r:
            raise RuntimeError(r["error"])
        reps.append(r["cands"])
    alu_c, oth_c = reps
    objs = []
    for c in alu_c:
        m = np.asarray(c["mask"], bool)
        if m.sum() > 0.08 * H * W:
            continue
        kind = "aluminium" if c["alu"] >= ALU_THR else None
        if not bar_like(m):  # not long and thin (a sheet, a lump): never an aluminium frame; a rigid obstacle
            # if it is metal. Checked before merging, so a sheet's mask never swallows the bar lying on it
            if kind:
                objs.append({"mask": m, "name": c["label"] + " (not bar-shaped)", "alu": float(c["alu"]),
                             "kind": "obstacle", "label": c["label"], "not_bar": True})
            continue
        # 2026-11-22: the same bar found twice (two prompts / boxes) made each copy the other's "neighbour", so every
        # grasp "hit" it. Keep one: the higher aluminium score, mask = union
        same = [o for o in objs if not o.get("not_bar") and (o["mask"] & m).sum() > 0.6 * max(1, min(o["mask"].sum(), m.sum()))]
        if same:
            o = same[0]; o["mask"] = o["mask"] | m
            if c["alu"] > o["alu"]:
                o.update(alu=float(c["alu"]), kind=kind or o["kind"], label=c["label"])
            continue
        objs.append({"mask": m, "name": "aluminium bar" if kind else c["label"], "alu": float(c["alu"]), "kind": kind,
                     "label": c["label"]})
    bars = np.zeros((H, W), bool)
    for o in objs:
        if not o.get("not_bar") and o["kind"] == "aluminium":
            bars |= o["mask"]
    for o in objs:  # a bar lying on a sheet: the bar's pixels are not part of the sheet obstacle
        if o.get("not_bar"):
            o["mask"] = o["mask"] & ~bars
    objs = [o for o in objs if o["mask"].sum() > 0]
    for c in oth_c:
        m = np.asarray(c["mask"], bool)
        if m.sum() > 0.15 * H * W:
            continue
        dup = [o for o in objs if iou(o["mask"], m) > 0.5]
        lab = c["label"].lower()
        if dup:  # the same thing found by both: keep the aluminium call only if it is confident
            o = dup[0]
            if o["kind"] is None or o["alu"] < ALU_SURE:
                o["kind"] = "soft" if any(w in lab for w in SOFT) else "obstacle"; o["name"] = c["label"]
            continue
        objs.append({"mask": m, "name": c["label"], "alu": 0.0, "kind": "soft" if any(w in lab for w in SOFT) else "obstacle",
                     "label": c["label"]})
    objs = [o for o in objs if o["kind"] is not None]
    # points, height, reach
    v, u = np.mgrid[0:H, 0:W]
    P = np.stack([(u - K[0, 2]) * depth / K[0, 0], (v - K[1, 2]) * depth / K[1, 1], depth], -1)
    n_c = T[:3, :3].T @ n_b; dd = float(d_b + n_b @ T[:3, 3])
    h_all = P @ n_c + dd
    x0, x1, y0, y1 = band
    out = []
    for o in objs:
        m = cv2.erode(o["mask"].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        ok = m & (depth > 0) & (h_all > 0.005) & (h_all < 0.20)
        if ok.sum() < 60:
            # 2026-11-22 16:35: a flat bar (~1 cm) on the sack read -5 mm against the floor plane (the plane is off by
            # +-7 mm across the near-vertical camera's view). Measure it against the ground right around it instead
            mu = o["mask"].astype(np.uint8)
            ring = cv2.dilate(mu, np.ones((25, 25), np.uint8)).astype(bool) & ~cv2.dilate(mu, np.ones((9, 9), np.uint8)).astype(bool) & (depth > 0)
            if ring.sum() > 100:
                base = float(np.median(h_all[ring]))
                ok = m & (depth > 0) & (h_all - base > 0.003) & (h_all < 0.20)
                if ok.sum() >= 60:
                    print(f"[decide] {o['name']}: measured against the ground around it ({base*1000:+.0f} mm vs the floor plane)")
                elif o["kind"] == "aluminium" and o["alu"] >= ALU_SURE:
                    # 16:42: a flat bar (a few mm) reads at the ground's own depth (D435i at 0.8 m: ~2.5 mm noise, shiny
                    # aluminium is filled in from around it). A sure, bar-shaped aluminium mask keeps all its points:
                    # the planner then sets the jaws by the ground (--tip-min above it)
                    ok = m & (depth > 0) & (h_all - base > -0.006) & (h_all < 0.20)
                    if ok.sum() >= 60:
                        print(f"[decide] {o['name']}: flat on the ground (no height in the depth image); kept as a flat bar")
        if ok.sum() < 60:
            continue
        # ground right around the object vs the floor plane (2026-11-22 16:45): the jaws' depth is set from it
        mu = o["mask"].astype(np.uint8)
        ring = cv2.dilate(mu, np.ones((25, 25), np.uint8)).astype(bool) & ~cv2.dilate(mu, np.ones((9, 9), np.uint8)).astype(bool) & (depth > 0)
        # 17:10: the chest depth as a whole drifts +-1 cm between captures (the floor did not move), so a ground that
        # reads HIGH is mostly that drift: raising the jaws by it closed them above the bar (16:59). Only lower (max 1 cm)
        o["ground"] = float(np.clip(np.median(h_all[ring]), -0.010, 0.0)) if ring.sum() > 100 else 0.0
        pts = P[ok]; pb = pts @ T[:3, :3].T + T[:3, 3]
        cx, cy = pb[:, 0].mean(), pb[:, 1].mean()
        if not (x0 < cx < x1 and y0 < cy < y1):
            continue
        o.update(points=pts.astype(np.float32), top=float(np.percentile(h_all[ok], 90)), centre=[float(cx), float(cy)],
                 size=float(np.ptp(pb[:, :2], axis=0).max()), h_med=float(np.median(h_all[ok])))
        if o["kind"] == "soft" and o["top"] > 0.025 and o["size"] < 0.20:  # 2.5 cm: the chest depth drifts +-1 cm (00:29 the 5 cm block read 3.0 cm)
            # 2026-11-22 22:27: a lone wood block was named "paper" (soft: never taken). A compact thing standing
            # 3.5 cm or more is rigid whatever it is called (a sack / cloth / paper lies flat or is large)
            o["kind"] = "obstacle"; o["name"] = o["label"] + " (rigid: stands up)"
        out.append(o)
    for i, o in enumerate(out):
        o["id"] = i + 1
    return out, h_all


def relations(objs, h_all):
    """touching pairs and 'A rests on B' (A higher than B along their shared border)."""
    rel = []
    k = np.ones((9, 9), np.uint8)
    for a in objs:
        da = cv2.dilate(a["mask"].astype(np.uint8), k).astype(bool)
        for b in objs:
            if b["id"] <= a["id"]:
                continue
            db = cv2.dilate(b["mask"].astype(np.uint8), k).astype(bool)
            border = da & db
            if border.sum() < 15:
                continue
            ha = np.nanmedian(np.where(a["mask"] & db, h_all, np.nan)) if (a["mask"] & db).any() else a["h_med"]
            hb = np.nanmedian(np.where(b["mask"] & da, h_all, np.nan)) if (b["mask"] & da).any() else b["h_med"]
            if ha > hb + 0.01:
                rel.append({"a": a["id"], "b": b["id"], "rel": "on"})
            elif hb > ha + 0.01:
                rel.append({"a": b["id"], "b": a["id"], "rel": "on"})
            else:
                rel.append({"a": a["id"], "b": b["id"], "rel": "touch"})
    return rel


def crowd(o, objs):
    """share of the 6.5 cm strip on both sides of o along its length that holds a rigid neighbour (0..1)."""
    pb = o["points"]
    rig = [x for x in objs if x is not o and x["kind"] != "soft"]
    if not rig:
        return 0.0
    q = np.concatenate([x["points"] for x in rig])
    mu = pb[:, :2].mean(0); w, V = np.linalg.eigh(np.cov((pb[:, :2] - mu).T)); ax = V[:, 1]; nx = np.array([-ax[1], ax[0]])
    al, ac = (pb[:, :2] - mu) @ ax, (pb[:, :2] - mu) @ nx
    hw = float(np.percentile(np.abs(ac), 90))
    qa, qc = (q[:, :2] - mu) @ ax, (q[:, :2] - mu) @ nx
    a0, a1, c0 = al.min() - 0.02, al.max() + 0.02, hw + 0.006
    ins = (qa > a0) & (qa < a1) & (np.abs(qc) > c0) & (np.abs(qc) < 0.065)
    cells = {(int((x - a0) // 0.01), int(np.floor(y / 0.01))) for x, y in zip(qa[ins], qc[ins])}
    return round(len(cells) / max(1, int(np.ceil((a1 - a0) / 0.01)) * 2 * int(np.ceil((0.065 - c0) / 0.01))), 3)


def numbered(img, objs, chosen=None):
    vis = img.copy()
    col = {"aluminium": (0, 0, 255), "obstacle": (255, 0, 255), "soft": (200, 200, 200)}
    for o in objs:
        c = col[o["kind"]]
        ov = vis.copy(); ov[o["mask"]] = c; vis = cv2.addWeighted(ov, 0.35, vis, 0.65, 0)
        cnts, _ = cv2.findContours(o["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 220, 0) if chosen == o["id"] else c, 3 if chosen == o["id"] else 1)
    for o in sorted(objs, key=lambda o: o["kind"] != "soft"):  # numbers of hard things on top
        v, u = np.nonzero(o["mask"]); p = (int(u.mean()), int(v.mean()))
        if o["kind"] == "soft":  # big soft things: put the number at a corner of the mask so it does not hide a bar
            p = (int(u.min()) + 15, int(v.max()) - 15)
        cv2.circle(vis, p, 13, (255, 255, 255), -1); cv2.circle(vis, p, 13, (0, 0, 0), 2)
        cv2.putText(vis, str(o["id"]), (p[0] - 7 if o["id"] < 10 else p[0] - 11, p[1] + 6), 0, 0.6, (0, 0, 0), 2)
    return vis


JA_KIND = {"aluminium": "アルミ", "obstacle": "アルミ以外（硬い物）", "soft": "柔らかい物（袋・紙など）"}


def vlm_order(img_num, objs, rel, port):
    lines = []
    for o in objs:
        r_on = [f"#{r['b']} の上に乗っている" for r in rel if r["a"] == o["id"] and r["rel"] == "on"]
        r_under = [f"#{r['a']} が上に乗っている" for r in rel if r["b"] == o["id"] and r["rel"] == "on"]
        r_t = [f"#{r['b'] if r['a'] == o['id'] else r['a']} に接している" for r in rel if r["rel"] == "touch" and o["id"] in (r["a"], r["b"])]
        lines.append(f"#{o['id']}: {JA_KIND[o['kind']]}（{o['name']}、アルミらしさ {o['alu']:.2f}）、床からの高さ {o['top']*100:.0f} cm、"
                     f"大きさ {o['size']*100:.0f} cm、混み {o['crowd']:.2f}" + ("、" + "・".join(r_on + r_under + r_t) if r_on + r_under + r_t else ""))
    prompt = ("これはロボットの胸カメラから見た、ごみの山の写真です（画像の上がロボット側）。番号は検出した物です。\n"
              + "\n".join(lines) +
              "\n\nロボットの目的は、アルミの棒だけを山から取り出すことです。爪は平行の 2 本指で、棒を横から挟みます。"
              "他の物がアルミに接していたり上に乗っていたりして爪が入らないときは、先にその物をどかしてよい（柔らかい物はどかさない）。"
              "次に何をすべきか、順番の候補を最大 4 つ、次の形の JSON 1 行だけで答えてください（説明文は書かない、why は 15 字以内）。\n"
              '形: {"scene":"<山の様子>","order":[{"id":<番号>,"action":"<pick か remove>","why":"<理由>"}, ...]}\n'
              "上の一覧の番号だけを使い、写真と一覧を見て自分で順番を決めること。")
    t0 = time.time()
    r = ask(port, {"bgr": img_num, "prompt": prompt, "max_new_tokens": 220}, 120000)
    if "error" in r:
        raise RuntimeError(r["error"])
    txt = r["text"]
    m = re.search(r"\{.*\}", txt, re.S)
    j = json.loads(m.group(0)) if m else {"scene": txt[:200], "order": []}
    j["seconds"] = time.time() - t0; j["raw"] = txt; j["prompt"] = prompt
    return j


def run_plan(argv):
    from grasp import plan as P
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            rc = P.main(argv) or 0
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 1
    return rc, buf.getvalue()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cap_dir")
    ap.add_argument("--q", required=True)
    ap.add_argument("--plan-args", default="", help="plan.py args for aluminium (bar grasps etc.)")
    ap.add_argument("--clear-all", action="store_true", help="also take the rigid things that do not touch aluminium (clear the area)")
    ap.add_argument("--other-args", default=OTHER_ARGS, help="plan.py args for moving a non-aluminium obstacle")
    ap.add_argument("--band", default="0.25,0.56,-0.20,0.35")
    ap.add_argument("--perceive-port", type=int, default=5558)
    ap.add_argument("--vlm-port", type=int, default=5560)
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--max-tries", type=int, default=4)
    ap.add_argument("--only", type=int, default=None, help="try only this object id (debug)")
    a = ap.parse_args(argv)
    d = a.cap_dir
    t_start = time.time()
    from grasp.plan import floor_plane_base
    T = np.array(json.load(open(os.environ.get("GRASP_T_BASE_CAM", os.path.join(REPO, "grasp", "config", "T_base_cam.json"))))["T_base_cam"])
    img = cv2.imread(os.path.join(d, "color.png")); depth = np.load(os.path.join(d, "depth.npy")).astype(np.float32)
    K = np.array(json.load(open(os.path.join(d, "intrinsics.json")))["K"])
    n_b, d_b = floor_plane_base(d, T)
    objs, h_all = objects(img, depth, K, T, n_b, d_b, [float(v) for v in a.band.split(",")], a.perceive_port)
    rel = relations(objs, h_all)
    for o in objs:
        o["crowd"] = crowd(o, objs) if o["kind"] == "aluminium" else None
        o["crowd_v"] = o["crowd"] if o["crowd"] is not None else 0.0
    for o in objs:
        o["crowd"] = o["crowd_v"]
    print(f"[decide] {len(objs)} objects: " + ", ".join(f"#{o['id']} {o['kind']} {o['name']} alu {o['alu']:.2f}" for o in objs), flush=True)
    img_num = numbered(img, objs)
    cv2.imwrite(os.path.join(d, "objects.jpg"), img_num)
    # the model's order (advice), then a fallback order so nothing is left out
    vlm, order = None, []
    if not a.no_vlm:
        try:
            vlm = vlm_order(img_num, objs, rel, a.vlm_port)
            for it in vlm.get("order", []):
                if any(o["id"] == int(it.get("id", -1)) for o in objs):
                    order.append({"id": int(it["id"]), "action": it.get("action", "pick"), "why": it.get("why", ""), "from": "vlm"})
            print(f"[decide] model ({vlm['seconds']:.1f} s): {vlm.get('scene', '')} -> " +
                  ", ".join(f"#{x['id']} {x['action']}" for x in order), flush=True)
        except Exception as e:  # noqa: BLE001
            vlm = {"error": str(e)}
            print(f"[decide] model not available ({e}); fixed order", flush=True)
    alus = sorted([o for o in objs if o["kind"] == "aluminium"], key=lambda o: (o["crowd"] > 0.10, o["crowd"], -o["top"]))
    for o in alus:
        if not any(x["id"] == o["id"] for x in order):
            order.append({"id": o["id"], "action": "pick", "why": "アルミ（混みの少ない順）", "from": "rule"})
    touching = {r["a"] for r in rel} | {r["b"] for r in rel}
    for o in sorted([o for o in objs if o["kind"] == "obstacle" and o["id"] in touching], key=lambda o: -o["top"]):
        if not any(x["id"] == o["id"] for x in order):
            order.append({"id": o["id"], "action": "remove", "why": "アルミに接している硬い物（高い順）", "from": "rule"})
    if a.clear_all:  # 2026-11-22 (Sota: 映ってるやつ全部取り除いて): the other rigid things too, highest first
        for o in sorted([o for o in objs if o["kind"] == "obstacle"], key=lambda o: -o["top"]):
            if not any(x["id"] == o["id"] for x in order):
                order.append({"id": o["id"], "action": "remove", "why": "残りの硬い物（全部取り除く・高い順）", "from": "rule"})
    # check each action with the real planner
    by_id = {o["id"]: o for o in objs}
    tried, chosen = [], None
    if a.only is not None:
        order = [x for x in order if x["id"] == a.only] or [{"id": a.only, "action": "pick", "why": "--only", "from": "rule"}]
    for it in order[: a.max_tries + 2]:
        o = by_id[it["id"]]
        if o["kind"] == "soft":
            tried.append({**it, "ok": False, "why_fail": "柔らかい物は取らない"}); continue
        if it["action"] == "pick" and o["kind"] != "aluminium":
            it["action"] = "remove"
        if o["kind"] == "aluminium":  # aluminium is always the thing we take, even when the model says "remove"
            it["action"] = "pick"
        np.save(os.path.join(d, "object.npy"), o["points"])
        rig = np.zeros(depth.shape, bool)
        for x in objs:
            if x is not o and x["kind"] != "soft":
                rig |= x["mask"]
        rig &= ~cv2.dilate(o["mask"].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        cv2.imwrite(os.path.join(d, "rigid.png"), rig.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(d, "mask.png"), o["mask"].astype(np.uint8) * 255)
        for f in glob.glob(os.path.join(d, "object_alt_*.npy")) + glob.glob(os.path.join(d, "grasps_*.npz")) + [os.path.join(d, "plan.json")]:
            if os.path.exists(f):
                os.remove(f)
        args = a.plan_args if o["kind"] == "aluminium" else a.other_args
        t0 = time.time()
        if o["kind"] == "aluminium" and abs(o.get("ground", 0.0)) > 0.002:
            os.environ["GRASP_FLOOR_SHIFT"] = f"{o['ground']:.4f}"
            print(f"[decide] #{o['id']}: the ground around it is {o['ground']*1000:+.0f} mm from the floor plane; the jaws' depth uses it")
        try:
            rc, out = run_plan([d, f"--q={a.q}", "--reinfer", *args.split()])
        finally:
            os.environ.pop("GRASP_FLOOR_SHIFT", None)
        st = (json.load(open(os.path.join(d, "plan_stats.json"))) or [{}])[-1] if os.path.exists(os.path.join(d, "plan_stats.json")) else {}
        ok = rc == 0 and os.path.exists(os.path.join(d, "plan.json"))
        n_pass = sum(n for k, n in (st.get("fates") or {}).items() if k in ("選んだ", "試す前に決まった"))
        rec = {**it, "ok": ok, "seconds": round(time.time() - t0, 1), "fates": st.get("fates"), "n_pass": n_pass,
               "advice": st.get("advice"), "grasp_source": (st.get("chosen") or {}).get("grasp_source")}
        if not ok:
            rec["why_fail"] = st.get("advice") or "掴み方が見つからない"
        tried.append(rec)
        print(f"[decide] #{o['id']} {it['action']}: {'OK' if ok else 'no'} ({rec['seconds']} s) {rec.get('why_fail') or ''}", flush=True)
        if ok:
            chosen = rec
            for f in glob.glob(os.path.join(d, "plan_stats.json")) + glob.glob(os.path.join(d, "plan_funnel_*.jpg")) + glob.glob(os.path.join(d, "advice_*.jpg")):
                pass  # keep the chosen object's planner records as they are
            break
    # confidence
    why_low = []
    if chosen:
        o = by_id[chosen["id"]]
        if o["kind"] == "aluminium" and o["alu"] < ALU_SURE:
            why_low.append(f"アルミかどうかがきわどい（点数 {o['alu']:.2f}）")
        if chosen["n_pass"] < 3:
            why_low.append(f"通った掴み方が {chosen['n_pass']} 個しかない")
        if order and order[0]["from"] == "vlm" and order[0]["id"] != chosen["id"]:
            why_low.append(f"モデルの 1 番（#{order[0]['id']}）は掴み方がなかった")
        if chosen["action"] == "remove":
            why_low.append("アルミ以外の物を動かす")
    vis = numbered(img, objs, chosen["id"] if chosen else None)
    cv2.imwrite(os.path.join(d, "decision.jpg"), vis)
    say = (f"#{chosen['id']} を{'取る' if chosen['action'] == 'pick' else 'どける'}: {chosen['why']}" if chosen else
           "今は取れる物がない: " + " / ".join(f"#{t['id']} {t.get('why_fail', '')}" for t in tried[:3]))
    dec = {"objects": [{k: o[k] for k in ("id", "kind", "name", "alu", "top", "size", "crowd", "centre", "label")} for o in objs],
           "relations": rel, "vlm": None if vlm is None else {k: vlm.get(k) for k in ("scene", "order", "seconds", "raw", "error")},
           "order": order, "tried": tried, "chosen": chosen, "confident": bool(chosen) and not why_low, "why_low": why_low,
           "say": say, "seconds": round(time.time() - t_start, 1)}
    json.dump(dec, open(os.path.join(d, "decision.json"), "w"), indent=1, ensure_ascii=False, default=float)
    print(f"[decide] {say}" + (f"  (要確認: {'; '.join(why_low)})" if why_low else ""), flush=True)
    return 0 if chosen else 1


if __name__ == "__main__":
    raise SystemExit(main())
