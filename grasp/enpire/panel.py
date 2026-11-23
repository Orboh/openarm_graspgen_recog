#!/usr/bin/env python3
"""Operator panel for the ENPIRE live trials (localhost, stdlib + numpy only).

    python3 grasp/enpire/panel.py               # -> http://localhost:8770
    python3 grasp/enpire/panel.py --port 8771
    Thor (2026-11-21, all compute on Thor): grasp/tools/thor_servers.sh start also starts this panel on the Thor
    Tailscale address :8770 with grasp/config/sites/tokyo-thor-local.env loaded -> open http://100.124.57.58:8770 on the PC

The buttons run the same commands as the CLI (`python -m grasp.enpire.live new-iter|prepare|execute|batch|close`,
`grasp/tools/thor_snap.py`) as subprocesses, so every record is written by live.py exactly as in a CLI session:
grasp/runs/enpire/iter_NNN/iteration.json, trials/NN_<cap>/{result.json,overlay.jpg,plan.png,run.log,traj.csv,wrist_*.jpg},
events.jsonl, best.json. The panel adds only:
  - events.jsonl  kind=panel_job_start / panel_job_end (which button, args, exit code)
  - events.jsonl  kind=operator_note (free note, e.g. how the object was placed)
  - iteration.json "finding" (the operator's conclusion, written after `close`)
  - iter_NNN/panel_jobs.log (full stdout of every job of that iteration; snapshots go to enpire/panel_jobs.log)

Motion: only "実行 (GO)" and "バッチ (GO)" move the arm. Both ask for confirmation and pass --i-have-operator-go.
A running execute cannot be cancelled from here (the arm runs in tmux `enpire` on Thor); batch stops after the
current trial via the STOP file.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
from grasp.enpire import dashboard, loop  # noqa: E402

RUNS = os.path.join(REPO, "grasp", "runs")
ROOT = loop.ROOT
GGX = os.path.expanduser("~/work/GraspGenX")
PY = os.path.join(GGX, ".venv", "bin", "python")
ENV = {**os.environ, "PYTHONPATH": f"{REPO}:{GGX}", "PYTHONUNBUFFERED": "1"}
MOVES = {"execute", "batch"}

JOB: dict = {}
LOCK = threading.Lock()


# ------------------------------------------------------------------ jobs
def _iter_dir(it):
    if not it or not it.startswith("iter_") or "/" in it:
        raise ValueError(f"bad iteration {it!r}")
    d = os.path.join(ROOT, it)
    if not os.path.isdir(d):
        raise ValueError(f"no such iteration {it}")
    return d


def build(cmd, f):
    """Structured form fields -> argv (no shell). Returns (argv, iteration or None)."""
    live = [PY, "-u", "-m", "grasp.enpire.live"]
    s = lambda k: str(f.get(k) or "").strip()  # noqa: E731
    if cmd == "busy":
        return live + ["busy"], None
    if cmd == "servers":  # Thor: perceive :5558 / GraspGenX :5556 / cuRobo :5559 / dashboard (grasp/tools/thor_servers.sh)
        act = s("action") or "status"
        if act not in ("start", "status"):
            raise ValueError("servers: start か status")
        return ["bash", os.path.join(REPO, "grasp", "tools", "thor_servers.sh"), act], None
    if cmd == "snap":
        b = busy_now()
        if b:  # prepare checks this inside live.py; a bare snapshot would otherwise grab the chest camera mid-capture
            raise ValueError("Thor で腕かカメラが使用中です（別のセッション？）: " + " / ".join(b)[:200])
        return [PY, "-u", "grasp/tools/thor_snap.py", "--no-view", *s("snap_args").split()], None
    if cmd == "new-iter":
        if not s("hypothesis"):
            raise ValueError("仮説を書いてください")
        a = live + ["new-iter", "--hypothesis", s("hypothesis"), "--budget", str(int(f.get("budget") or 10)),
                    "--plan-args=" + s("plan_args"), "--executor", s("executor") or "grasp/run_grasp.py",
                    "--exec-args=" + s("exec_args")]
        if s("home"):
            a += ["--home", s("home")]
        if f.get("auto_place"):
            a.append("--auto-place")
        return a, None
    it = s("iter")
    d = _iter_dir(it)
    status = (dashboard._j(os.path.join(d, "iteration.json"), {}) or {}).get("status")
    if cmd in ("prepare", "execute", "batch") and status != "running":
        # live.py would append to a closed iteration; its close() numbers would then no longer match its trials
        raise ValueError(f"{it} は締め済み（{status}）です。②で新しい回を開いてください")
    if cmd == "prepare":
        a = live + ["prepare", "--iter", it, "--snap-args=" + s("snap_args")]
        if s("place_offset"):
            a.append("--place-offset=" + s("place_offset"))
        return a, it
    if cmd == "execute":
        t = trial(it, int(f["trial"]))
        if t["stage"] != "prepared" or not t["ready"]:
            raise ValueError(f"{t['name']} は実行できません（{t['stage']} / {t['cause']}）。置き直して準備してください")
        return live + ["execute", "--iter", it, "--trial", str(int(f["trial"])), "--i-have-operator-go"], it
    if cmd == "batch":
        return live + ["batch", "--iter", it, "--n", str(int(f.get("n") or 10)),
                       "--snap-args=" + s("snap_args"), "--i-have-operator-go"], it
    if cmd == "close":
        return live + ["close", "--iter", it], it
    if cmd == "observe":  # the operator's eye decides "lifted" (the gripper-angle check misreads bars, 2026-11-21)
        t = trial(it, int(f["trial"]))
        if t["stage"] != "executed":
            raise ValueError(f"{t['name']} はまだ実行していません")
        a = live + ["observe", "--iter", it, "--trial", str(int(f["trial"])), "--lifted", "yes" if f.get("lifted") else "no",
                    "--note=" + s("note"), "--completed"]
        return a, it
    raise ValueError(f"unknown command {cmd}")


def _run(job, finding):
    log_path = os.path.join(ROOT, job["iter"], "panel_jobs.log") if job["iter"] else os.path.join(ROOT, "panel_jobs.log")
    with open(log_path, "a") as lf:
        lf.write(f"\n===== {job['started']} {job['cmd']} {json.dumps(job['argv'][3:], ensure_ascii=False)}\n")
        try:
            p = subprocess.Popen(job["argv"], cwd=REPO, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
            job["proc"] = p
            for line in p.stdout:
                lf.write(line)
                lf.flush()
                if "warn" in line.lower():
                    continue
                job["lines"].append(line.rstrip("\n"))
                del job["lines"][:-400]
            job["rc"] = p.wait()
        except Exception as e:  # noqa: BLE001
            job["lines"].append(f"[panel] {type(e).__name__}: {e}")
            job["rc"] = -1
        lf.write(f"===== exit {job['rc']}\n")
    if job["cmd"] == "new-iter" and job["rc"] == 0:
        its = [l for l in job["lines"] if l.startswith("iter_")]
        job["iter"] = its[-1].strip() if its else None
    if job["cmd"] == "close" and job["rc"] == 0 and finding:
        set_finding(job["iter"], finding)
        job["lines"].append("[panel] finding を iteration.json に書きました")
    job["ended"] = loop.now()
    job.pop("proc", None)
    loop.event("panel_job_end", job=job["cmd"], iter=job["iter"], rc=job["rc"])


def start(cmd, f):
    with LOCK:
        if JOB and JOB.get("ended") is None:
            raise ValueError(f"実行中のジョブがあります: {JOB['cmd']}")
        argv, it = build(cmd, f)
        if cmd in MOVES and f.get("confirm") != "GO":
            raise ValueError("腕が動くコマンドには確認が必要です")
        JOB.clear()
        JOB.update(cmd=cmd, argv=argv, iter=it, started=loop.now(), ended=None, rc=None, lines=[])
        loop.event("panel_job_start", job=cmd, iter=it, args=argv[3:])
        threading.Thread(target=_run, args=(JOB, str(f.get("finding") or "").strip()), daemon=True).start()
        return {"ok": True}


def cancel():
    with LOCK:
        if not JOB or JOB.get("ended") is not None:
            raise ValueError("実行中のジョブはありません")
        if JOB["cmd"] in MOVES:
            raise ValueError("腕の動作中はここから止められません（バッチは STOP、緊急時は非常停止）")
        p = JOB.get("proc")
        if p:
            p.terminate()
        return {"ok": True}


# ------------------------------------------------------------------ records the panel adds
def set_finding(it, text):
    p = os.path.join(_iter_dir(it), "iteration.json")
    m = json.load(open(p))
    m["finding"] = text
    m["finding_by"] = "operator (panel) " + loop.now()
    loop.wjson(p, m)
    loop.event("finding", iter=it, finding=text)


def note(it, text):
    if not text.strip():
        raise ValueError("メモが空です")
    if it:
        _iter_dir(it)
    loop.event("operator_note", iter=it or None, note=text.strip())
    return {"ok": True}


def stop_batch():
    open(os.path.join(ROOT, "STOP"), "w").write(loop.now())
    loop.event("panel_stop_file", iter=JOB.get("iter"))
    return {"ok": True}


# ------------------------------------------------------------------ read side
def defaults():
    """Form defaults = params of the latest iteration (the next run usually changes one thing)."""
    its = sorted(glob.glob(os.path.join(ROOT, "iter_*", "iteration.json")))
    p = {}
    if its:
        p = (json.load(open(its[-1])).get("params") or {})
    homes = sorted(os.path.relpath(h, REPO) for h in glob.glob(os.path.join(REPO, "grasp", "config", "*.yaml")))
    src = (json.load(open(its[-1])).get("source") or "") if its else ""
    snap = "--perceive localhost:5558 --band 0.25,0.56,-0.20,0.35 --hmax 0.10" if "pile" in src else "--hmax 0.10"  # pile: GDINO+SAM2
    return {"snap_args": snap, "plan_args": p.get("plan_args", ""), "exec_args": p.get("exec_args", ""),
            "executor": p.get("executor", "grasp/run_grasp.py"), "home": p.get("home", ""),
            "auto_place": bool(p.get("auto_place")), "homes": homes}


def trial(it, k):
    d = _iter_dir(it)
    t = sorted(glob.glob(os.path.join(d, "trials", f"{int(k):02d}_*")))
    if not t:
        raise ValueError("no such trial")
    r = dashboard._j(os.path.join(t[0], "result.json"), {}) or {}
    rel = os.path.relpath(t[0], RUNS)
    imgs = [f"/files/{rel}/{n}" for n in ("overlay.jpg", "plan.png", "wrist_align2.jpg", "wrist_at_grasp.jpg",
                                         "wrist_hold.jpg") if os.path.exists(os.path.join(t[0], n))]
    info = r.get("info") or {}
    # step-by-step view (2026-11-21, Sota: 判別器や IK がどこでどう動いているか)
    f = lambda n: f"/files/{rel}/{n}" if os.path.exists(os.path.join(t[0], n)) else None  # noqa: E731
    pj = dashboard._j(os.path.join(t[0], "perceive.json"), None)
    perceive = None if not pj else {
        "cands": [{k2: c.get(k2) for k2 in ("label", "alu", "crowd", "aspect", "score")} for c in pj.get("cands", [])],
        "chosen": {k2: (pj.get("chosen") or {}).get(k2) for k2 in ("alu", "crowd", "label")} if pj.get("chosen") else None,
        "points_from": pj.get("points_from"), "seconds": pj.get("seconds"), "img": f("perceive.jpg")}
    stats = dashboard._j(os.path.join(t[0], "plan_stats.json"), None) or []
    for st in stats:
        st["img"] = f(st.get("image", ""))
        st["advice_img"] = f(st["advice_image"]) if st.get("advice_image") else None
    phases = []
    log = os.path.join(t[0], "run.log")
    if os.path.exists(log):
        import re
        for line in open(log, errors="replace"):
            m = re.match(r"\[run\] (.+?) done \(max joint error ([0-9.]+)°\)", line.strip())
            if m:
                phases.append({"phase": m.group(1), "err": float(m.group(2))})
            elif line.startswith("[run]") and any(k2 in line for k2 in ("held check", "回り込む", "STOP", "nothing held", "Stopping", "cuRobo")):
                phases.append({"phase": line.strip()[6:], "err": None})
    detours = [l for l in (info.get("dry_check") or "").splitlines() if "回り込む" in l]
    dj = dashboard._j(os.path.join(t[0], "decision.json"), None)
    decision = None if not dj else {"say": dj.get("say"), "confident": dj.get("confident"), "why_low": dj.get("why_low"),
                                    "objects": dj.get("objects"), "relations": dj.get("relations"), "tried": dj.get("tried"),
                                    "vlm_scene": (dj.get("vlm") or {}).get("scene"), "vlm_error": (dj.get("vlm") or {}).get("error"),
                                    "vlm_seconds": (dj.get("vlm") or {}).get("seconds"), "seconds": dj.get("seconds"),
                                    "img": f("decision.jpg")}
    return {"name": os.path.basename(t[0]), "stage": r.get("stage"), "ready": r.get("ready_to_execute"),
            "success": r.get("success"), "cause": r.get("cause"), "score": r.get("score"),
            "predicates": {k2: v.get("status") for k2, v in (r.get("predicates") or {}).items()},
            "plan": info.get("plan"), "dry_check": info.get("dry_check"), "snap": info.get("snap"),
            "plan_log": info.get("plan_log"), "gripper_at_hold": r.get("gripper_at_hold_deg"),
            "completed": r.get("run_completed"), "imgs": imgs,
            "steps": {"decision": decision, "snap_img": f("overlay.jpg"), "perceive": perceive, "plan_stats": stats, "plan_img": f("plan.png"),
                      "detours": detours, "phases": phases, "wrist": f("wrist_at_grasp.jpg")}}


def busy_now():
    r = subprocess.run([PY, "-m", "grasp.enpire.live", "busy"], cwd=REPO, env=ENV, capture_output=True, text=True, timeout=25)
    if r.returncode != 0:
        raise ValueError("Thor に届かないので使用中か確かめられません")
    out = [l for l in r.stdout.strip().splitlines() if "warn" not in l.lower()]
    return [] if out == ["idle"] else out


def others_active(events, job):
    """Recent records not made by a panel job -> someone else (a CLI / Claude session) is driving the loop."""
    if job and job.get("ended") is None:
        return None
    for e in events[:10]:
        if e.get("kind", "").startswith(("panel_", "operator_note", "finding")):
            continue
        t = dt.datetime.fromisoformat(e["t"])
        if (dt.datetime.now() - t).total_seconds() < 120:
            return f"{e['t'][11:]} {e['kind']} {e.get('iter') or ''}"
        return None
    return None


def health():
    import socket
    def up(port):
        s = socket.socket()
        s.settimeout(1.0)
        ok = s.connect_ex(("localhost", port)) == 0
        s.close()
        return ok
    ggx = up(5556)
    servers = {"perceive": up(5558), "graspgenx": ggx, "curobo": up(5559), "vlm": up(5560)}
    try:
        r = subprocess.run([PY, "-m", "grasp.enpire.live", "busy"], cwd=REPO, env=ENV, capture_output=True,
                           text=True, timeout=25)
        out = [l for l in r.stdout.strip().splitlines() if "warn" not in l.lower()]
        thor = r.returncode == 0
        busy = [] if out == ["idle"] else out
        err = None if thor else (r.stderr.strip().splitlines() or ["?"])[-1]
    except Exception as e:  # noqa: BLE001
        thor, busy, err = False, [], str(e)
    return {"graspgenx": ggx, "servers": servers, "local": os.environ.get("GRASP_ROBOT_HOST") == "local",
            "curobo_env": os.environ.get("GRASP_CUROBO", ""), "arm": os.environ.get("GRASP_ARM", "left"),
            "thor_ssh": thor, "busy": busy, "error": err, "t": loop.now()}


def api_state():
    st = dashboard.state()
    st["events"] = st["events"][:25]
    job = {k: v for k, v in JOB.items() if k != "proc"} if JOB else None
    latest = os.path.join(RUNS, "latest.jpg")
    return {**st, "job": job, "defaults": defaults(), "others_active": others_active(st["events"], JOB), "stop_file": os.path.exists(os.path.join(ROOT, "STOP")),
            "latest_jpg": f"/files/latest.jpg?{int(os.path.getmtime(latest))}" if os.path.exists(latest) else None}


# ------------------------------------------------------------------ page
PAGE = os.path.join(HERE, "panel.html")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if not isinstance(body, bytes):
            body = json.dumps(dashboard._clean(body), default=float, allow_nan=False, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/":
                return self._send(200, open(PAGE, "rb").read(), "text/html; charset=utf-8")
            if u.path == "/api/state":
                return self._send(200, api_state())
            if u.path == "/api/health":
                return self._send(200, health())
            if u.path == "/api/trial":
                return self._send(200, trial(q.get("iter"), q.get("trial", 0)))
            if u.path.startswith("/files/"):
                f = os.path.realpath(os.path.join(RUNS, unquote(u.path[len("/files/"):])))
                if f.startswith(os.path.realpath(RUNS) + os.sep) and os.path.isfile(f):
                    ct = {".jpg": "image/jpeg", ".png": "image/png"}.get(os.path.splitext(f)[1], "text/plain; charset=utf-8")
                    return self._send(200, open(f, "rb").read(), ct)
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        u = urlparse(self.path)
        origin, host = self.headers.get("Origin"), self.headers.get("Host")
        if origin not in (None, f"http://localhost:{self.server.server_port}", f"http://127.0.0.1:{self.server.server_port}",
                          f"http://{host}"):  # same-origin only (no other site can press the buttons)
            return self._send(403, {"error": "cross-origin request refused"})
        n = int(self.headers.get("Content-Length") or 0)
        try:
            f = json.loads(self.rfile.read(n) or b"{}")
            if u.path.startswith("/api/run/"):
                return self._send(200, start(u.path[len("/api/run/"):], f))
            if u.path == "/api/cancel":
                return self._send(200, cancel())
            if u.path == "/api/stop":
                return self._send(200, stop_batch())
            if u.path == "/api/note":
                return self._send(200, note(f.get("iter") or "", f.get("text") or ""))
            if u.path == "/api/finding":
                set_finding(f.get("iter"), str(f.get("text") or "").strip())
                return self._send(200, {"ok": True})
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            return self._send(400, {"error": str(e)})
        self._send(404, {"error": "not found"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    os.makedirs(ROOT, exist_ok=True)
    if not os.path.exists(PY):
        print(f"[panel] warning: {PY} not found (GraspGenX venv); the buttons will fail", flush=True)
    print(f"[panel] http://localhost:{a.port}  (records -> {ROOT})", flush=True)
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
