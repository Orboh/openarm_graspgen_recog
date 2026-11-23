#!/usr/bin/env python3
"""ENPIRE live trials on the real OpenArm left arm. Reuses the step-1..6 tools; only `execute` moves the arm.

One live trial = prepare (no motion) -> operator GO -> execute (motion) -> verify:
    new-iter  --hypothesis "..."            open a live iteration (records params + hand-eye)
    prepare   --iter iter_NNN               busy check -> read q (CAN read-only) -> thor_snap (chest capture +
                                            cut-out) -> plan.py (GraspGenX -> IK -> floor checks) -> run_grasp dry
                                            check on Thor -> verifier at plan level. Writes trials/NN_<cap>/
    execute   --iter iter_NNN --trial NN --i-have-operator-go
                                            run_grasp.py --go on Thor (arm moves ~1 min), fetch traj.csv,
                                            judge `lifted` from the gripper angle while holding the lift
    close     --iter iter_NNN               success rate, failure histogram, keep/revert vs best of this track

Why `lifted` from the gripper: run_grasp closes to +2 deg; on an empty close the jaws reach ~-1..-3 deg,
with a 5-7 cm block between them they stall around -30 deg. The hold row of traj.csv is read 2 s into
the 10 cm lift, so a stalled gripper there means the object came up with the hand.

    PY="env PYTHONPATH=$HOME/work/GraspGenX:$HOME/work/openarm-orboh $HOME/work/GraspGenX/.venv/bin/python"
    $PY -m grasp.enpire.live new-iter --hypothesis "live baseline: current plan.py + run_grasp.py"
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import subprocess
import time
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
from grasp import remote  # noqa: E402
from grasp.enpire import contract, loop  # noqa: E402

THOR = remote.HOST
T_BC_PATH = os.environ.get("GRASP_T_BASE_CAM", os.path.join(REPO, "grasp", "config", "T_base_cam.json"))
GPY = os.path.expanduser("~/work/GraspGenX/.venv/bin/python")
ENV = {**os.environ, "PYTHONPATH": f"{REPO}:{os.path.expanduser('~/work/GraspGenX')}"}
HELD_MAX_DEG = -8.0   # gripper.pos at the hold row below this = jaws stopped on something (closed cmd is +2)
LIFT_M = 0.10         # run_grasp lifts the grasp pose 10 cm straight up

READ_Q = r'''
import can,time,numpy as np,json,os
cal=json.load(open(os.path.expanduser("~/.cache/huggingface/lerobot/calibration/robots/openarm_follower/@ROBOT_ID@.json")))
bus=can.Bus(channel="@CAN_PORT@",interface="socketcan",fd=True); q=[]
for sid in range(1,8):
    bus.send(can.Message(arbitration_id=0x7FF,data=bytes([sid,0,0xCC,0,0,0,0,0]),is_extended_id=False)); end=time.monotonic()+0.1
    while time.monotonic()<end:
        m=bus.recv(timeout=0.01)
        if m and m.arbitration_id==0x10+sid: d=bytes(m.data); q.append(np.degrees(((d[1]<<8)|d[2])/65535*25-12.5)-cal[f"joint_{sid}"]["homing_offset"]); break
bus.shutdown(); print(",".join(f"{v:.1f}" for v in q))
'''
# 腕の CAN と較正の名前（既定は東京の Thor の左フォロワー）。ロボット側には環境変数が届かないので、ここで埋めて送る
READ_Q = (READ_Q.replace("@CAN_PORT@", os.environ.get("GRASP_CAN_PORT", "can_fol_l"))
          .replace("@ROBOT_ID@", os.environ.get("GRASP_ROBOT_ID", "follower_left")))


os.environ.setdefault("OPENARM_REPO", REPO)  # run_grasp.py dry checks on this PC use this checkout


def _inproc(modname, argv):
    """Run a tool's main(argv) inside this process (imports stay loaded between trials -> seconds saved per trial)."""
    import contextlib
    import importlib
    import io
    mod = importlib.import_module(modname)
    buf = io.StringIO()
    rc = 0
    with contextlib.redirect_stdout(buf):
        try:
            rc = mod.main(argv) or 0
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 1
            print(e.code)
        except Exception as e:  # noqa: BLE001
            rc = 1
            print(f"{type(e).__name__}: {e}")
    return rc, buf.getvalue()


def ssh(cmd, timeout=60, check=True):
    r = remote.run(cmd, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"ssh failed ({r.returncode}): {cmd[:80]}\n{r.stderr[-800:]}")
    return r.stdout


_DEPLOYED = set()


def deploy_executor(rel):
    """Copy the executor (and the wrist jaw model it may need) to Thor ~/grasp_work/grasp/ (once per process)."""
    if rel in _DEPLOYED:
        return os.path.basename(rel)
    _DEPLOYED.add(rel)
    ssh("mkdir -p ~/grasp_work/grasp/config")
    remote.put(os.path.join(REPO, rel), "grasp_work/grasp/")
    remote.put(os.path.join(REPO, "grasp", "config", "wrist_jaw_model.json"), "grasp_work/grasp/config/")
    return os.path.basename(rel)


def busy():
    """Anything else driving the arm or holding the chest camera on Thor? (another session's tools)"""
    out = ssh("pgrep -af 'run_grasp.py|goto_hold.py|handeye_auto.py|handeye_collect.py|capture.py|record_bimanual|run_rollout' "
              "| grep -v pgrep || true")
    return [l for l in out.splitlines() if l.strip() and "sleep 3600" not in l and "tee " not in l]


def iter_dir(it):
    d = os.path.join(loop.ROOT, it)
    if not os.path.isdir(d):
        raise SystemExit(f"no such iteration {d}")
    return d


# ------------------------------------------------------------------ commands
def cmd_new_iter(a):
    from grasp.enpire import policy
    d = loop.next_iter_dir()
    it = os.path.basename(d)
    src = "live:graspnet" if "--model graspnet" in (a.plan_args or "") else "live"  # separate track per grasp model (step 7 comparison)
    best = loop.best_record(src, T_BC_PATH)
    meta = {"iter": it, "hypothesis": a.hypothesis, "source": src, "parent": best and best["iter"],
            "params": {**policy.load(None), "planner": "grasp/plan.py", "executor": a.executor, "home": a.home,
                       "exec_args": a.exec_args, "auto_place": a.auto_place, "drop": a.drop,
                       "plan_args": a.plan_args, "decide": bool(getattr(a, "decide", False)),
                       "other_args": getattr(a, "other_args", "")},
            "t_base_cam": T_BC_PATH, "verifier_version": contract.VERIFIER_VERSION,
            "gate": contract.SUCCESS_GATE, "budget": a.budget, "started": loop.now(), "status": "running"}
    loop.wjson(os.path.join(d, "iteration.json"), meta)
    loop.event("iter_start", iter=it, hypothesis=a.hypothesis, source=src, budget=a.budget)
    print(it)


def cmd_prepare(a):
    d = iter_dir(a.iter)
    meta = json.load(open(os.path.join(d, "iteration.json")))
    b = busy()
    if b:
        loop.event("trial_blocked", iter=a.iter, reason="arm/camera busy on Thor", procs=b)
        raise SystemExit("[live] Thor is busy (another session?):\n  " + "\n  ".join(b))
    q_home = None
    if meta["params"].get("home"):  # plan from the home pose (the arm goes there first)
        import yaml
        hy = yaml.safe_load(open(os.path.join(REPO, meta["params"]["home"])))
        q_home = ",".join(f"{float(hy[f'joint_{i}']):.1f}" for i in range(1, 8))
    prev = sorted(glob.glob(os.path.join(d, "trials", "*", "run.log")))
    held = q_home is not None and prev and "holding (torque on)" in open(prev[-1]).read()
    if held:  # the arm is held at home since the last trial: no need to read it
        q = q_home
    else:
        q = ssh("~/projects/lerobot/.venv/bin/python - <<'EOF'\n" + READ_Q + "EOF", timeout=30).strip().splitlines()[-1]
        if len(q.split(",")) != 7:
            raise SystemExit(f"[live] could not read 7 joint angles: {q!r}")
    q_rest = q
    if q_home is not None:
        q = q_home
    src, out = _inproc("grasp.tools.thor_snap", ["--no-view", "--fast", *a.snap_args.split()])
    line = [l for l in out.splitlines() if " | " in l]
    if src != 0 or not line:
        raise SystemExit(f"[live] thor_snap failed:\n{out[-600:]}")
    cap = line[-1].split(" | ")[0].strip()
    cap_dir = os.path.join(REPO, "grasp", "runs", cap)
    k = len(glob.glob(os.path.join(d, "trials", "*")))
    tdir = os.path.join(d, "trials", f"{k:02d}_{cap}")
    os.makedirs(tdir)
    loop.event("trial_prepare", iter=a.iter, trial=k, scene=cap, q_start=q)

    obs, info = {"obj": None}, {"scene": cap, "scene_dir": cap_dir, "q_start_deg": q, "snap": line[-1]}
    if os.path.exists(os.path.join(cap_dir, "object.npy")):
        obs["obj"] = np.load(os.path.join(cap_dir, "object.npy"))
    else:
        obs["seg_error"] = line[-1].split(" | ", 1)[-1]
    plan_ok = False
    if meta["params"].get("decide"):
        # 2026-11-21: decide what to take first (all objects + image-language model + planner check), grasp/tools/decide.py
        for f in ("object.npy", "plan.json"):
            if os.path.exists(os.path.join(cap_dir, f)):
                os.remove(os.path.join(cap_dir, f))
        drc, dout = _inproc("grasp.tools.decide", [cap_dir, f"--q={q}", "--plan-args=" + meta["params"]["plan_args"],
                                                  *(["--other-args=" + meta["params"]["other_args"]] if meta["params"].get("other_args") else [])])
        info["plan_log"] = "\n".join(l for l in dout.splitlines() if "INFO" not in l and "warn" not in l.lower())[-2500:]
        dj = os.path.join(cap_dir, "decision.json")
        if os.path.exists(dj):
            dec = json.load(open(dj))
            info["decision"] = {k2: dec.get(k2) for k2 in ("say", "confident", "why_low", "chosen", "seconds")}
        plan_ok = drc == 0 and os.path.exists(os.path.join(cap_dir, "plan.json"))
        obs["obj"] = np.load(os.path.join(cap_dir, "object.npy")) if plan_ok else None
        if not plan_ok:
            obs["seg_error"] = (info.get("decision") or {}).get("say") or "decide: nothing to take"
    elif obs["obj"] is not None:
        if os.path.exists(os.path.join(cap_dir, "plan.json")):
            os.remove(os.path.join(cap_dir, "plan.json"))
        prc, pout = _inproc("grasp.plan", [cap_dir, f"--q={q}", "--reinfer", *meta["params"]["plan_args"].split()])
        info["plan_log"] = "\n".join(l for l in pout.splitlines() if "INFO" not in l)[-1500:]
        plan_ok = prc == 0 and os.path.exists(os.path.join(cap_dir, "plan.json"))
    T_bc = np.array(json.load(open(T_BC_PATH))["T_base_cam"])
    if plan_ok:
        plan = json.load(open(os.path.join(cap_dir, "plan.json")))
        # block extents along the grasp axes (chest capture) -> the wrist-camera servo needs the Yg width
        ob = obs["obj"] @ T_bc[:3, :3].T + T_bc[:3, 3]
        Tgb = np.array(plan["T_grasp_base"])
        plan["block_extent_Yg_m"] = float(np.ptp(ob @ Tgb[:3, 1]))
        plan["block_extent_Xg_m"] = float(np.ptp(ob @ Tgb[:3, 0]))
        plan["exclude_wood"] = "--exclude-wood" in a.snap_args
        plan["exclude_bright"] = "--exclude-bright" in a.snap_args
        if a.place_offset:
            plan["place_offset_m"] = [float(v) for v in a.place_offset.split(",")]
        plan["block_height_m"] = float(np.percentile(ob[:, 2], 98) - np.percentile(ob[:, 2], 2))
        json.dump(plan, open(os.path.join(cap_dir, "plan.json"), "w"), indent=2)
        if meta["params"].get("drop"):  # throw aside beside the left shoulder after the lift (grasp/tools/plan_drop.py)
            rc, dout = _inproc("grasp.tools.plan_drop", [cap_dir, "--at", meta["params"]["drop"]])
            info["drop"] = [l for l in dout.splitlines() if l.startswith("[drop]")][-1:] or [dout[-300:]]
            drop_fail = rc != 0
            plan = json.load(open(os.path.join(cap_dir, "plan.json")))
        if meta["params"].get("auto_place"):
            _, ppout = _inproc("grasp.tools.plan_place", [cap_dir, "--seed", str(k)])
            info["place"] = [l for l in ppout.splitlines() if l.startswith("[place]")][-1:] or [ppout[-300:]]
            plan = json.load(open(os.path.join(cap_dir, "plan.json")))
        # plan.py may have swapped to another aluminium frame (object_alt_* -> object.npy): judge the bar it planned on
        obs["obj"] = np.load(os.path.join(cap_dir, "object.npy"))
        G_cam = np.linalg.inv(T_bc) @ np.array(plan["T_grasp_base"])
        obs.update(grasp=G_cam, score=plan["score"],
                   ik={"converged": True, "pos_err_m": 0.0, "rot_err_deg": 0.0})
        if meta["params"].get("drop") and drop_fail:  # no reachable drop pose -> not executable (counts as IK)
            obs["ik"] = {"converged": False, "pos_err_m": float("nan"), "rot_err_deg": float("nan")}
        info["plan"] = {k2: plan[k2] for k2 in ("rank", "score", "tilt_deg", "tcp_to_object_m", "q_pre_deg",
                                               "q_grasp_deg", "q_lift_deg")}
        # dry check on Thor (FK floor checks of every segment, no motion)
        ssh(f"mkdir -p ~/grasp_runs/{cap} ~/grasp_work/grasp")
        remote.put(os.path.join(cap_dir, "plan.json"), f"grasp_runs/{cap}/")
        exe = deploy_executor(meta["params"]["executor"])
        homearg = ""
        if meta["params"].get("home"):
            if meta["params"]["home"] not in _DEPLOYED:
                remote.put(os.path.join(REPO, meta["params"]["home"]), "grasp_work/grasp/config/")
                _DEPLOYED.add(meta["params"]["home"])
            homearg = f" --home ~/grasp_work/grasp/config/{os.path.basename(meta['params']['home'])}"
        if exe == "run_grasp.py":  # same FK floor checks, on this PC (no ssh / interpreter start on Thor)
            largs = [cap_dir, f"--start={q_rest}"]
            if meta["params"].get("home"):
                largs += ["--home", os.path.join(REPO, meta["params"]["home"])]
            _, dout = _inproc("grasp.run_grasp", largs)
            ls_ = dout.strip().splitlines()
            # keep the cuRobo detours too (the panel shows them before GO)
            dry = "\n".join([l for l in ls_[:-4] if "回り込む" in l or "Stopping" in l] + ls_[-4:])
        else:
            dry = ssh(f"cd ~/grasp_work && {remote.env_assign()}~/projects/lerobot/.venv/bin/python grasp/{exe} ~/grasp_runs/{cap} "
                      f"--start={q_rest}{homearg} 2>&1 | grep -v Warn | tail -4", timeout=120, check=False)
        info["dry_check"] = dry.strip()
        if "DRY RUN" not in dry:
            obs["ik"] = {"converged": False, "pos_err_m": float("nan"), "rot_err_deg": float("nan")}
            info["dry_check_failed"] = True
    elif obs["obj"] is not None:
        cache = os.path.join(cap_dir, "grasps_cam.npz")
        if os.path.exists(cache):  # grasps exist, none plannable -> attribute to reachability, not to the grasp net
            z = np.load(cache)
            i0 = int(np.argmax(z["S"]))
            obs["grasp"], obs["score"] = z["G"][i0], float(z["S"][i0])
        else:
            obs["infer_error"] = "plan.py produced no grasps"
        obs["ik"] = {"converged": False, "pos_err_m": float("nan"), "rot_err_deg": float("nan")}
        info["plan_failed"] = "no candidate with pregrasp/grasp/lift all solvable"
    # table plane in camera frame for the top_down predicate (same floor fit plan.py uses)
    try:
        sys.path.insert(0, os.path.join(REPO, "grasp"))
        from grasp.plan import floor_plane_base
        n_b, d_b = floor_plane_base(cap_dir, T_bc)
        n_c = T_bc[:3, :3].T @ n_b
        obs["plane"] = [*n_c.tolist(), float(d_b + n_b @ T_bc[:3, 3])]
    except Exception as e:
        info["plane_error"] = str(e)
    res = contract.evaluate(obs)
    res.update(info=info, stage="prepared", ready_to_execute=bool(res["success"]),
               grasp=None if obs.get("grasp") is None else np.asarray(obs["grasp"]).tolist(), score=obs.get("score"))
    if os.path.exists(os.path.join(cap_dir, "overlay.jpg")):
        import shutil
        shutil.copy(os.path.join(cap_dir, "overlay.jpg"), os.path.join(tdir, "overlay.jpg"))
    if os.path.exists(os.path.join(cap_dir, "plan.png")):
        import shutil
        shutil.copy(os.path.join(cap_dir, "plan.png"), os.path.join(tdir, "plan.png"))
    import shutil
    for f in ["perceive.jpg", "perceive.json", "plan_stats.json", "decision.json", "decision.jpg", "objects.jpg"] + [os.path.basename(x) for x in glob.glob(os.path.join(cap_dir, "plan_funnel_*.jpg")) + glob.glob(os.path.join(cap_dir, "advice_*.jpg"))]:
        if os.path.exists(os.path.join(cap_dir, f)):  # the panel's step-by-step view (2026-11-21)
            shutil.copy(os.path.join(cap_dir, f), os.path.join(tdir, f))
    loop.wjson(os.path.join(tdir, "result.json"), res)
    loop.event("trial_prepared", iter=a.iter, trial=k, scene=cap, ready=res["ready_to_execute"], cause=res["cause"])
    print(json.dumps({"trial": os.path.basename(tdir), "ready": res["ready_to_execute"], "cause": res["cause"],
                      "plan": info.get("plan"), "dry_check": info.get("dry_check")}, indent=1, default=float))


def cmd_execute(a):
    if not a.i_have_operator_go:
        raise SystemExit("[live] refused: pass --i-have-operator-go only after the operator said GO for THIS trial")
    d = iter_dir(a.iter)
    tdir = sorted(glob.glob(os.path.join(d, "trials", f"{int(a.trial):02d}_*")))[0]
    res = json.load(open(os.path.join(tdir, "result.json")))
    if not res.get("ready_to_execute"):
        raise SystemExit(f"[live] trial not ready: {res['cause']}")
    b = busy()
    if b:
        raise SystemExit("[live] Thor is busy:\n  " + "\n  ".join(b))
    cap = res["info"]["scene"]
    meta = json.load(open(os.path.join(d, "iteration.json")))
    exe = deploy_executor(meta["params"].get("executor", "grasp/run_grasp.py"))
    homearg = ""
    if meta["params"].get("home"):
        if meta["params"]["home"] not in _DEPLOYED:
            remote.put(os.path.join(REPO, meta["params"]["home"]), "grasp_work/grasp/config/")
            _DEPLOYED.add(meta["params"]["home"])
        homearg = f" --home ~/grasp_work/grasp/config/{os.path.basename(meta['params']['home'])}"
    loop.event("trial_execute", iter=a.iter, trial=a.trial, scene=cap, executor=exe)
    ssh("tmux kill-session -t enpire 2>/dev/null; "
        f"tmux new -d -s enpire 'cd ~/grasp_work && {remote.env_assign()}~/projects/lerobot/.venv/bin/python -u grasp/{exe} "
        f"~/grasp_runs/{cap} --go{homearg}{' --wrist-snap' if exe == 'run_grasp.py' else ''} {meta['params'].get('exec_args', '')} 2>&1 | tee ~/grasp_runs/{cap}/run.log; sleep 600'")
    import time
    log = ""
    t_end = time.time() + 600  # 2026-11-21: a fixed 80 polls (~2.5 min with ssh) cut off the longer drop runs -> "(h) aborted"
    while time.time() < t_end:
        time.sleep(0.5)
        log = ssh(f"grep -v -i warn ~/grasp_runs/{cap}/run.log 2>/dev/null || true", check=False)
        if any(s in log for s in ("torque off", "holding (torque on)", "Traceback")):
            break
        alive = ssh(f"pgrep -f 'grasp/{exe} .*{cap}' >/dev/null && echo yes || echo no", check=False).strip()
        if alive == "no" and log:  # the executor exited without its end line (crash / kill): stop waiting
            time.sleep(1.0)
            log = ssh(f"grep -v -i warn ~/grasp_runs/{cap}/run.log 2>/dev/null || true", check=False)
            break
    traj = ssh(f"cat ~/grasp_runs/{cap}/traj.csv 2>/dev/null || true", check=False)
    open(os.path.join(tdir, "run.log"), "w").write(log)
    for pat in ("wrist_*.jpg", "servo.json"):  # 無くてもよい（手首カメラを使わない回）
        try:
            remote.get(f"grasp_runs/{cap}/{pat}", tdir, check=False)
        except Exception:  # noqa: BLE001
            pass
    open(os.path.join(tdir, "traj.csv"), "w").write(traj)
    rows = list(csv.DictReader(io.StringIO(traj)))
    hold = [r for r in rows if r["phase"] == "hold"]
    g_hold = float(hold[-1]["gripper.pos"]) if hold else None
    completed = ("torque off" in log or "holding (torque on)" in log) and "Traceback" not in log and ("stopping" not in log or "placed aside; done" in log)
    held = g_hold is not None and g_hold <= HELD_MAX_DEG
    # re-evaluate with the execution evidence (verifier unchanged; lift_m is the measured outcome)
    obs = {"obj": np.load(os.path.join(res["info"]["scene_dir"], "object.npy")),
           "plane": None, "grasp": np.array(res["grasp"]), "score": res["score"],
           "ik": {"converged": True, "pos_err_m": 0.0, "rot_err_deg": 0.0},
           "lift_m": (LIFT_M if held else 0.0) if completed else 0.0}
    pr_before = res["predicates"]
    new = contract.evaluate(obs)
    for k2 in ("top_down",):  # keep the plane-based predicate computed at prepare time
        new["predicates"][k2] = pr_before[k2]
    new["predicates"]["lifted"]["note"] = f"gripper at hold = {g_hold} deg (held if <= {HELD_MAX_DEG}); run completed={completed}"
    app = [k2 for k2 in contract.PREDICATES if new["predicates"][k2]["status"] != "SKIPPED"]
    new["success"] = all(new["predicates"][k2]["status"] == "PASS" for k2 in app)
    new["cause"] = "success" if new["success"] else next(contract.CAUSE[k2] for k2 in contract.PREDICATES
                                                         if new["predicates"][k2]["status"] == "FAIL")
    if not completed:
        new["cause"] = "(h) execution aborted: " + (log.strip().splitlines() or ["no log"])[-1][:120]
    res.update(new, stage="executed", gripper_at_hold_deg=g_hold, run_completed=completed)
    loop.wjson(os.path.join(tdir, "result.json"), res)
    loop.event("trial", iter=a.iter, trial=a.trial, scene=cap, success=res["success"], cause=res["cause"],
               gripper_at_hold=g_hold)
    print(json.dumps({"success": res["success"], "cause": res["cause"], "gripper_at_hold_deg": g_hold,
                      "completed": completed, "log_tail": log.strip().splitlines()[-6:]}, indent=1))


def cmd_observe(a):
    """Record what the operator saw and use it as the lift measurement for an executed trial.

    2026-11-21: live.py measured "lifted" by the gripper angle at the hold (<= -8 deg = something between the jaws).
    That proxy fails for the bare aluminium bar: with a firm grip the fin-ray jaws close to about -1..-4 deg around the
    4.6 cm bar, so every lift the operator saw was recorded as a failure. The verifier (contract.py) is unchanged: it
    still needs lift_m >= LIFT_MIN_M; only the source of lift_m changes (operator: LIFT_M if lifted and held, else 0)."""
    d = iter_dir(a.iter)
    tdir = sorted(glob.glob(os.path.join(d, "trials", f"{int(a.trial):02d}_*")))[0]
    res = json.load(open(os.path.join(tdir, "result.json")))
    if res.get("stage") != "executed":
        raise SystemExit(f"[live] trial {a.trial} was not executed")
    lifted = a.lifted == "yes"
    if a.completed:  # the run log shows it finished (fixes trials the old 80-poll watcher marked "(h) aborted")
        res["run_completed"] = True
    obs = {"obj": np.load(os.path.join(res["info"]["scene_dir"], "object.npy")), "plane": None,
           "grasp": np.array(res["grasp"]), "score": res["score"],
           "ik": {"converged": True, "pos_err_m": 0.0, "rot_err_deg": 0.0},
           "lift_m": LIFT_M if (lifted and res.get("run_completed", True)) else 0.0}
    new = contract.evaluate(obs)
    new["predicates"]["top_down"] = res["predicates"]["top_down"]
    new["predicates"]["lifted"]["note"] = (f"operator: {'lifted' if lifted else 'not lifted'}"
                                           + (f" ({a.note})" if a.note else "") + f"; gripper at hold {res.get('gripper_at_hold_deg')} deg")
    app = [k2 for k2 in contract.PREDICATES if new["predicates"][k2]["status"] != "SKIPPED"]
    new["success"] = all(new["predicates"][k2]["status"] == "PASS" for k2 in app)
    new["cause"] = "success" if new["success"] else next(contract.CAUSE[k2] for k2 in contract.PREDICATES
                                                         if new["predicates"][k2]["status"] == "FAIL")
    res.update(new, lift_source="operator", operator_lifted=lifted, operator_note=a.note)
    loop.wjson(os.path.join(tdir, "result.json"), res)
    loop.event("operator_lift", iter=a.iter, trial=a.trial, lifted=lifted, note=a.note, success=res["success"])
    print(f"[live] {a.iter} trial {a.trial}: operator {'lifted' if lifted else 'not lifted'} -> success={res['success']} ({res['cause']})")


def cmd_teach(a):
    """Record one hand-taught pose (arm moved by hand, torque off): read q over CAN, append to teach.jsonl. No motion."""
    b = busy()
    if b:
        raise SystemExit("[live] Thor is busy:\n  " + "\n  ".join(b))
    reads = [ssh("~/projects/lerobot/.venv/bin/python - <<'EOF'\n" + READ_Q + "EOF", timeout=30).strip().splitlines()[-1]
             for _ in range(3)]
    Q = np.array([[float(v) for v in r.split(",")] for r in reads])
    spread = float(np.max(np.ptp(Q, axis=0)))
    path = os.path.join(REPO, "grasp", "runs", "teach", "teach.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = {"t": loop.now(), "q_deg": Q.mean(0).round(2).tolist(), "cap": a.cap, "spread_deg": round(spread, 2), "note": a.note}
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
    loop.event("teach", **row)
    print(json.dumps(row))


def cmd_batch(a):
    """Self-reset series: prepare -> execute -> verify, a.n times. Stops on 3 failed grasps in a row, 3 prepares in a
    row without a plan, an execution that did not complete, or the file grasp/runs/enpire/STOP."""
    import argparse as _ap
    if not a.i_have_operator_go:
        raise SystemExit("[batch] refused: needs --i-have-operator-go (one approval for the whole batch)")
    d = iter_dir(a.iter)
    stop_file = os.path.join(loop.ROOT, "STOP")
    if os.path.exists(stop_file):
        os.remove(stop_file)
    fails = noplan = done = ok = 0
    reason = "batch complete"
    while done < a.n:
        if os.path.exists(stop_file):
            reason = "STOP file"; break
        import time as _t
        for _ in range(30):  # the previous trial's process may still be shutting down on Thor
            if not busy():
                break
            _t.sleep(1)
        ready = None
        for _ in range(3):
            cmd_prepare(_ap.Namespace(iter=a.iter, snap_args=a.snap_args, place_offset=""))
            tdir = sorted(glob.glob(os.path.join(d, "trials", "*")))[-1]
            r = json.load(open(os.path.join(tdir, "result.json")))
            if r.get("ready_to_execute"):
                ready = os.path.basename(tdir)[:2]; break
        if ready is None:
            noplan += 1
            loop.event("batch_noplan", iter=a.iter, count=noplan)
            if noplan >= 3:
                reason = "no plan 3 times in a row"; break
            continue
        noplan = 0
        cmd_execute(_ap.Namespace(iter=a.iter, trial=ready, i_have_operator_go=True))
        tdir = sorted(glob.glob(os.path.join(d, "trials", f"{int(ready):02d}_*")))[0]
        r = json.load(open(os.path.join(tdir, "result.json")))
        done += 1
        ok += int(bool(r.get("success")))
        fails = 0 if r.get("success") else fails + 1
        loop.event("batch_progress", iter=a.iter, done=done, ok=ok, last=r.get("cause"))
        print(f"[batch] {done}/{a.n}: {'OK' if r.get('success') else 'FAIL'} ({r.get('cause')})  total {ok}/{done}  "
              f"gripper at hold {r.get('gripper_at_hold_deg')} deg  trial {ready}", flush=True)
        if not r.get("run_completed"):
            reason = "execution did not complete"; break
        if a.max_fails and fails >= a.max_fails:
            reason = f"{a.max_fails} failed grasps in a row"; break
    loop.event("batch_end", iter=a.iter, reason=reason, done=done, ok=ok)
    print(f"[batch] end: {reason}; {ok}/{done} lifted", flush=True)


def cmd_close(a):
    d = iter_dir(a.iter)
    meta = json.load(open(os.path.join(d, "iteration.json")))
    rs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "trials", "*", "result.json")))]
    # success rate over executed trials only; prepares that found no plan are re-tries of the same placement,
    # counted separately (plan_failures) instead of as failed grasps
    done = [r for r in rs if r.get("stage") == "executed"]
    n = len(done)
    sr = sum(r["success"] for r in done) / n if n else 0.0
    hist = {}
    for r in done:
        hist[r["cause"]] = hist.get(r["cause"], 0) + 1
    src = meta.get("source", "live")
    best = loop.best_record(src, T_BC_PATH)
    if not n:
        beats, decision = False, "invalid (no trials)"
    elif best is None:
        beats, decision = True, "keep (first baseline of this track)"
    else:
        beats = sr > best["success_rate"]
        decision = "keep (new best)" if beats else "revert (no improvement)"
    meta.update(status="done", finished=loop.now(), n_trials=n, success_rate=sr, failure_histogram=hist,
                passes_gate=sr >= contract.SUCCESS_GATE, decision=decision,
                not_executed=len(rs) - n, plan_failures=sum(1 for r in rs if not r.get("ready_to_execute")))
    loop.wjson(os.path.join(d, "iteration.json"), meta)
    if beats:
        loop.save_best({"iter": a.iter, "success_rate": sr, "source": src, "params": meta["params"],
                        "t_base_cam": T_BC_PATH, "verifier_version": contract.VERIFIER_VERSION})
    loop.event("iter_done", iter=a.iter, success_rate=round(sr, 3), decision=decision, n=n)
    print(decision, sr, hist)


def cmd_stir(a):
    """Loosen the pile (grasp/tools/plan_stir.py): capture -> stroke along the seam beside the target bar -> dry run.
    With --i-have-operator-go it runs on Thor (run_grasp.py --push, closed jaws) and captures the pile again after.
    Records go to <iter>/stirs/NN_<cap>/ (not a grasp trial: no verifier)."""
    d = iter_dir(a.iter)
    meta = json.load(open(os.path.join(d, "iteration.json")))
    b = busy()
    if b:
        raise SystemExit("[live] Thor is busy:\n  " + "\n  ".join(b))
    import yaml
    hy = yaml.safe_load(open(os.path.join(REPO, meta["params"]["home"])))
    q_home = ",".join(f"{float(hy[f'joint_{i}']):.1f}" for i in range(1, 8))
    sd = os.path.join(d, "stirs"); os.makedirs(sd, exist_ok=True)
    if a.planned:  # run the stroke the operator approved (planned and dry-run by an earlier `stir` call), unchanged
        tdir = sorted(glob.glob(os.path.join(sd, f"{int(a.planned):02d}_*")))[0]
        cap = os.path.basename(tdir).split("_", 1)[1]; cap_dir = os.path.join(REPO, "grasp", "runs", cap); k = int(a.planned)
        import shutil
        shutil.copy(os.path.join(tdir, "push.json"), cap_dir)
        if not a.i_have_operator_go:
            raise SystemExit("[live] --planned needs --i-have-operator-go")
        return _stir_exec(a, d, meta, sd, tdir, cap, cap_dir, k)
    src, out = _inproc("grasp.tools.thor_snap", ["--no-view", "--fast", *a.snap_args.split()])
    line = [l for l in out.splitlines() if " | " in l]
    if not line:
        raise SystemExit(f"[live] thor_snap failed:\n{out[-600:]}")
    cap = line[-1].split(" | ")[0].strip(); cap_dir = os.path.join(REPO, "grasp", "runs", cap)
    print("[live] " + line[-1])
    k = len(glob.glob(os.path.join(sd, "*")))
    tdir = os.path.join(sd, f"{k:02d}_{cap}"); os.makedirs(tdir)
    rc, out = _inproc("grasp.tools.plan_stir", [cap_dir, "--q", q_home, *a.stir_args.split()])
    print(out.strip())
    open(os.path.join(tdir, "plan_stir.log"), "w").write(out)
    if rc != 0 or not os.path.exists(os.path.join(cap_dir, "push.json")):
        loop.event("stir_planned", iter=a.iter, stir=k, scene=cap, ok=False); raise SystemExit("[live] no stir stroke")
    rc, out = _inproc("grasp.run_grasp", [cap_dir, "--push", "--home", os.path.join(REPO, meta["params"]["home"]), "--hold-home"])
    open(os.path.join(tdir, "dry_run.log"), "w").write(out)
    if "DRY RUN" not in out:
        loop.event("stir_planned", iter=a.iter, stir=k, scene=cap, ok=False); raise SystemExit("[live] dry run failed:\n" + out[-800:])
    import shutil
    shutil.copy(os.path.join(cap_dir, "push.json"), tdir); shutil.copy(os.path.join(cap_dir, "color.png"), os.path.join(tdir, "before.png"))
    loop.event("stir_planned", iter=a.iter, stir=k, scene=cap, ok=True)
    print(f"[live] stir {k} planned, dry run ok ({tdir})")
    if not a.i_have_operator_go:
        return
    return _stir_exec(a, d, meta, sd, tdir, cap, cap_dir, k)


def _stir_exec(a, d, meta, sd, tdir, cap, cap_dir, k):
    import shutil
    exe = deploy_executor("grasp/run_grasp.py")
    remote.put(os.path.join(REPO, meta["params"]["home"]), "grasp_work/grasp/config/")
    remote.put(os.path.join(cap_dir, "push.json"), f"grasp_runs/{cap}/")
    homearg = f" --home ~/grasp_work/grasp/config/{os.path.basename(meta['params']['home'])}"
    ex = meta["params"].get("exec_args", "").replace("--retries", "--_x").split(" --_x")[0]
    loop.event("stir_execute", iter=a.iter, stir=k, scene=cap)
    ssh("tmux kill-session -t enpire 2>/dev/null; "
        f"tmux new -d -s enpire 'cd ~/grasp_work && {remote.env_assign()}~/projects/lerobot/.venv/bin/python -u grasp/{exe} "
        f"~/grasp_runs/{cap} --go --push{homearg} {ex} 2>&1 | tee ~/grasp_runs/{cap}/run.log; sleep 600'")
    t_end = time.time() + 400
    log = ""
    while time.time() < t_end:
        time.sleep(3)
        log = ssh(f"grep -v -i warn ~/grasp_runs/{cap}/run.log 2>/dev/null || true", check=False)
        if "back at start" in log or "Traceback" in log:
            break
    open(os.path.join(tdir, "run.log"), "w").write(log)
    remote.get(f"grasp_runs/{cap}/traj.csv", tdir, check=False)
    print("\n".join(l for l in log.splitlines() if "[run]" in l))
    src, out = _inproc("grasp.tools.thor_snap", ["--no-view", "--fast", *a.snap_args.split()])
    line = [l for l in out.splitlines() if " | " in l]
    if line:
        after = line[-1].split(" | ")[0].strip()
        shutil.copy(os.path.join(REPO, "grasp", "runs", after, "color.png"), os.path.join(tdir, "after.png"))
        print("[live] after: " + line[-1])
    loop.event("stir_done", iter=a.iter, stir=k, scene=cap, completed="back at start" in log)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("new-iter"); s.add_argument("--hypothesis", required=True)
    s.add_argument("--budget", type=int, default=10); s.add_argument("--plan-args", default="")
    s.add_argument("--executor", default="grasp/run_grasp.py", help="repo-relative executor run on Thor")
    s.add_argument("--drop", nargs="?", const="0.05,0.45,0.42", default="",
                   help="after the lift carry the object beside the left shoulder and drop it there (x,y,z jaw-TCP point, base frame)")
    s.add_argument("--auto-place", action="store_true", help="self-reset: after the lift put the object down at a new spot (plan_place.py)")
    s.add_argument("--exec-args", default="", help="extra args for the executor on --go (e.g. speeds)")
    s.add_argument("--home", default="", help="repo-relative start-pose YAML: every trial starts and ends there (plan seeds from it)")
    s.add_argument("--decide", action="store_true", help="pile: decide what to take first (grasp/tools/decide.py: all objects, "
                   "image-language model, planner check); --plan-args are for aluminium")
    s.add_argument("--other-args", default="", help="with --decide: plan.py args for moving a non-aluminium obstacle")
    s = sub.add_parser("prepare"); s.add_argument("--iter", required=True)
    s.add_argument("--place-offset", default="", help="dx,dy [m] base frame: after the lift, put the object down there (clear an obstacle)")
    s.add_argument("--snap-args", default="", help="extra args for thor_snap.py (ROI etc.)")
    s = sub.add_parser("execute"); s.add_argument("--iter", required=True); s.add_argument("--trial", required=True)
    s.add_argument("--i-have-operator-go", action="store_true")
    s = sub.add_parser("close"); s.add_argument("--iter", required=True)
    s = sub.add_parser("observe"); s.add_argument("--iter", required=True); s.add_argument("--trial", required=True)
    s.add_argument("--lifted", choices=["yes", "no"], required=True); s.add_argument("--note", default="")
    s.add_argument("--completed", action="store_true", help="the run log ends with 'torque off'/'holding' (mark run_completed)")
    s = sub.add_parser("teach"); s.add_argument("--cap", required=True); s.add_argument("--note", default="")
    s = sub.add_parser("batch"); s.add_argument("--iter", required=True); s.add_argument("--n", type=int, default=20)
    s.add_argument("--snap-args", default="--hmax 0.10"); s.add_argument("--i-have-operator-go", action="store_true")
    s.add_argument("--max-fails", type=int, default=3, help="stop after this many verifier failures in a row; 0 = never "
                   "(2026-11-21: the gripper-angle lift check reads every bare-aluminium lift as a failure, so the operator judges by eye)")
    s = sub.add_parser("stir"); s.add_argument("--iter", required=True)
    s.add_argument("--snap-args", default="--perceive localhost:5558 --band 0.25,0.56,-0.10,0.35 --hmax 0.10")
    s.add_argument("--stir-args", default="--max-len 0.06 --min-len 0.06")
    s.add_argument("--i-have-operator-go", action="store_true")
    s.add_argument("--planned", default=None, help="run stir number N as planned earlier (no new capture / plan)")
    s = sub.add_parser("busy")
    a = ap.parse_args()
    if a.cmd == "busy":
        b = busy(); print("\n".join(b) if b else "idle"); return
    {"new-iter": cmd_new_iter, "prepare": cmd_prepare, "execute": cmd_execute, "close": cmd_close,
     "teach": cmd_teach, "batch": cmd_batch, "observe": cmd_observe, "stir": cmd_stir}[a.cmd](a)


if __name__ == "__main__":
    main()
