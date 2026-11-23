#!/usr/bin/env python3
"""Live localhost dashboard for the ENPIRE grasp loop (stdlib only, read-only on grasp/runs/enpire/).

    python3 grasp/enpire/dashboard.py            # -> http://localhost:8765
    python3 grasp/enpire/dashboard.py --port 8800

Shows: current best vs the 70% gate, success-rate per iteration, each iteration's hypothesis /
decision / failure histogram, per-trial predicate grid with overlay images, and the event tail.
The page polls /api/state every 2 s, so it updates while loop.py is running.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.path.dirname(HERE), "runs", "enpire")


def _j(p, default=None):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return default


def state():
    iters = []
    for d in sorted(glob.glob(os.path.join(ROOT, "iter_*"))):
        m = _j(os.path.join(d, "iteration.json"), {}) or {}
        trials = []
        for t in sorted(glob.glob(os.path.join(d, "trials", "*"))):
            r = _j(os.path.join(t, "result.json"), {}) or {}
            rel = os.path.relpath(t, ROOT)
            trials.append({"name": os.path.basename(t), "success": r.get("success"), "cause": r.get("cause"),
                           "level": r.get("level"), "score": r.get("score"), "seconds": r.get("seconds"),
                           "predicates": {k: v.get("status") for k, v in (r.get("predicates") or {}).items()},
                           "metrics": {k: v.get("metric") for k, v in (r.get("predicates") or {}).items()},
                           "img": f"/files/{rel}/overlay.jpg" if os.path.exists(os.path.join(t, "overlay.jpg")) else None,
                           "wrist": [f"/files/{rel}/{w}" for w in ("wrist_align2.jpg", "wrist_at_grasp.jpg", "wrist_hold.jpg",
                                                                   "wrist_near0.jpg") if os.path.exists(os.path.join(t, w))][:3],
                           "stage": r.get("stage"), "gripper_at_hold": r.get("gripper_at_hold_deg"),
                           "completed": r.get("run_completed"),
                           "log_tail": (open(os.path.join(t, "run.log")).read().strip().splitlines()[-2:]
                                        if os.path.exists(os.path.join(t, "run.log")) else [])})
        if m.get("status") == "running":  # live count while the iteration is still open
            ex = [t for t in trials if t["stage"] == "executed" or t["stage"] is None and t["success"] is not None]
            if ex:
                m["n_trials"] = len(ex)
                m["success_rate"] = sum(1 for t in ex if t["success"]) / len(ex)
                hist = {}
                for t in ex:
                    hist[t["cause"]] = hist.get(t["cause"], 0) + 1
                m["failure_histogram"] = hist
        m = {k: v for k, v in m.items() if k != "params"} | {"params_diff": None, "trials": trials}
        iters.append(m)
    ev = []
    try:
        with open(os.path.join(ROOT, "events.jsonl")) as f:
            ev = [json.loads(l) for l in f.readlines()[-40:]]
    except Exception:
        pass
    b = _j(os.path.join(ROOT, "best.json")) or {}
    tracks = b.get("tracks") or ({"legacy": b} if b else {})
    best = tracks.get(b.get("latest")) if b.get("latest") else (max(tracks.values(), key=lambda r: r["success_rate"]) if tracks else None)
    return {"best": best, "tracks": tracks, "iters": iters, "events": ev[::-1]}


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>ENPIRE grasp loop</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#f7f7f5;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e3e2dd;--ok:#1a7f37;--ng:#c62828;--sk:#9e9e9e;--acc:#2f5bd3}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9a9993;--card:#1f1f1d;--line:#33322f;--ok:#4cc26a;--ng:#ef6b6b;--sk:#777;--acc:#7d9bf0}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
main{max-width:1200px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:0 0 4px} .mut{color:var(--mut)}
.row{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;min-width:150px}
.kpi b{display:block;font-size:26px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin:12px 0}
table{border-collapse:collapse;width:100%} td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
.P{color:var(--ok);font-weight:600}.F{color:var(--ng);font-weight:600}.S{color:var(--sk)}
.bar{height:10px;background:var(--line);border-radius:5px;position:relative;min-width:120px}
.bar i{display:block;height:100%;border-radius:5px;background:var(--acc)}
.bar s{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--ng)}
.trials{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:10px}
.trial{border:1px solid var(--line);border-radius:8px;padding:6px}
.trial img{width:100%;border-radius:4px;cursor:zoom-in}
.ev{font:12px ui-monospace,monospace;max-height:260px;overflow:auto;white-space:pre-wrap}
details summary{cursor:pointer}
svg text{fill:var(--mut);font-size:11px}
</style></head><body><main>
<h1>ENPIRE grasp loop — OpenArm left arm × chest D435i × GraspGenX</h1>
<div class="mut" id="sub">loading…</div>
<div class="row" id="kpis"></div>
<div class="card"><b>Success rate per iteration</b> <span class="mut">(red line = 70% gate)</span><div id="chart"></div></div>
<div class="card"><b>Iterations</b><table id="iters"></table></div>
<div class="card"><b>Trials of</b> <select id="pick"></select><div class="trials" id="trials"></div></div>
<div class="card"><b>Events</b><div class="ev" id="ev"></div></div>
</main><script>
const PR=["object_found","grasp_found","grasp_on_object","top_down","width_ok","reachable","lifted"];
let S=null, chosen=null;
const pct=x=>x==null?"–":(100*x).toFixed(0)+"%";
const esc=s=>String(s??"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
function chart(it){const W=Math.max(320,document.getElementById("chart").clientWidth||600),H=150,p=26;
 const d=it.filter(x=>x.success_rate!=null);if(!d.length)return '<div class="mut">no finished iteration yet</div>';
 const x=i=>p+(W-2*p)*(d.length==1?0.5:i/(d.length-1)),y=v=>H-p-(H-2*p)*v;
 let s=`<svg width="${W}" height="${H}"><line x1="${p}" x2="${W-p}" y1="${y(.7)}" y2="${y(.7)}" stroke="var(--ng)" stroke-dasharray="4 3"/>`;
 s+=`<polyline fill="none" stroke="var(--acc)" stroke-width="2" points="${d.map((v,i)=>x(i)+','+y(v.success_rate)).join(' ')}"/>`;
 d.forEach((v,i)=>{s+=`<circle cx="${x(i)}" cy="${y(v.success_rate)}" r="4" fill="${v.decision&&v.decision.startsWith('keep')?'var(--ok)':'var(--sk)'}"><title>${esc(v.iter)} ${pct(v.success_rate)} ${esc(v.hypothesis)}</title></circle><text x="${x(i)}" y="${H-6}" text-anchor="middle">${v.iter.slice(5)}</text>`});
 return s+`<text x="2" y="${y(1)+4}">100</text><text x="2" y="${y(0)+4}">0</text></svg>`}
function render(){const it=S.iters,b=S.best,last=it[it.length-1]||{};
 document.getElementById("sub").textContent=`${it.length} iterations · verifier ${last.verifier_version||"–"} · updated ${new Date().toLocaleTimeString()}`;
 document.getElementById("kpis").innerHTML=[["Best",b?pct(b.success_rate):"–",b?b.iter:""],["Gate","70%",b&&b.success_rate>=.7?"passed":"not yet"],
  ["Latest",pct(last.success_rate),(last.status||"")+" "+(last.iter||"")],["Level",(last.trials&&last.trials[0]&&last.trials[0].level)||"–","perception→plan→real"]]
  .map(([k,v,n])=>`<div class="kpi"><span class="mut">${k}</span><b>${v}</b><span class="mut">${esc(n)}</span></div>`).join("");
 document.getElementById("chart").innerHTML=chart(it);
 document.getElementById("iters").innerHTML="<tr><th>iter</th><th>hypothesis</th><th>source</th><th>success</th><th>decision</th><th>failures</th></tr>"+
  it.slice().reverse().map(m=>`<tr><td>${m.iter}</td><td>${esc(m.hypothesis)}</td><td class="mut">${esc(m.source)}</td>
  <td><div class="bar"><i style="width:${100*(m.success_rate||0)}%"></i><s style="left:70%"></s></div>${pct(m.success_rate)} (${m.n_trials??"…"})</td>
  <td>${esc(m.status=="done"?m.decision:m.status+(m.reason?": "+m.reason:""))}</td>
  <td class="mut">${esc(Object.entries(m.failure_histogram||{}).filter(([k])=>k!="success").map(([k,v])=>k+" ×"+v).join("; "))}</td></tr>`).join("");
 const sel=document.getElementById("pick"),names=it.map(m=>m.iter);
 if(sel.options.length!=names.length){sel.innerHTML=names.slice().reverse().map(n=>`<option>${n}</option>`).join("");}
 if(!chosen||!names.includes(chosen))chosen=names[names.length-1];sel.value=chosen;
 const m=it.find(x=>x.iter==chosen)||{trials:[]};
 document.getElementById("trials").innerHTML=m.trials.map(t=>`<div class="trial">${t.img?`<a href="${t.img}" target="_blank"><img src="${t.img}?${Date.now()}"></a>`:""}
  <div><b class="${t.success?"P":"F"}">${t.success?"SUCCESS":"FAIL"}</b> <span class="mut">${esc(t.name)} · ${t.seconds??"–"}s · score ${t.score!=null?t.score.toFixed(2):"–"}</span></div>
  <div class="mut">${esc(t.cause)}</div>
  ${t.stage?`<div class="mut">${esc(t.stage)}${t.gripper_at_hold!=null?` · gripper at hold ${t.gripper_at_hold.toFixed(1)}°`:""}${t.completed!=null?` · run ${t.completed?"completed":"aborted"}`:""}</div>`:""}
  ${(t.wrist||[]).length?`<div style="display:flex;gap:4px;margin:4px 0">${t.wrist.map(w=>`<a href="${w}" target="_blank" style="flex:1"><img src="${w}?${Date.now()}" title="${esc(w.split('/').pop())}"></a>`).join("")}</div>`:""}
  ${(t.log_tail||[]).length?`<div class="mut" style="font:11px ui-monospace,monospace">${t.log_tail.map(esc).join("<br>")}</div>`:""}<div>${PR.map(k=>{const s=(t.predicates||{})[k]||"S",mt=(t.metrics||{})[k];return `<span class="${s[0]}" title="${k} ${mt!=null?mt.toFixed(3):""}">${k.replace(/_/g," ")}:${s=="PASS"?"✓":s=="FAIL"?"✗":"–"}</span>`}).join(" ")}</div></div>`).join("")||'<div class="mut">no trials</div>';
 document.getElementById("ev").textContent=S.events.map(e=>`${e.t} ${e.kind} ${Object.entries(e).filter(([k])=>!["t","kind","params"].includes(k)).map(([k,v])=>k+"="+(typeof v=="object"?JSON.stringify(v):v)).join(" ")}`).join("\n");}
document.getElementById("pick").onchange=e=>{chosen=e.target.value;render()};
async function tick(){try{S=await (await fetch("/api/state")).json();render()}catch(e){document.getElementById("sub").textContent="dashboard server unreachable"}}
tick();setInterval(tick,2000);
</script></body></html>"""


def _clean(o):
    """NaN/inf -> null: browsers' JSON.parse rejects NaN (live results store NaN for 'not computed')."""
    import math
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if path == "/api/state":
            return self._send(200, json.dumps(_clean(state()), default=float, allow_nan=False).encode(), "application/json")
        if path.startswith("/files/"):
            f = os.path.realpath(os.path.join(ROOT, unquote(path[len("/files/"):])))
            if f.startswith(os.path.realpath(ROOT) + os.sep) and os.path.isfile(f):
                ctype = "image/jpeg" if f.endswith(".jpg") else "application/json"
                return self._send(200, open(f, "rb").read(), ctype)
        self._send(404, b"not found", "text/plain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    os.makedirs(ROOT, exist_ok=True)
    print(f"[dashboard] http://localhost:{a.port}  (reading {ROOT})", flush=True)
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
