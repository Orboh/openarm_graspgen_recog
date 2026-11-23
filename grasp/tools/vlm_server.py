#!/usr/bin/env python3
"""Image-language model server for "what to take first" (runs on Thor). **Does not move the arm.**

2026-11-21 (Sota: 画像言語モデルに何から取るかを最初から相談させる): Qwen3-VL-8B-Instruct (Apache-2.0, commercial use
OK) answers questions about a picture of the pile. grasp/tools/decide.py sends the chest image with numbered objects
and a list of what was found, and asks for the order to take things (JSON). The answer is only advice: every action
it proposes is checked by the real planner (plan.py: grasps, floor, IK, cuRobo) before anything moves.

    ~/work/perceive/.venv/bin/python grasp/tools/vlm_server.py --port 5560
    ~/work/perceive/.venv/bin/python grasp/tools/vlm_server.py --once img.jpg --prompt "What is in this picture?"

Wire (REQ/REP, msgpack + msgpack-numpy): {"bgr": HxWx3 uint8, "prompt": str, "max_new_tokens": int}
  -> {"text": str, "seconds": float} or {"error": str}
"""
from __future__ import annotations

import argparse
import time

import numpy as np

MODEL = "Qwen/Qwen3-VL-4B-Instruct"  # 2026-11-22: 8B -> 4B (Sota: もっと小さいモデル). 2B gave nonsense orders


class VLM:
    def __init__(self, model=MODEL):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.torch = torch
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model, dtype=torch.bfloat16, device_map="cuda").eval()
        self.proc = AutoProcessor.from_pretrained(model)

    def __call__(self, bgr, prompt, max_new_tokens=700):
        from PIL import Image
        img = Image.fromarray(np.ascontiguousarray(bgr[..., ::-1]))
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        inp = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                            return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5560)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--once", default=None, help="offline: one image file")
    ap.add_argument("--prompt", default="この画像に何が写っていますか？")
    a = ap.parse_args()
    t0 = time.time()
    V = VLM(a.model)
    print(f"[vlm] {a.model} loaded in {time.time() - t0:.0f} s", flush=True)
    if a.once:
        import cv2
        t0 = time.time(); txt = V(cv2.imread(a.once), a.prompt)
        print(f"[vlm] {time.time() - t0:.1f} s\n{txt}")
        return 0
    import msgpack
    import msgpack_numpy as mn
    import zmq
    sock = zmq.Context().socket(zmq.REP); sock.bind(f"tcp://*:{a.port}")
    print(f"[vlm] listening on :{a.port}", flush=True)
    while True:
        req = msgpack.unpackb(sock.recv(), object_hook=mn.decode, raw=False)
        try:
            t0 = time.time()
            rep = {"text": V(np.asarray(req["bgr"]), req["prompt"], int(req.get("max_new_tokens", 700))), "seconds": time.time() - t0}
        except Exception as e:  # noqa: BLE001  never die on one bad request
            rep = {"error": f"{type(e).__name__}: {e}"}
        sock.send(msgpack.packb(rep, default=mn.encode, use_bin_type=True))


if __name__ == "__main__":
    raise SystemExit(main())
