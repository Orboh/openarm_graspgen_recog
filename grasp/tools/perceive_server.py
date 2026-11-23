#!/usr/bin/env python3
"""Find aluminium frames in a cluttered pile (chest RGB). **Does not move the arm.** Commercial-use models only.

2026-11-21 (Sota): the next goal is to pick only the aluminium frames out of a combustible-waste pile (Shinshu Takeei,
11/25). Pipeline per image:
  1. Grounding DINO (IDEA-Research/grounding-dino-base, Apache-2.0) proposes boxes for text prompts.
  2. SAM 2.1 (facebook/sam2.1-hiera-small, Apache-2.0) cuts a mask inside each box.
  3. Each mask is scored as "aluminium frame": long and straight (minAreaRect aspect, rectangularity), bright and
     unsaturated (anodised / bare aluminium), not huge. Site images 11/17 showed silver plastic bags, white boards and
     the robot's own silver arm also get boxed by step 1, so shape is the main filter, colour only a tie-breaker.
     Optional: DINOv2 (facebook/dinov2-small, Apache-2.0) k-NN against example crops in --refs (pos/ and neg/ folders).
Runs as a resident ZMQ server (models load once, ~0.5 s per image) in its own venv (~/work/perceive/.venv):

    ~/work/perceive/.venv/bin/python grasp/tools/perceive_server.py --port 5558 [--refs grasp/config/alu_refs]
    ~/work/perceive/.venv/bin/python grasp/tools/perceive_server.py --once <image|cap_dir>... -o out/   # offline check

Wire (REQ/REP, msgpack + msgpack-numpy): request {"bgr": HxWx3 uint8}; reply {"cands": [{"box", "mask" (HxW bool),
"score", "label", "aspect", "rect", "bright", "alu"}...]} sorted by "alu" (0..1) descending, or {"error": str}.
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import cv2
import numpy as np

PROMPTS = ["aluminum frame", "metal bar", "aluminum extrusion", "silver metal rod", "white metal bar", "long thin bar"]


class Perceiver:
    def __init__(self, dev="cuda", refs=None, box_thr=0.20):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.torch, self.dev, self.box_thr = torch, dev, box_thr
        self.gp = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        self.gm = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(dev).eval()
        self.sam = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-small", device=dev)
        self.knn = None
        if refs and os.path.isdir(refs):
            from transformers import AutoImageProcessor, AutoModel
            self.dp = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
            self.dm = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
            feats, labs = [], []
            for lab, sub in ((1, "pos"), (0, "neg")):
                for f in sorted(glob.glob(os.path.join(refs, sub, "*"))):
                    feats.append(self._embed(cv2.imread(f))); labs.append(lab)
            if feats:
                self.knn = (np.stack(feats), np.array(labs))
                print(f"[perceive] DINOv2 refs: {int(np.sum(self.knn[1]))} aluminium, {int(np.sum(self.knn[1] == 0))} other", flush=True)

    def _embed(self, bgr):
        from PIL import Image
        with self.torch.no_grad():
            x = self.dp(images=Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), return_tensors="pt").to(self.dev)
            f = self.dm(**x).last_hidden_state[:, 0].float().cpu().numpy()[0]
        return f / (np.linalg.norm(f) + 1e-9)

    def __call__(self, bgr, prompts=None):
        from PIL import Image
        H, W = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); im = Image.fromarray(rgb)
        with self.torch.no_grad():
            inp = self.gp(images=im, text=". ".join(prompts or PROMPTS) + ".", return_tensors="pt").to(self.dev)
            out = self.gm(**inp)
            r = self.gp.post_process_grounded_object_detection(out, inp.input_ids, threshold=self.box_thr,
                                                               text_threshold=self.box_thr, target_sizes=[(H, W)])[0]
        boxes = r["boxes"].cpu().numpy(); scores = r["scores"].cpu().numpy()
        labels = list(r.get("text_labels", r.get("labels", [""] * len(boxes))))
        keep = [i for i, b in enumerate(boxes) if (b[2] - b[0]) * (b[3] - b[1]) < 0.35 * W * H]  # drop "everything" boxes
        boxes, scores, labels = boxes[keep], scores[keep], [labels[i] for i in keep]
        cands = []
        if len(boxes):
            with self.torch.no_grad(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
                self.sam.set_image(rgb)
                masks, mscore, _ = self.sam.predict(box=boxes, multimask_output=False)
            masks = np.asarray(masks).reshape(len(boxes), H, W) > 0.5
            # gray-world white balance before the colour gate: on 11/21 the D435 auto white balance drifted between
            # captures of the same pile (bar saturation 0.15 -> 0.30, aluminium score 0.95 -> 0.38)
            f = bgr.reshape(-1, 3).astype(np.float32).mean(0)
            wb = np.clip(bgr.astype(np.float32) * (f.mean() / np.maximum(f, 1.0)), 0, 255).astype(np.uint8)
            hsv = cv2.cvtColor(wb, cv2.COLOR_BGR2HSV)
            for b, s, l, m in zip(boxes, scores, labels, masks):
                area = int(m.sum())
                if area < 150:
                    continue
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                c = max(cnts, key=cv2.contourArea)
                (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
                long_, short = max(rw, rh), max(1.0, min(rw, rh))
                aspect = long_ / short
                rect = area / max(1.0, rw * rh)               # 1 = perfect rectangle (bars); bags are blobby
                bright = float(np.mean(hsv[..., 2][m])) / 255.0
                sat = float(np.mean(hsv[..., 1][m])) / 255.0
                # rule score: elongated rigid rectangle that is bright and unsaturated
                # shape: long and straight (bars), colour: a gate (aluminium is bright and grey; the red hose on 11/17 was
                # long and straight too), size: not the robot's own arm / a whole bag
                f_aspect = np.clip((aspect - 2.0) / 3.0, 0, 1)
                f_rect = np.clip((rect - 0.35) / 0.35, 0, 1)
                g_col = np.clip((bright - 0.40) / 0.25, 0, 1) * np.clip((0.32 - sat) / 0.12, 0, 1)
                f_size = 1.0 if area < 0.08 * W * H else 0.3
                alu = float((0.65 * f_aspect + 0.35 * f_rect) * (0.3 + 0.7 * g_col)) * f_size
                if self.knn is not None:  # appearance check on the box crop
                    x0, y0, x1, y1 = b.astype(int)
                    crop = bgr[max(0, y0):y1, max(0, x0):x1]
                    if crop.size:
                        f = self._embed(crop); sim = self.knn[0] @ f
                        top = np.argsort(-sim)[:5]
                        p = float(np.mean(self.knn[1][top]))
                        alu = 0.5 * alu + 0.5 * p
                cands.append({"box": b.tolist(), "mask": m, "score": float(s), "label": str(l), "aspect": float(aspect),
                              "rect": float(rect), "bright": bright, "sat": sat, "alu": alu, "centre": [float(cx), float(cy)],
                              "angle": float(ang), "length_px": float(long_)})
        # overlapping boxes from different prompts: keep the best per mask (IoU > 0.6)
        cands.sort(key=lambda c: -c["alu"])
        uniq = []
        for c in cands:
            if all((c["mask"] & u["mask"]).sum() / max(1, (c["mask"] | u["mask"]).sum()) < 0.6 for u in uniq):
                uniq.append(c)
        return uniq


def draw(bgr, cands, alu_thr):
    vis = bgr.copy()
    for i, c in enumerate(cands):
        col = (0, 0, 255) if c["alu"] >= alu_thr else (180, 180, 180)
        ov = vis.copy(); ov[c["mask"]] = col; vis = cv2.addWeighted(ov, 0.4, vis, 0.6, 0)
        x0, y0, x1, y1 = (int(v) for v in c["box"])
        cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
        cv2.putText(vis, f"#{i} alu {c['alu']:.2f} ar {c['aspect']:.1f} rc {c['rect']:.2f}", (x0, max(12, y0 - 4)), 0, 0.42, col, 1)
    return vis


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5558)
    ap.add_argument("--refs", default=None, help="folder with pos/ and neg/ example crops for the DINOv2 k-NN check")
    ap.add_argument("--box-thr", type=float, default=0.20)
    ap.add_argument("--alu-thr", type=float, default=0.55)
    ap.add_argument("--once", nargs="*", default=None, help="offline: images (.jpg/.png/.npy) or capture folders")
    ap.add_argument("-o", "--out", default="perceive_out")
    a = ap.parse_args()
    P = Perceiver(refs=a.refs, box_thr=a.box_thr)
    if a.once is not None:
        os.makedirs(a.out, exist_ok=True)
        for p in a.once:
            if os.path.isdir(p):
                bgr = cv2.imread(os.path.join(p, "color.png"))
            elif p.endswith(".npy"):
                bgr = np.load(p)  # rollout frames are stored BGR
            else:
                bgr = cv2.imread(p)
            t0 = time.time(); cands = P(bgr); dt = time.time() - t0
            name = os.path.basename(os.path.normpath(p)).rsplit(".", 1)[0]
            cv2.imwrite(os.path.join(a.out, name + ".jpg"), draw(bgr, cands, a.alu_thr))
            good = [c for c in cands if c["alu"] >= a.alu_thr]
            print(f"{name}: {len(cands)} candidates, {len(good)} aluminium ({dt:.2f}s) "
                  + " ".join(f"[alu {c['alu']:.2f} ar {c['aspect']:.1f} rc {c['rect']:.2f} br {c['bright']:.2f} sat {c['sat']:.2f}]" for c in cands[:4]), flush=True)
        return 0
    import msgpack
    import msgpack_numpy as mn
    import zmq
    sock = zmq.Context().socket(zmq.REP); sock.bind(f"tcp://*:{a.port}")
    print(f"[perceive] listening on :{a.port}", flush=True)
    while True:
        req = msgpack.unpackb(sock.recv(), object_hook=mn.decode, raw=False)
        try:
            t0 = time.time(); cands = P(np.asarray(req["bgr"]), req.get("prompts"))
            rep = {"cands": cands, "seconds": time.time() - t0}
        except Exception as e:  # never die on one bad request
            rep = {"error": f"{type(e).__name__}: {e}"}
        sock.send(msgpack.packb(rep, default=mn.encode, use_bin_type=True))


if __name__ == "__main__":
    raise SystemExit(main())
