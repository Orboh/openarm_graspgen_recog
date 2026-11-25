"""拾う物 1 つの切り抜き画像から材質を SigLIP2 のゼロショットで当てる（thor_snap.py --yoloe の pick_target が使う）。

モデルは最初に classify_material を呼んだときに 1 度だけ読み込む（import しただけでは GPU を使わない）。
  GRASP_SIGLIP_MODEL: モデル名（既定 google/siglip2-so400m-patch14-384）
  GRASP_RECOG_DEVICE: cuda / cpu（既定 cuda。GPU が空いていなければ cpu）
  GRASP_SIGLIP_DEVICE: SigLIP2 だけの置き場（GRASP_RECOG_DEVICE より先に見る。8 GB の GPU を共用するときは cpu）
"""
from __future__ import annotations

import os
import time

import numpy as np

candidates = ["metal", "wood", "composite"]
# 材質ごとに言い方を何通りか並べ、その平均の点で比べる（2026-11-23 Shreyan: 1 語や「made of …」の 1 文だと、細いアルミのバーを
# composite 0.84 と取り違えた。下の言い方の平均では metal 0.87）。ラベルを足すときは、ここと ROUTE_OFFSET の両方に足す
PROMPTS = {
    "metal": ["a photo of a metal object", "a shiny aluminium bar", "a piece of metal", "a steel or aluminium part"],
    "wood": ["a photo of a wooden object", "a piece of wood", "a wooden block or plank", "a wooden log or cylinder"],
    # 2026-11-23: 発泡スチロールの白い塊を metal 0.57 と取り違えた（言い方に泡の物が無かった）。泡の言い方を足して composite 0.92
    "composite": ["a photo of a plastic object", "a piece of styrofoam", "a white foam block",
                  "a piece of plastic or composite material", "polystyrene foam packaging"],
}

# grasp/tools/plan_place.py --by-material: where to carry each material, as a --to dx,dy offset [m] from the
# object's own current spot (base frame; right is negative y, per plan_place.py's own convention). Anything
# classify_material returns that isn't a key here (shouldn't happen with the 3-class list above, but a model
# swap or a mis-classification could) falls back to "composite"'s offset.
ROUTE_OFFSET = {
    # 2026-11-23 Shreyan: 北九州の左腕で届く量に縮めた（白い板で試した: 前 15 cm・左 20 cm・右 15 cm まで解けた。
    # 20〜25 cm は解けなかった）。画面の端には出るが、外までは出ない。前は 2026-11-23 の実機で 1 回置けた
    "metal": (0.0, -0.15),       # right（画面の右）
    "wood": (0.0, 0.20),         # left（画面の左）
    "composite": (0.15, 0.0),    # forward（画面の上）
}
model_id = os.environ.get("GRASP_SIGLIP_MODEL", "google/siglip2-so400m-patch14-384")
_MODEL = None


def _load():
    """SigLIP2 を 1 度だけ読み込み、(processor, 画像の側のモデル, 言い方の特徴, logit_scale, logit_bias, 置き場) を返す。

    2026-11-23 Shreyan（速くする）: 言い方は変わらないので、文字の側は読み込みのときに 1 度だけ CPU で特徴にして捨てる
    （毎回 10 文を通すと CPU で約 5 秒）。画像の側（約 0.8 GB・fp16）だけを GPU に載せる。GPU が足りなければ CPU に戻る。
    """
    global _MODEL
    if _MODEL is None:
        _t0 = time.time()  # 段ごとの時間（記録だけ・2026-11-25 Shreyan）
        import torch
        from transformers import AutoModel, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id, dtype=torch.float32).eval()
        texts = [t for c in candidates for t in PROMPTS[c]]
        tin = processor(text=texts, return_tensors="pt", padding="max_length", max_length=64)
        with torch.no_grad():
            tf = model.get_text_features(**tin)
            tf = getattr(tf, "pooler_output", tf)
            scale, bias = float(model.logit_scale.exp()), float(model.logit_bias)
        tf = torch.nn.functional.normalize(tf.float(), dim=-1)
        vision = model.vision_model
        del model
        want = os.environ.get("GRASP_SIGLIP_DEVICE") or os.environ.get("GRASP_RECOG_DEVICE", "cuda")
        device = "cpu"
        if want.startswith("cuda") and torch.cuda.is_available():
            try:
                vision = vision.to(want, torch.float16)
                device = want
            except torch.OutOfMemoryError:
                vision = vision.to("cpu", torch.float32)
                torch.cuda.empty_cache()
                print("[classify] GPU に載らなかったので SigLIP2 は CPU で動かす", flush=True)
        _MODEL = (processor, vision, tf, scale, bias, device)
        from grasp import timing
        timing.event("siglip/load (processor+model+text features+to device)", _t0, time.time(), device=device)
    return _MODEL


def classify_material(crop_bgr: np.ndarray) -> tuple[str, float]:
    """
    Classify the material of an object given its cropped image in BGR format.

    Args:
        crop_bgr (np.ndarray): The cropped image of the object in BGR format.

    Returns:
        str: The predicted material label.
        float: The confidence score of the prediction.
    """
    import torch
    _t0 = time.time()  # 段ごとの時間（記録だけ・2026-11-25 Shreyan）
    processor, vision, tf, scale, bias, device = _load()
    _t1 = time.time()
    crop_rgb = crop_bgr[..., ::-1].copy()  # BGR → RGB
    pix = processor(images=crop_rgb, return_tensors="pt")["pixel_values"]
    dtype = torch.float16 if device != "cpu" else torch.float32
    with torch.no_grad():
        im = vision(pixel_values=pix.to(device, dtype)).pooler_output.float().cpu()
    im = torch.nn.functional.normalize(im, dim=-1)
    logits = (im @ tf.T)[0] * scale + bias  # SiglipModel の logits_per_image と同じ
    owner = np.array([i for i, c in enumerate(candidates) for _ in PROMPTS[c]])
    # 言い方ごとの logit を材質ごとに平均し、3 つの間の softmax にする
    per_class = torch.stack([logits[torch.from_numpy(owner == i)].mean() for i in range(len(candidates))])
    probs = per_class.softmax(0)
    pred_idx = int(probs.argmax())
    from grasp import timing
    timing.event("siglip/get model (load if first)", _t0, _t1)
    timing.event("siglip/classify", _t1, time.time(), device=device, label=candidates[pred_idx])
    return candidates[pred_idx], float(probs[pred_idx])
