"""手首の D405 で直しながら掴む姿勢へ近づく（run_grasp.py --wrist-servo が呼ぶ）。**腕が動く。**

2026-11-23・作業者: Munenori。計算は grasp/wrist_servo.py、カメラは grasp/wrist.py の WristCam。ここは段の流れだけで、
腕の動かし方（go）・区間の確かめ（check）・IK（solve）・順運動学（fk）は run_grasp.py のものを受け取る（同じ確かめを通す）。

段:
  1. 掴む姿勢を近づく向きに near だけ戻した姿勢（近い所）へ（pregrasp から）
  2. 測る → 横（閉じる・厚み）を 0.8 倍に弱めて直す。最大 iters 回、直す量の合計は max_total まで
  3. 一番太い所が爪の先から depth_target 奥に来るまで（机で止まる）、step_m ごとに下ろし、そのたびに測って横を小さく直す
  4. 下ろし終えた姿勢を返す（run_grasp.py がそこで閉じる）
止めるとき（物を見失う・直す量が大きすぎる・確かめを通らない）は ServoAbort を投げる。run_grasp.py が pregrasp に戻してから
いつもの「途中で止めた」戻り方で戻る。左腕だけ（爪の向きの写し AX_L7_G は左の link7 の向き）。
"""
from __future__ import annotations

import json
import os

import numpy as np

from grasp.wrist_servo import descend_travel, find_target, lateral_step, offsets, t_covered

AX_L7_G = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)  # 列: 爪の座標の Xg・Yg・Zg を link7 の座標で（plan.py の T_L7_G）
FLANGE_Z = 0.0895   # link7 の原点から手首の先の面
TIP_FROM_FLANGE = 0.14


class ServoAbort(Exception):
    """手首のカメラで直すのをやめる（腕は止まった所。呼んだ側が戻す）。"""


def tips_and_axes(fk, qd) -> tuple[np.ndarray, np.ndarray]:
    """姿勢 qd（lerobot の度）での爪の先の中点（腕の付け根の座標）と、爪の座標の向き（列が Xg・Yg・Zg）。"""
    T = fk(qd)
    Rg = T[:3, :3] @ AX_L7_G
    tips = T[:3, 3] + T[:3, :3] @ [0, 0, FLANGE_Z] + Rg[:, 2] * TIP_FROM_FLANGE
    return tips, Rg


def look_back(fk, grasp_q, obj_top_z: float, near_min: float = 0.04, clearance: float = 0.025, near_max: float = 0.12) -> float:
    """最初に測る所: 掴む姿勢から近づく向きに戻す量 [m]。爪の先が物の上の面（胸のカメラ）より clearance 上に来るまで戻す。

    2026-11-23 北九州の 1 回目: 挟む深さ 52 mm の計画で 4 cm だけ戻した所へ下りる間に、爪の先が木材の上の面より下に入り、
    腕の誤差で爪が木材に当たって押した（測る前に当たった）。東京の v2 も 8 cm 戻した所で見ていた。"""
    tips, Rg = tips_and_axes(fk, grasp_q)
    up_per_m = max(-Rg[2, 2], 0.3)  # 近づく向きに 1 m 戻すと爪の先が上がる量
    return float(np.clip((obj_top_z + clearance - tips[2]) / up_per_m, near_min, near_max))


def fits_between_jaws(points_base, max_w: float = 0.06) -> bool:
    """物が丸ごと爪の間に入る太さか（机に沿った短い辺が max_w 以下）。手首のカメラの合わせ方は物全体を爪の真ん中へ寄せるので、
    縁や壁を掴む太い物（テープの輪・上の開いた箱）では使わない（2026-11-23）。"""
    from grasp.objects import footprint
    return footprint(points_base, up=[0.0, 0.0, 1.0])[1] <= max_w


def servo_approach(*, grasp_q, pre_q, fk, solve, go, check, cam, jaw, obj_base: np.ndarray, obj_extent: float,
                   g_open: float, out_dir: str, near: float = 0.04, depth_target: float = 0.030, speed: float = 4.0,
                   iters: int = 4, max_total: float = 0.05, step_m: float = 0.02, max_extra: float = 0.04,
                   t_shift: float = 0.0, one_shot: bool = False, one_shot_c_max: float = 0.012, log=print) -> list:
    """手首のカメラで直しながら近づき、閉じる直前の姿勢（lerobot の度）を返す。

    fk(qd) → link7 の 4x4、solve(T4x4, seed_qd) → qd か None、go(q, speed, g, phase)、check(q0, q1, name, allow_low)。
    obj_base: 胸のカメラで見た物の中心（腕の付け根の座標）。obj_extent: 物の水平の大きさ [m]（狙いの塊を絞る半径に使う）。
    max_extra: 計画の掴む姿勢より奥へ下ろしてよい量（計画が浅すぎたときの分）。
    t_shift: 物に対して手を、爪の厚みの向き（爪の座標の +Yg）へ寄せる量 [m]。カメラは爪に固定なので、手首のカメラから見た爪の模型の
    厚みの向きのずれは、爪の座標で毎回同じに出る。2026-11-23 Munenori: 毎回、爪の奥側の長辺で木材を挟んでいた（閉じた 3 回とも奥は −Yg）。
    one_shot: 近い所で 1 回だけ測り、横の直し（測った差を全部）と下ろす動きを 1 回の動きにまとめる。閉じる向きの差が one_shot_c_max を
    超える・物が画像の端で切れているときは、今までどおり何度も直してから下ろす（斜めに下ろす間に、開いた爪（物とのすき間 片側約 15 mm）が
    物に当たらない範囲だけ）。2026-11-23: 直した量への応答はほぼ 1 倍（閉じる 0.86・厚み 1.13）で、測りの揺れが 4〜5 mm あり、
    5 mm 以内まで直すくり返しは揺れを追って 1〜3 回増えていた（1 回 1.1〜3.3 秒）。"""
    records = []
    bias_t = -t_shift  # 厚みの向きの目標（物が爪の真ん中からこれだけずれて見える所で止める）。手を +Yg へ寄せると物は −Yg に見える

    def shifted(q_ref, delta_g, seed):
        """q_ref の link7 を、その姿勢の爪の座標で delta_g [m] だけ平行に動かした姿勢（向きは同じ）。"""
        T = fk(q_ref).copy()
        Rg = T[:3, :3] @ AX_L7_G
        T[:3, 3] = T[:3, 3] + Rg @ np.asarray(delta_g, float)
        q = solve(T, seed)
        if q is None:
            raise ServoAbort(f"爪の座標で {np.round(np.asarray(delta_g) * 1000).tolist()} mm 動かした姿勢が解けない")
        return list(q)

    def measure(tag, q_cmd):
        """手首のカメラで測って、爪から見た物の位置を返す（見えなければ止める）。"""
        import cv2
        col, depth = cam.grab()
        cv2.imwrite(os.path.join(out_dir, f"wrist_{tag}.jpg"), col)
        np.save(os.path.join(out_dir, f"wrist_{tag}_depth.npy"), depth.astype(np.float16))
        tips, Rg = tips_and_axes(fk, q_cmd)
        hint = Rg.T @ (obj_base - tips)  # 計画どおりなら物の中心はここ（爪の座標）
        tgt = find_target(depth, jaw, hint_jaw=hint, keep_radius=0.03 + obj_extent / 2, max_hint_dist=0.06 + obj_extent / 2)
        if tgt is None:
            records.append({"tag": tag, "found": False, "hint_mm": (hint * 1000).round(1).tolist()})
            raise ServoAbort(f"{tag}: 手首のカメラで物が見つからない（計画の位置 {np.round(hint * 1000).tolist()} mm の近く）")
        off = offsets(tgt, jaw)
        off["e_t_raw"] = off["e_t"]
        off["e_t"] = off["e_t"] - bias_t  # 目標との差（これを 0 に追い込む）
        off["t_cover"] = tgt.get("Q") is not None and t_covered(tgt["Q"][:, 1], bias_t)
        if off["t_cover"]:  # 長い物: 爪の幅が全部物の上に乗っている。厚みの向きは直さない（端で切れていても良い）
            off["e_t"], off["t_ok"], off["t_dir"] = 0.0, True, 0.0
        vis = col.copy()
        vis[tgt["mask"]] = (0.5 * vis[tgt["mask"]] + [0, 0, 127]).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"wrist_{tag}_seg.jpg"), vis)
        rec = {"tag": tag, "found": True, "pick": tgt["pick"], "hint_mm": (hint * 1000).round(1).tolist(),
               **{k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in off.items()}}
        records.append(rec)
        aw = "—" if off["a_wide"] is None else f"{off['a_wide'] * 1000:+.0f}"
        log(f"[servo] {tag}: 閉じる {off['e_c'] * 1000:+.0f}{'' if off['c_ok'] else '(端で切れ)'} 厚み {off['e_t'] * 1000:+.0f}"
            f"{'' if off['t_ok'] else '(端で切れ)'}{'(爪の幅は物の上)' if off.get('t_cover') else ''} mm・一番太い所 {aw} mm（+ は爪の先より前）・爪の先の高さ {off['tip_h'] * 1000:.0f} mm"
            f"・{tgt['pick']}")
        return off

    def dump():
        json.dump(records, open(os.path.join(out_dir, "servo.json"), "w"), indent=1, ensure_ascii=False)

    try:
        q_near = shifted(grasp_q, [0, 0, -near], seed=grasp_q)
        check(pre_q, q_near, "pregrasp→servo near", True)
        go(q_near, speed, g_open, "servo near")
        cur = q_near
        if t_shift:
            yg = tips_and_axes(fk, q_near)[1][:, 1]
            log(f"[servo] 手を爪の厚みの向きへ {t_shift * 1000:+.0f} mm 寄せる（+Yg は付け根の座標で {np.round(yg, 2).tolist()}・"
                f"奥へ {t_shift * yg[0] * 1000:+.0f} mm）")
            records.append({"t_shift_m": t_shift, "yg_base": np.round(yg, 3).tolist()})
        total = np.zeros(3)
        off = measure("near0", cur)
        if one_shot and off["c_ok"] and off["t_ok"] and abs(off["e_c"]) <= one_shot_c_max:
            total = np.array([off["e_c"], off["e_t"], 0.0])
            if np.linalg.norm(total) > max_total:
                raise ServoAbort(f"横に直す量が {max_total * 100:.0f} cm を超える（{np.round(total * 1000).tolist()} mm）")
            travel, why = descend_travel(off, depth_target=depth_target)
            travel = min(travel, near + max_extra)
            log(f"[servo] 1 回で直して下ろす: 横 {np.round(total[:2] * 1000).tolist()} mm・下ろす {travel * 1000:.0f} mm（{why}）")
            records.append({"one_shot": True, "descend_m": round(travel, 4), "why": why, "lateral_total_m": total.round(4).tolist()})
            q = shifted(q_near, total + [0, 0, travel], seed=cur)
            check(cur, q, "servo one-shot", True)
            go(q, speed, g_open, "servo one-shot")
            records.append({"final_q_deg": [round(float(v), 2) for v in q], "lateral_total_m": total.round(4).tolist()})
            return q
        if one_shot:
            log(f"[servo] 1 回では直さない（閉じる {off['e_c'] * 1000:+.0f} mm・端で切れ: 閉じる {not off['c_ok']}・厚み {not off['t_ok']}）。"
                "何度か直してから下ろす")
        for it in range(iters):
            step, done = lateral_step(off, max_step=0.03)
            if done:
                break
            if np.linalg.norm(total + step) > max_total:
                raise ServoAbort(f"横に直す量の合計が {max_total * 100:.0f} cm を超える（{np.round((total + step) * 1000).tolist()} mm）")
            total = total + step
            q = shifted(q_near, total, seed=cur)
            check(cur, q, f"servo align {it}", True)
            go(q, speed, g_open, f"servo align {it}")
            cur = q
            off = measure(f"align{it}", cur)
        # 横が合わないまま下ろさない（2026-11-23 の 1 回目: 厚みの向きに 25 mm ずれて切れたまま下ろし、閉じて外した）
        if not (off["c_ok"] and off["t_ok"]) or max(abs(off["e_c"]), abs(off["e_t"])) > 0.010:
            raise ServoAbort(f"横を {iters} 回直しても合わない（閉じる {off['e_c'] * 1000:+.0f}・厚み {off['e_t'] * 1000:+.0f} mm・"
                             f"端で切れ: 閉じる {not off['c_ok']}・厚み {not off['t_ok']}）。下ろさない")
        travel, why = descend_travel(off, depth_target=depth_target)
        travel = min(travel, near + max_extra)
        log(f"[servo] 下ろす {travel * 1000:.0f} mm（{why}・計画の掴む姿勢より {(travel - near) * 1000:+.0f} mm）・横に直した合計 "
            f"{np.round(total * 1000).tolist()} mm")
        records.append({"descend_m": round(travel, 4), "why": why, "lateral_total_m": total.round(4).tolist()})
        steps = max(1, int(np.ceil(travel / step_m)))
        for k in range(1, steps + 1):
            q = shifted(q_near, total + [0, 0, travel * k / steps], seed=cur)
            check(cur, q, f"servo descend {k}/{steps}", True)
            go(q, speed, g_open, f"servo descend {k}/{steps}")
            cur = q
            if k == steps:
                break
            try:
                off = measure(f"descend{k}", cur)
            except ServoAbort as e:  # 近づくと物が画像の端に寄って見えなくなることがある: 直さずに続ける
                log(f"[servo] {e}（直さずに下ろし続ける）")
                continue
            step, done = lateral_step(off, max_step=0.015)
            if not done and np.linalg.norm(total + step) <= max_total:
                total = total + step
        records.append({"final_q_deg": [round(float(v), 2) for v in cur], "lateral_total_m": total.round(4).tolist()})
        return cur
    finally:
        dump()
