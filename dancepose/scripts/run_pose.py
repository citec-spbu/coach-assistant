import os, cv2, yaml, math
import numpy as np
import torch
from pathlib import Path
from src.utils.io_utils import ensure_dir, JsonlWriter, SimpleLogger
from src.inference.pose_infer import PoseExtractor
from src.viz.overlay import draw_pose, SmoothBuffer

def main(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    video_path = cfg["video_path"]
    out_dir = Path(cfg["output_dir"])
    ensure_dir(out_dir)

    log = SimpleLogger(out_dir / "run.log")
    log.log(f"video_path = {video_path}")
    log.log(f"config = {cfg}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.log("ERROR: cannot open video.")
        log.flush()
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 25.0
        log.log("WARN: FPS not found, fallback to 25.0")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.log(f"Video info: {W}x{H}, fps={fps}, frames={total}")

    # 初始化推理器
    pe = PoseExtractor(
        model_name=cfg["model_name"],
        device=str(cfg.get("device", "0")),
        imgsz=int(cfg.get("imgsz", 640)),
        conf=float(cfg.get("conf", 0.25)),
        iou=float(cfg.get("iou", 0.5)),
        vid_stride=int(cfg.get("vid_stride", 1)),
    )

    print(f"current device: {torch.cuda.current_device()}")

    jsonl = JsonlWriter(out_dir / "poses.jsonl")

    # 可视化输出
    save_overlay = bool(cfg.get("save_overlay", True))
    overlay_path = out_dir / "overlay.mp4"
    writer = None
    if save_overlay:
        ov_fps = fps if cfg.get("overlay_fps") in (None, 0) else float(cfg["overlay_fps"])
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(overlay_path), fourcc, ov_fps, (W, H))
        smooth = SmoothBuffer(window=int(cfg.get("smooth_window", 5)))
        kp_thresh = float(cfg.get("kp_score_thresh", 0.35))
        line_thickness = int(cfg.get("line_thickness", 2))
        point_radius = int(cfg.get("point_radius", 3))

    valid_frames, total_frames = 0, 0
    conf_sum = 0.0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        total_frames += 1
        ts = frame_idx / fps

        valid, bbox, kps_xyc, kp_mean = pe.infer_frame(frame)

        rec = {
            "frame_idx": frame_idx,
            "timestamp": round(ts, 4),
            "width": W, "height": H,
            "model": cfg["model_name"],
            "valid": bool(valid),
        }

        if valid:
            valid_frames += 1
            conf_sum += kp_mean

            # bbox xywh
            rec["bbox"] = [float(bbox[0]-bbox[2]/2), float(bbox[1]-bbox[3]/2), float(bbox[2]), float(bbox[3])]
            # keypoints as [[x,y,score], ...]
            rec["num_joints"] = int(kps_xyc.shape[0])
            rec["keypoints"] = [[float(x), float(y), float(s)] for (x, y, s) in kps_xyc]

            if save_overlay:
                kps_vis = kps_xyc.copy()
                # 仅用于可视化做轻微平滑
                kps_vis = smooth.apply(kps_vis)
                frame = draw_pose(frame, kps_vis, bbox=rec["bbox"], kp_thresh=kp_thresh,
                                  line_thickness=line_thickness, point_radius=point_radius)

        jsonl.write(rec)

        if save_overlay:
            writer.write(frame)

        frame_idx += 1

    cap.release()
    jsonl.close()
    if writer is not None:
        writer.release()

    coverage = valid_frames / max(1, total_frames)
    avg_conf = (conf_sum / valid_frames) if valid_frames > 0 else 0.0
    log.log(f"frames_total = {total_frames}")
    log.log(f"frames_valid = {valid_frames}")
    log.log(f"coverage = {coverage:.4f}")
    log.log(f"avg_kp_conf = {avg_conf:.4f}")
    log.log(f"poses.jsonl = {out_dir/'poses.jsonl'}")
    if save_overlay:
        log.log(f"overlay.mp4 = {overlay_path}")
    log.flush()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="D:/WorkSoftware/pycharm/PyCharm 2024.1.3/projects/dancepose/default.yaml")
    args = ap.parse_args()
    main(args.cfg)
