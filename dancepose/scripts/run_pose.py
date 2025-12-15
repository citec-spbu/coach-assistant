# scripts/run_pose.py
from __future__ import annotations

import os
import cv2
import yaml
import torch
import time
import queue
import threading
from pathlib import Path
from typing import Optional, Dict, Any

from dancepose.src.utils.io_utils import ensure_dir, JsonlWriter, SimpleLogger
from dancepose.src.inference.pose_infer import PoseExtractor
from dancepose.src.viz.overlay import draw_pose, SmoothBuffer


# ==========================================
# Класс для асинхронной записи видео
# ==========================================
class AsyncVideoWriter:
    def __init__(self, path, fourcc, fps, dims, queue_size=128):
        self.writer = cv2.VideoWriter(path, fourcc, fps, dims)
        self.queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._write_loop, daemon=True)
        self.thread.start()

    def write(self, frame):
        if not self.stopped:
            self.queue.put(frame)

    def _write_loop(self):
        while not self.stopped or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
                self.writer.write(frame)
                self.queue.task_done()
            except queue.Empty:
                continue

    def release(self):
        self.stopped = True
        self.thread.join()
        self.writer.release()


def _load_cfg(cfg_path: str) -> Dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pose(
        video_path: str,
        cfg_path: str = "configs/default.yaml",
        overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # --- PROFILING START ---
    t_start_total = time.time()
    times = {"read": 0.0, "infer": 0.0, "draw": 0.0, "write": 0.0}
    # -----------------------

    cfg = _load_cfg(cfg_path)
    cfg["video_path"] = video_path or cfg.get("video_path")
    if overrides:
        cfg.update(overrides)

    out_dir = Path(cfg.get("output_dir", "outputs"))
    ensure_dir(out_dir)

    log = SimpleLogger(out_dir / "run.log")

    cap = cv2.VideoCapture(cfg["video_path"])
    if not cap.isOpened():
        return {"error": "cannot open video"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pe = PoseExtractor(
        model_name=cfg["model_name"],
        device=str(cfg.get("device", "0")),
        imgsz=int(cfg.get("imgsz", 640)),
        conf=float(cfg.get("conf", 0.25)),
        iou=float(cfg.get("iou", 0.5)),
        vid_stride=int(cfg.get("vid_stride", 1)),
    )

    # Логирование устройства
    device_name = "CPU"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    print(f"Processing on: {device_name}")

    jsonl_path = out_dir / "poses.jsonl"
    jsonl = JsonlWriter(jsonl_path)

    save_overlay = bool(cfg.get("save_overlay", True))
    overlay_path = out_dir / "overlay.mp4"
    writer = None
    smooth = None

    if save_overlay:
        ov_fps = fps if cfg.get("overlay_fps") in (None, 0) else float(cfg["overlay_fps"])
        # Используем асинхронный писатель
        writer = AsyncVideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), ov_fps, (W, H))
        smooth = SmoothBuffer(window=int(cfg.get("smooth_window", 5)))

    frame_idx = 0
    valid_frames = 0

    # Параметры визуализации вынесены из цикла
    kp_thresh = float(cfg.get("kp_score_thresh", 0.35))
    line_th = int(cfg.get("line_thickness", 2))
    pt_rad = int(cfg.get("point_radius", 3))

    print(f"Start processing: {total_frames_video} frames")

    while True:
        # 1. READ
        t0 = time.time()
        ok, frame = cap.read()
        times["read"] += time.time() - t0

        if not ok:
            break

        ts = frame_idx / fps

        # 2. INFER (GPU)
        t1 = time.time()
        valid, bbox, kps_xyc, kp_mean = pe.infer_frame(frame)
        times["infer"] += time.time() - t1

        rec = {
            "frame_idx": frame_idx,
            "timestamp": round(ts, 4),
            "width": W,
            "height": H,
            "model": cfg["model_name"],
            "valid": bool(valid),
        }

        # 3. DRAW (CPU)
        t2 = time.time()
        if valid:
            valid_frames += 1
            rec["bbox"] = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
            rec["keypoints"] = [[float(x), float(y), float(s)] for (x, y, s) in kps_xyc]

            if save_overlay:
                # Рисуем только если нужно
                kps_vis = smooth.apply(kps_xyc.copy())
                # Оптимизация: draw_pose может быть медленным на больших разрешениях
                frame = draw_pose(frame, kps_vis, rec["bbox"], kp_thresh, line_th, pt_rad)

        jsonl.write(rec)
        times["draw"] += time.time() - t2

        # 4. WRITE (Async IO)
        t3 = time.time()
        if save_overlay:
            writer.write(frame)
        times["write"] += time.time() - t3

        frame_idx += 1

        # Логирование прогресса каждые 100 кадров
        if frame_idx % 100 == 0:
            print(f"   Frame {frame_idx}/{total_frames_video}...")

    cap.release()
    jsonl.close()
    if writer:
        writer.release()

    total_time = time.time() - t_start_total
    avg_fps = frame_idx / total_time if total_time > 0 else 0

    # === ОТЧЕТ ПО ПРОИЗВОДИТЕЛЬНОСТИ ===
    print("\n" + "=" * 40)
    print(f"PERFORMANCE REPORT ({total_frames_video} frames)")
    print(f"Total Time:   {total_time:.2f}s (Avg FPS: {avg_fps:.1f})")
    print("-" * 40)
    print(f"I/O Read:     {times['read']:.2f}s")
    print(f"GPU Infer:    {times['infer']:.2f}s  <-- Нейросеть")
    print(f"CPU Draw:     {times['draw']:.2f}s")
    print(f"Disk Write:   {times['write']:.2f}s  <-- Запись видео (теперь асинхронно)")
    print("=" * 40 + "\n")

    return {
        "success": True,
        "video_path": str(cfg["video_path"]),
        "video_name": Path(cfg["video_path"]).stem,
        "poses_file": str(jsonl_path),
        # 关键修复：必须包含 'overlay_mp4' 这个键，run_pose_async.py 依赖它
        "overlay_mp4": str(overlay_path) if save_overlay else None,
        "overlay_file": str(overlay_path) if save_overlay else None,
    }


# Сохраняем совместимость (копируем из старого run_pose)
def cli_main(cfg_path):
    cfg = _load_cfg(cfg_path)
    run_pose(cfg.get("video_path"), cfg_path=cfg_path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="../configs/default.yaml")
    args = ap.parse_args()
    cli_main(args.cfg)

# # scripts/run_pose.py
# from __future__ import annotations
#
# import os
# import cv2
# import yaml
# import torch
# from pathlib import Path
# from typing import Optional, Dict, Any
#
# from dancepose.src.utils.io_utils import ensure_dir, JsonlWriter, SimpleLogger
# from dancepose.src.inference.pose_infer import PoseExtractor
# from dancepose.src.viz.overlay import draw_pose, SmoothBuffer
#
#
# def _load_cfg(cfg_path: str) -> Dict[str, Any]:
#     with open(cfg_path, "r", encoding="utf-8") as f:
#         return yaml.safe_load(f)
#
#
# def run_pose(
#     video_path: str,
#     cfg_path: str = "configs/default.yaml",
#     overrides: Optional[Dict[str, Any]] = None,
# ) -> Dict[str, Any]:
#     """
#     Синхронная программируемая точка входа: выполняет извлечение одной позы из видео.
#
#     Параметры:
#       - video_path: Введите видеопуть (с более высоким приоритетом, чем video_path в файле конфигурации).
#       - cfg_path:   Путь к файлу конфигурации YAML
#       - overrides:  Необязательный словарь, который может переопределить любой ключ в конфигурации (например, {"save_overlay": False}).
#
#     Возвращает:
#     - Словарь, содержащий путь к выходному файлу и основные статистические данные:
#         {
#           "video_path": <abs path>,
#           "output_dir": <abs path>,
#           "poses_jsonl": <abs path>,
#           "overlay_mp4": <abs path or None>,
#           "stats": {"frames": int, "valid_frames": int, "avg_kp_conf": float, "coverage": float}
#         }
#     """
#     cfg = _load_cfg(cfg_path)
#     cfg["video_path"] = video_path or cfg.get("video_path")
#     if overrides:
#         cfg.update(overrides)
#
#     out_dir = Path(cfg.get("output_dir", "outputs"))
#     ensure_dir(out_dir)
#
#     log = SimpleLogger(out_dir / "run.log")
#     log.log(f"video_path = {cfg['video_path']}")
#     log.log(f"config = {cfg}")
#
#     cap = cv2.VideoCapture(cfg["video_path"])
#     if not cap.isOpened():
#         log.log("ERROR: cannot open video.")
#         log.flush()
#         return {
#             "video_path": os.fspath(Path(cfg["video_path"]).resolve()),
#             "output_dir": os.fspath(out_dir.resolve()),
#             "poses_jsonl": None,
#             "overlay_mp4": None,
#             "stats": {"frames": 0, "valid_frames": 0, "avg_kp_conf": 0.0, "coverage": 0.0},
#         }
#
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     if not fps or fps <= 1e-3:
#         fps = 25.0
#         log.log("WARN: FPS not found, fallback to 25.0")
#
#     W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     log.log(f"Video info: {W}x{H}, fps={fps}, frames={total}")
#
#     # Инициализируйте механизм вывода
#     pe = PoseExtractor(
#         model_name=cfg["model_name"],
#         device=str(cfg.get("device", "0")),
#         imgsz=int(cfg.get("imgsz", 640)),
#         conf=float(cfg.get("conf", 0.25)),
#         iou=float(cfg.get("iou", 0.5)),
#         vid_stride=int(cfg.get("vid_stride", 1)),
#     )
#
#     # Информация об устройстве печати
#     if torch.cuda.is_available():
#         try:
#             dev = torch.cuda.current_device()
#             print(f"Using GPU: {torch.cuda.get_device_name(dev)} (index {dev})")
#         except Exception:
#             print("Using GPU")
#     else:
#         print("Using CPU")
#
#     jsonl_path = out_dir / "poses.jsonl"
#     jsonl = JsonlWriter(jsonl_path)
#
#     # Визуальный вывод
#     save_overlay = bool(cfg.get("save_overlay", True))
#     overlay_path = out_dir / "overlay.mp4"
#     writer = None
#     if save_overlay:
#         ov_fps = fps if cfg.get("overlay_fps") in (None, 0) else float(cfg["overlay_fps"])
#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#         writer = cv2.VideoWriter(str(overlay_path), fourcc, ov_fps, (W, H))
#         smooth = SmoothBuffer(window=int(cfg.get("smooth_window", 5)))
#         kp_thresh = float(cfg.get("kp_score_thresh", 0.35))
#         line_thickness = int(cfg.get("line_thickness", 2))
#         point_radius = int(cfg.get("point_radius", 3))
#
#     valid_frames, total_frames = 0, 0
#     conf_sum = 0.0
#     frame_idx = 0
#
#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             break
#
#         total_frames += 1
#         ts = frame_idx / fps
#
#         valid, bbox, kps_xyc, kp_mean = pe.infer_frame(frame)
#
#         rec = {
#             "frame_idx": frame_idx,
#             "timestamp": round(ts, 4),
#             "width": W,
#             "height": H,
#             "model": cfg["model_name"],
#             "valid": bool(valid),
#         }
#
#         if valid:
#             valid_frames += 1
#             conf_sum += kp_mean
#
#             rec["bbox"] = [
#                 float(bbox[0] - bbox[2] / 2),
#                 float(bbox[1] - bbox[3] / 2),
#                 float(bbox[2]),
#                 float(bbox[3]),
#             ]
#             rec["num_joints"] = int(kps_xyc.shape[0])
#             rec["keypoints"] = [[float(x), float(y), float(s)] for (x, y, s) in kps_xyc]
#
#             if save_overlay:
#                 kps_vis = kps_xyc.copy()
#                 # Небольшое сглаживание только для визуализации.
#                 kps_vis = smooth.apply(kps_vis)
#                 frame = draw_pose(
#                     frame,
#                     kps_vis,
#                     bbox=rec["bbox"],
#                     kp_thresh=kp_thresh,
#                     line_thickness=line_thickness,
#                     point_radius=point_radius,
#                 )
#
#         jsonl.write(rec)
#
#         if save_overlay:
#             writer.write(frame)
#
#         frame_idx += 1
#
#     cap.release()
#     jsonl.close()
#     if writer is not None:
#         writer.release()
#
#     coverage = valid_frames / max(1, total_frames)
#     avg_conf = (conf_sum / valid_frames) if valid_frames > 0 else 0.0
#
#     log.log(f"frames_total = {total_frames}")
#     log.log(f"frames_valid = {valid_frames}")
#     log.log(f"coverage = {coverage:.4f}")
#     log.log(f"avg_kp_conf = {avg_conf:.4f}")
#     log.log(f"poses.jsonl = {jsonl_path}")
#     if save_overlay:
#         log.log(f"overlay.mp4 = {overlay_path}")
#     log.flush()
#
#     return {
#         "video_path": os.fspath(Path(cfg["video_path"]).resolve()),
#         "output_dir": os.fspath(out_dir.resolve()),
#         "poses_jsonl": os.fspath(jsonl_path.resolve()),
#         "overlay_mp4": os.fspath(overlay_path.resolve()) if save_overlay else None,
#         "stats": {
#             "frames": total_frames,
#             "valid_frames": valid_frames,
#             "avg_kp_conf": float(avg_conf),
#             "coverage": float(coverage),
#         },
#     }
#
#
# # ---------Асинхронная упаковка для удовлетворения соглашения о вызовах await main(path) фронтенда.---------
# async def main(path: str, cfg_path: str = "configs/default.yaml", overrides: Optional[Dict[str, Any]] = None):
#     import asyncio
#     return await asyncio.to_thread(run_pose, path, cfg_path, overrides)
#
#
# # --------- Режим CLI остается совместимым：python scripts/run_pose.py --cfg configs/default.yaml ---------
# def cli_main(cfg_path: str):
#     cfg = _load_cfg(cfg_path)
#     return run_pose(cfg.get("video_path"), cfg_path=cfg_path, overrides=None)
#
#
# if __name__ == "__main__":
#     import argparse
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--cfg", type=str, default="../configs/default.yaml")
#     args = ap.parse_args()
#     cli_main(args.cfg)
