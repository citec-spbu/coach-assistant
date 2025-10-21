import cv2
import numpy as np
from ultralytics import YOLO

def select_largest_box(boxes_xywh):
    """选择面积最大的框返回其索引；若无框则返回 -1"""
    if boxes_xywh is None or len(boxes_xywh) == 0:
        return -1
    areas = boxes_xywh[:, 2] * boxes_xywh[:, 3]
    return int(np.argmax(areas))

class PoseExtractor:
    def __init__(self, model_name="yolov8s-pose.pt", device="0", imgsz=640, conf=0.25, iou=0.5, vid_stride=1):
        self.model = YOLO(model_name)
        self.kw = dict(device=device, imgsz=imgsz, conf=conf, iou=iou, vid_stride=vid_stride)

    def infer_frame(self, frame):
        """对单帧推理，返回 (valid, bbox_xywh, keypoints_xyc[J,3], kp_score_mean)"""
        res = self.model.predict(frame, verbose=False, **self.kw)[0]
        if res is None or res.boxes is None or len(res.boxes) == 0:
            return False, None, None, 0.0

        boxes = res.boxes.xywh.cpu().numpy()          # (N,4)
        kps = res.keypoints.xy.cpu().numpy() if res.keypoints is not None else None       # (N,J,2)
        kps_conf = res.keypoints.conf.cpu().numpy() if res.keypoints is not None else None  # (N,J)

        idx = select_largest_box(boxes)
        if idx < 0 or kps is None or kps_conf is None:
            return False, None, None, 0.0

        bbox = boxes[idx]
        kp_xy = kps[idx]          # (J,2)
        kp_sc = kps_conf[idx]     # (J,)
        kp_xyc = np.concatenate([kp_xy, kp_sc[:, None]], axis=1)  # (J,3)
        return True, bbox, kp_xyc, float(kp_sc.mean())
