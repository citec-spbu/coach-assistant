import cv2
import numpy as np
from collections import deque

# COCO 17 Keypoint skeleton connections (index based on YOLOv8-Pose/COCO)
COCO_PAIRS = [
    (5, 7), (7, 9),      # Left arm
    (6, 8), (8, 10),     # Right arm
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
    (5, 6), (5, 11), (6, 12), (11, 12),  # Torso 
    (5, 1), (6, 2), (1, 2),              # shoulder to head
]

class SmoothBuffer:
    """Only used for visualization smoothing (does not change original values ​​exported to JSON)"""
    def __init__(self, window=5):
        self.window = window
        self.buf = None  # deque per joint

    def apply(self, kps: np.ndarray):
        if self.window <= 1:
            return kps
        J = kps.shape[0]
        if self.buf is None:
            self.buf = [deque(maxlen=self.window) for _ in range(J)]
        out = kps.copy()
        for j in range(J):
            self.buf[j].append(kps[j])
            arr = np.array(self.buf[j])
            out[j] = arr.mean(axis=0)
        return out

def draw_pose(frame, keypoints_xyc, bbox=None,
              kp_thresh=0.35, line_thickness=2, point_radius=3):
    """keypoints_xyc: (J,3) -> x,y,score"""
    J = keypoints_xyc.shape[0]
    # Draw a connection
    for a, b in COCO_PAIRS:
        if a < J and b < J:
            pa, pb = keypoints_xyc[a], keypoints_xyc[b]
            if pa[2] >= kp_thresh and pb[2] >= kp_thresh:
                cv2.line(frame,
                         (int(pa[0]), int(pa[1])),
                         (int(pb[0]), int(pb[1])),
                         (0, 255, 0), line_thickness)
    # Draw key points
    for j in range(J):
        x, y, s = keypoints_xyc[j]
        if s >= kp_thresh:
            cv2.circle(frame, (int(x), int(y)), point_radius, (0, 170, 255), -1)

    # Picture Frame
    if bbox is not None:
        x, y, w, h = bbox
        x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), line_thickness)
    return frame
