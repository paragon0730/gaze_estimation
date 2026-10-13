import numpy as np

from preprocess.mpii_norm import gaze_to_3d


def angular_error(pred, gt):
    """pred, gt: (N, 2) (yaw, pitch) 라디안 → (N,) 도. README의 평가 지표와 같다."""
    a, b = gaze_to_3d(pred), gaze_to_3d(gt)
    cos = np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
