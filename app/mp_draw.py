import numpy as np
import cv2
import mediapipe as mp

mp_hands = mp.tasks.vision.HandLandmarksConnections

def _normalize_connections(connections):
    out = []
    for c in connections:
        a = b = None
        if isinstance(c, (tuple, list)) and len(c) == 2:
            a, b = c[0], c[1]
        else:
            for a_key, b_key in [("start", "end"), ("from_landmark", "to_landmark"), ("from_", "to")]:
                if hasattr(c, a_key) and hasattr(c, b_key):
                    a, b = getattr(c, a_key), getattr(c, b_key)
                    break
            if a is None:
                try:
                    a, b = c[0], c[1]
                except Exception:
                    pass
        if a is None or b is None:
            continue
        out.append((int(a), int(b)))
    return out

CONNS = _normalize_connections(mp_hands.HAND_CONNECTIONS)

def _lm_to_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def draw_landmarks_filtered(rgb_image: np.ndarray, detection_result, enabled_mask):
    if enabled_mask is None:
        enabled_mask = [True] * 21
    enabled_mask = list(enabled_mask)

    out = rgb_image.copy()
    h, w, _ = out.shape

    col_on  = (0, 255, 0)
    col_off = (200, 200, 200)
    line_on  = (0, 200, 255)
    line_off = (170, 170, 170)

    for hand_landmarks in detection_result.hand_landmarks:
        for a, b in CONNS:
            ax, ay = _lm_to_px(hand_landmarks[a], w, h)
            bx, by = _lm_to_px(hand_landmarks[b], w, h)

            if enabled_mask[a] and enabled_mask[b]:
                cv2.line(out, (ax, ay), (bx, by), line_on, 2, cv2.LINE_AA)
            else:
                cv2.line(out, (ax, ay), (bx, by), line_off, 1, cv2.LINE_AA)

        for i in range(21):
            x, y = _lm_to_px(hand_landmarks[i], w, h)
            if enabled_mask[i]:
                cv2.circle(out, (x, y), 4, col_on, -1, cv2.LINE_AA)
            else:
                cv2.circle(out, (x, y), 3, col_off, -1, cv2.LINE_AA)

    return out
