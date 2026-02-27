import numpy as np
import mediapipe as mp

mp_hands = mp.tasks.vision.HandLandmarksConnections


def _normalize_connections(connections):
    out = []
    for c in connections:
        try:
            a, b = c[0], c[1]
            out.append((int(a), int(b)))
        except Exception:
            pass
    return out


CONNS = _normalize_connections(mp_hands.HAND_CONNECTIONS)

# Joints "reales" (constantes, no O(N^2))
JOINTS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),        # thumb
    (0, 5, 6), (5, 6, 7), (6, 7, 8),        # index
    (0, 9, 10), (9, 10, 11), (10, 11, 12),  # middle
    (0, 13, 14), (13, 14, 15), (14, 15, 16),# ring
    (0, 17, 18), (17, 18, 19), (18, 19, 20) # pinky
]


def _angle(a, b, c):
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return np.arccos(cosang)


def _robust_scale(pts, enabled_mask):
    """
    Escala robusta aunque se apaguen landmarks.
    Usa promedio de varias distancias "centrales" si están disponibles.
    Si no, cae a 1.0.
    """
    pairs = [(0, 9), (0, 5), (0, 13), (0, 17)]  # wrist->mcp
    ds = []
    for a, b in pairs:
        if enabled_mask[a] and enabled_mask[b]:
            ds.append(np.linalg.norm(pts[b] - pts[a]))
    if len(ds) == 0:
        return 1.0
    return float(np.mean(ds) + 1e-6)


def result_to_feature(result, enabled_mask=None, target_dim=64):
    """
    Features geométricas rápidas + mask-aware + dimensión fija.

    Componentes:
      1) distancias por CONNS (mask-aware)
      2) ángulos en JOINTS (mask-aware)
      3) distancias wrist->landmark (mask-aware)

    Devuelve: np.ndarray shape (target_dim,)
    """

    if enabled_mask is None:
        enabled_mask = [True] * 21
    enabled_mask = list(bool(x) for x in enabled_mask)

    # Si no hay mano: vector fijo de ceros
    if result is None or not getattr(result, "hand_landmarks", None):
        return np.zeros((target_dim,), dtype=np.float32)

    hand = result.hand_landmarks[0]
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)

    # wrist: si está apagado, no podemos definir bien distancias a wrist -> todo eso queda 0
    wrist_on = enabled_mask[0]
    wrist = pts[0] if wrist_on else np.zeros((3,), dtype=np.float32)

    scale = _robust_scale(pts, enabled_mask)

    feats = []

    # 1) distancias por conexión (estructura local)
    for a, b in CONNS:
        if enabled_mask[a] and enabled_mask[b]:
            feats.append(np.linalg.norm(pts[a] - pts[b]) / scale)
        else:
            feats.append(0.0)

    # 2) ángulos fijos (articulaciones reales)
    for a, b, c in JOINTS:
        if enabled_mask[a] and enabled_mask[b] and enabled_mask[c]:
            feats.append(_angle(pts[a], pts[b], pts[c]))
        else:
            feats.append(0.0)

    # 3) distancias a wrist (forma global)
    # Si wrist está apagado -> no agregamos info (todo 0)
    for i in range(21):
        if wrist_on and enabled_mask[i]:
            feats.append(np.linalg.norm(pts[i] - wrist) / scale)
        else:
            feats.append(0.0)

    feats = np.asarray(feats, dtype=np.float32)

    # Dimensión fija: recorta/pad
    out = np.zeros((target_dim,), dtype=np.float32)
    n = min(target_dim, feats.shape[0])
    out[:n] = feats[:n]
    return out