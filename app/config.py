# app/config.py
import os
import json

MODEL_TASK_PATH = "hand_landmarker.task"
LSTM_CKPT_PATH = "lstm_3gestures.pt"
DATASET_PATH = "dataset_lsc_3gestos.npz"

MAX_CONSEC_LOST = 999999
MIN_VALID_FRAMES = 5

# --- speed ---
DETECT_W = 640          # ancho para MediaPipe (baja a 512 si quieres más FPS)
#PREDICT_EVERY = 2       # cada cuántos frames predice
DRAW_EVERY = 1          # cada cuántos frames dibuja

# --- UI config persistente ---
UI_CFG_PATH = "ui_classes.json"

DEFAULT_CLASSES = [
    {"name": "LSC_G1", "icon": ""},  # icon: ruta opcional (png/jpg)
    {"name": "LSC_G2", "icon": ""},
    {"name": "LSC_G3", "icon": ""},
]

def load_ui_classes():
    if os.path.exists(UI_CFG_PATH):
        try:
            with open(UI_CFG_PATH, "r", encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, list) and len(d) >= 1:
                return d
        except Exception:
            pass
    return DEFAULT_CLASSES

def save_ui_classes(classes_list):
    with open(UI_CFG_PATH, "w", encoding="utf-8") as f:
        json.dump(classes_list, f, ensure_ascii=False, indent=2)

# Helpers
def class_names_from_meta(meta):
    return [c.get("name", f"C{i}") for i, c in enumerate(meta)]

# app/config.py

SEQ_LEN = 60                 # ventana más larga => mejor contexto
FEATURE_DIM = 64             # debe coincidir con features.py SIEMPRE

# entrenamiento CPU rápido
TRAIN_BATCH = 32             # 32 si tu CPU se ahoga
TRAIN_EPOCHS = 40            # con early stopping casi nunca llega a 40
TRAIN_LR = 1e-3
TRAIN_WD = 1e-4
TRAIN_PATIENCE = 6           # early stopping

# realtime
PREDICT_EVERY = 15           # eval cada ~0.5s si tu cam va ~30fps (ajusta)
HOLD_SECONDS = 2.0           # mantener predicción 2s
FPS_EST = 30                 # estimado (solo para hold)
CONF_THRESH = 0.60
