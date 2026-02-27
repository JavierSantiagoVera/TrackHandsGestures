import time
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------
# Visualización (estilo Meta)
# -------------------------
mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # verde (BGR en OpenCV realmente da igual para texto)

def draw_landmarks_on_image(rgb_image: np.ndarray, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Dibuja landmarks + conexiones (Tasks API)
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # Posición de texto (esquina superior izq del bbox)
        height, width, _ = annotated_image.shape
        x_coordinates = [lm.x for lm in hand_landmarks]
        y_coordinates = [lm.y for lm in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN
        text_y = max(text_y, 0)

        # handedness[0].category_name -> "Left"/"Right"
        """cv2.putText(
            annotated_image,
            f"{handedness[0].category_name}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA
        )
"""
    return annotated_image

# -------------------------
# Live stream con HandLandmarker (Tasks)
# -------------------------
MODEL_PATH = "hand_landmarker.task"
MAX_NUM_HANDS = 2
MIN_DET_CONF = 0.5
MIN_TRACK_CONF = 0.5

_latest = {"result": None, "ts": -1}

def on_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # Guardar el resultado más reciente (callback async)
    _latest["result"] = result
    _latest["ts"] = timestamp_ms

def main():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=on_result,
        num_hands=MAX_NUM_HANDS,
        min_hand_detection_confidence=MIN_DET_CONF,
        min_hand_presence_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRACK_CONF,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No pude abrir la webcam (VideoCapture(0)).")

    t0 = time.perf_counter()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        # (Opcional) espejo tipo selfie
        frame_bgr = cv2.flip(frame_bgr, 1)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # timestamp monótono creciente (MUY importante para LIVE_STREAM)
        timestamp_ms = int((time.perf_counter() - t0) * 1000)
        detector.detect_async(mp_image, timestamp_ms)

        # Dibujar último resultado disponible
        res = _latest["result"]
        if res is not None:
            annotated_rgb = draw_landmarks_on_image(frame_rgb, res)
        else:
            annotated_rgb = frame_rgb

        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        cv2.putText(annotated_bgr, "ESC to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("HandLandmarker (MediaPipe Tasks) LIVE", annotated_bgr)
        if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
