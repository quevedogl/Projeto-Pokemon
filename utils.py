"""Funcoes compartilhadas entre extracao, treinamento e deteccao."""
import cv2
import numpy as np

IMG_SIZE = 224
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm")

def letterbox(frame, size=IMG_SIZE):
    """Redimensiona sem distorcer e completa com bordas pretas."""
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Frame vazio.")
    escala = size / max(h, w)
    nh = max(1, int(round(h * escala)))
    nw = max(1, int(round(w * escala)))
    redimensionada = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = redimensionada
    return canvas
