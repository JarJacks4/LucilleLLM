import cv2
import os
import numpy as np
from PIL import Image

cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)


def detect_faces(image_pil):
    image_np = np.array(image_pil)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    cropped_faces = []
    for (x, y, w, h) in faces:
        padding = int(0.1 * max(w, h))
        x_pad = max(0, x - padding)
        y_pad = max(0, y - padding)
        w_pad = min(image_pil.width - x_pad, w + 2 * padding)
        h_pad = min(image_pil.height - y_pad, h + 2 * padding)
        cropped_faces.append(image_pil.crop((x_pad, y_pad, x_pad + w_pad, y_pad + h_pad)))
    return cropped_faces or [image_pil]
