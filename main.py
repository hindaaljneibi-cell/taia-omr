from fastapi import FastAPI
from pydantic import BaseModel
import cv2
import numpy as np
import requests

app = FastAPI()

class ScanRequest(BaseModel):
    image_url: str
    questions: int

@app.get("/")
def health():
    return {"status": "OMR service running"}

def download_image(url: str) -> np.ndarray:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    arr = np.asarray(bytearray(r.content), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img

def circularity(contour) -> float:
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return 0.0
    return 4.0 * np.pi * (area / (peri * peri))

@app.post("/scan")
def scan_omr(data: ScanRequest):
    img = download_image(data.image_url)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold works better across lighting/print variations
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 8
    )

    h, w = th.shape

    # Focus only on the right side where bubbles are (reduces false detections)
    x0 = int(w * 0.55)
    roi = th[:, x0:w]

    # Find contours in ROI
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubbles = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 200 or area > 5000:
            continue

        circ = circularity(c)
        if circ < 0.6:  # closer to 1.0 = more circular
            continue

        x, y, bw, bh = cv2.boundingRect(c)

        # Filter roughly round shapes
        aspect = bw / float(bh) if bh else 0
        if aspect < 0.7 or aspect > 1.3:
            continue

        # Save bubble box in full-image coordinates
        bubbles.append((x + x0, y, bw, bh))

    # We expect questions * 4 bubbles
    bubbles = sorted(bubbles, key=lambda b: (b[1], b[0]))  # sort by y then x

    expected = data.questions * 4
    if len(bubbles) < expected:
        return {
            "ok": False,
            "error": f"Not enough bubbles detected. Detected={len(bubbles)}, Expected={expected}.",
            "detected_bubbles": len(bubbles)
        }

    # If we detect extra bubbles (noise), take the top expected ones (most likely the real set)
    bubbles = bubbles[:expected]

    answers = []
    idx = 0

    for q in range(1, data.questions + 1):
        group = bubbles[idx:idx+4]
        idx += 4

        # Sort group top-to-bottom (A,B,C,D)
        # In your template A is top, then B, then C, then D
        group = sorted(group, key=lambda b: b[1])

        fill_scores = []
        for (x, y, bw, bh) in group:
            pad = int(min(bw, bh) * 0.15)
            x1 = max(0, x + pad)
            y1 = max(0, y + pad)
            x2 = min(w, x + bw - pad)
            y2 = min(h, y + bh - pad)

            bubble_roi = th[y1:y2, x1:x2]
            fill = cv2.countNonZero(bubble_roi)
            fill_scores.append(fill)

        # Pick the darkest/most filled bubble
        best_i = int(np.argmax(fill_scores))
        selected = ["A", "B", "C", "D"][best_i]

        answers.append({"q": q, "selected": selected})

    return {
        "ok": True,
        "answers": answers
    }

