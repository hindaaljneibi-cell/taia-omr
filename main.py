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

@app.post("/scan")
def scan_omr(data: ScanRequest):

    response = requests.get(data.image_url)
    image = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)[1]

    h, w = thresh.shape

    answers = []

    # You will tune these ratios
    start_x = int(w * 0.68)
    bubble_size = int(w * 0.045)
    vertical_spacing = int(h * 0.045)
    question_start = int(h * 0.35)

    for q in range(data.questions):
        max_fill = 0
        selected = None

        for i, letter in enumerate(["A","B","C","D"]):
            y = question_start + q * vertical_spacing + i * int(vertical_spacing/4)
            x = start_x

            bubble = thresh[y:y+bubble_size, x:x+bubble_size]
            fill = cv2.countNonZero(bubble)

            if fill > max_fill:
                max_fill = fill
                selected = letter

        answers.append({"q": q+1, "selected": selected})

    return {
        "ok": True,
        "answers": answers
    }
