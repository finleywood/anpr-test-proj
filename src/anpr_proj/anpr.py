from ultralytics import YOLO
import torch
import easyocr
import numpy as np
from ultralytics.engine.results import Results
import cv2
from collections import defaultdict
import re

class ANPR:
  def __init__(self, model_path: str = "yolo26n.pt") -> None:
    self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    self.model = YOLO(model_path)
    self.decoders = [
        dict(decoder="greedy"),
        dict(decoder="beamsearch", beamWidth=10),
        dict(decoder="wordbeamsearch", beamWidth=10),
    ]
    self.reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

  def detect_plate_region(self, im: np.ndarray):
    results = list(self.model.predict(im, verbose=False))
    boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else []
    return boxes[0]

  def get_image_variants(self, im: np.ndarray):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    up = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return {"orig": im, "gray": gray, "clahe": clahe, "up2x": up, "otsu": otsu}

  def get_reader_candidates(self, im: np.ndarray, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"):
    results = []
    for vname, v in self.get_image_variants(im).items():
      h, w = v.shape[:2]
      for d in self.decoders:
          for _, text, conf in self.reader.recognize(v, horizontal_list=[[0, w, 0, h]], free_list=[], **d):
              results.append((text.replace(" ", "").upper(), conf, vname, d["decoder"]))
    return results

  def extract_text(self, im0: np.ndarray, bbox: np.ndarray):
    """Performs OCR on the cropped license plate region."""
    x1, y1, x2, y2 = map(int, bbox)
    roi = im0[y1:y2, x1:x2]
    candiates = self.get_reader_candidates(roi)
    return candiates

  def extract_best_candidate(self, candidates, pattern=None):
    scores = defaultdict(float)
    for text, conf, *_ in candidates:
        if pattern and not re.fullmatch(pattern, text):
            continue
        scores[text] += conf          # voting: agreement across passes adds up
    return max(scores.items(), key=lambda kv: kv[1]) if scores else None

  def get_plate(self, im: np.ndarray):
    boxes = self.detect_plate_region(im)
    if len(boxes) == 0:
      raise Exception("No license plate found in image.")
    lp_text = self.extract_text(im, boxes)
    print("--- Candiadates ---")
    for text, conf, img_type, decoder in lp_text:
      print(f"{text} {conf*100:.2f}%")
    print("-------------------")
    best_match = self.extract_best_candidate(lp_text, pattern=r"[A-Z]{2}\d{2}[A-Z]{3}")
    print(f"Best match: {best_match}")
