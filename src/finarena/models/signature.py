"""Optional signature detector. Requires `ultralytics` (AGPL-3.0): see the README licensing note."""
import io


class SignatureDetector:
    def __init__(self, weights):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))

    def detect(self, image_bytes, conf=0.25):
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        r = self.model.predict(img, conf=conf, verbose=False)[0]
        dets = [dict(xyxy=[round(v, 1) for v in b.xyxy[0].tolist()], confidence=round(float(b.conf), 4)) for b in r.boxes]
        return dets, img.width, img.height
