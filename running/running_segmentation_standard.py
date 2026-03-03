import numpy as np
import cv2 as cv
import supervision as sv
from PIL import Image
from rfdetr import RFDETRSegMedium
from rfdetr.util.coco_classes import COCO_CLASSES

model = RFDETRSegMedium()

# --- Load as RGB numpy ---
pil_img = Image.open("/home/jeff/Desktop/Project_Occulus/ComputerVision_WorkSpace/scripts/stable/running/testing/im1.png").convert("RGB")
img_rgb = np.array(pil_img)  # HWC, RGB, uint8

# --- Predict (RFDETR may accept PIL or np; we feed np for consistency) ---
detections = model.predict(img_rgb, threshold=0.8)
detections = detections[detections.class_id == 62]

labels = [COCO_CLASSES[cid] for cid in detections.class_id]

# --- Annotate on numpy image ---
mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_rgb = mask_annotator.annotate(scene=img_rgb.copy(), detections=detections)
annotated_rgb = label_annotator.annotate(scene=annotated_rgb, detections=detections, labels=labels)

# --- Show with OpenCV (convert RGB->BGR) ---
annotated_bgr = cv.cvtColor(annotated_rgb, cv.COLOR_RGB2BGR)
cv.imshow("frame", annotated_bgr)
cv.waitKey(0)
cv.destroyAllWindows()