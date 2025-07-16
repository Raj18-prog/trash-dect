import cv2
import numpy as np
from ultralytics import YOLO
import cvzone

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        b, g, r = frame[y, x]
        print(f"Coordinates: ({x}, {y}), RGB: ({r}, {g}, {b})")

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

model = YOLO("best.pt")
names = model.model.names

cap = cv2.VideoCapture('t2.mp4')
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 2 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    
    results = model.track(frame, persist=True) 
    
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [-1] * len(boxes)

        masks = results[0].masks
        overlay = frame.copy()
        
        if masks is not None:
            masks = masks.xy
            
            for box, track_id, class_id, mask in zip(boxes, track_ids, class_ids, masks):
                c = names[class_id]
                x1, y1, x2, y2 = box

                if mask.size > 0:
                   mask = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))
                   cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                   cv2.fillPoly(overlay, [mask], color=(0, 0, 255))
                   cvzone.putTextRect(frame, f'{track_id}', (x2, y2), 1, 1)
                   cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

            alpha =  0.5  
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0) 

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()