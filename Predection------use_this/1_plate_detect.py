import cv2
from ultralytics import YOLO
import cvzone
import os
from datetime import datetime

# Load YOLO model
model = YOLO('best1.pt') # should detect "numberplate"
names = model.names

# Create output folder for cropped plates
output_folder = "cropped_plates"
os.makedirs(output_folder, exist_ok=True)

# Memory to avoid duplicate saves
saved_ids = set()
plate_counter = 1

# Debug mouse
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print (f"Mouse moved to: [{x}, {y}]")
cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

cap = cv2.VideoCapture("vid1.mp4")
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    if frame_count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 600))
    results = model.track(frame, persist=True)
    if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            class_ids= results[0].boxes.cls.int().cpu().tolist()
            for track_id, box, class_id in zip(ids, boxes, class_ids):
                x1, y1, x2, y2 = box
                c = names[class_id]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cvzone.putTextRect(frame, f'{c.upper()}', (x1, y1 - 10), scale=1, thickness=2,
                                    colorT=(255, 255, 255), colorR=(0, 0, 255), offset=5, border=2)

                 # Check if it's a numberplate
                if c.lower() == "numberplate":
                    cropped_plate = frame[y1:y2, x1:x2]
                    if cropped_plate.size == 0:
                         continue

                    # Save cropped plate image if not already saved
                    if track_id not in saved_ids:
                        filename = f"{output_folder}/plate{plate_counter}.jpg"
                        cv2.imwrite(filename, cropped_plate)
                        print(f"Saved cropped plate: {filename}")
                        saved_ids.add(track_id)
                        plate_counter += 1

    # Show ID on frame
    if results[0].boxes.id is not None:
        for track_id, box, class_id in zip(ids, boxes, class_ids):
            x1, y1, x2, y2 = box
            cvzone.putTextRect(frame, f"ID: {track_id}", (x1, y2 + 10), scale=1, thickness=2,
                               colorT=(255, 255, 255), colorR=(0, 255, 0), offset=5, border=2)   

    cv2.imshow("RGB", frame) 

    if cv2.waitKey(1) & 0xFF == 27: # ESC key
        break

cap.release()
cv2.destroyAllWindows()
