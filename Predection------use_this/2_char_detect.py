import cv2
from ultralytics import YOLO
import cvzone
import os
from datetime import datetime

# Load YOLO model for character detection
model = YOLO('best2.pt') # should detect individual characters
names = model.names

# Create output folders
char_output_folder = "corepted_char"
plate_output_folder = "corepted_plate_char"
os.makedirs(char_output_folder, exist_ok=True)
os.makedirs(plate_output_folder, exist_ok=True)

# Input folder containing cropped plates
input_folder = "cropped_plates"

# Process all images in the cropped_plates folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_folder, filename)
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"Could not load image: {image_path}")
            continue
            
        print(f"Processing: {filename}")
        
        # Get plate name without extension for subfolder naming
        plate_name = filename.split('.')[0]  # e.g., "plate1" from "plate1.jpg"
        
        # Create subfolders for this plate
        plate_char_folder = os.path.join(char_output_folder, plate_name)
        plate_boxes_folder = os.path.join(plate_output_folder, plate_name)
        os.makedirs(plate_char_folder, exist_ok=True)
        os.makedirs(plate_boxes_folder, exist_ok=True)
        
        # Resize for better processing if needed
        frame = cv2.resize(frame, (400, 200))
        
        # Detect characters in the plate image
        results = model(frame)
        
        # Collect all detected characters
        detected_characters = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                class_ids = result.boxes.cls.int().cpu().tolist()
                confidences = result.boxes.conf.cpu().numpy()
                
                for box, class_id, confidence in zip(boxes, class_ids, confidences):
                    x1, y1, x2, y2 = box
                    c = names[class_id]
                    
                    # Draw rectangle around detected character
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cvzone.putTextRect(frame, f'{c.upper()}', (x1, y1 - 10), scale=0.8, thickness=1,
                                      colorT=(255, 255, 255), colorR=(0, 0, 255), offset=3, border=1)
                    
                    # Store character data for sorting
                    detected_characters.append({
                        'box': box,
                        'class_id': class_id,
                        'class_name': c,
                        'confidence': confidence,
                        'cropped_char': frame[y1:y2, x1:x2]
                    })
        
        # Sort characters from right to left (by x1 coordinate in descending order)
        detected_characters.sort(key=lambda char: char['box'][0], reverse=True)
        
        # Save characters in right-to-left order
        character_count = 0
        for char_data in detected_characters:
            if char_data['cropped_char'].size > 0:
                char_filename = f"{plate_char_folder}/{character_count + 1}.jpg"
                cv2.imwrite(char_filename, char_data['cropped_char'])
                character_count += 1
                print(f"Saved character (R->L): {char_filename}")
        
        # Save the plate image with character detection boxes
        plate_with_boxes_filename = f"{plate_boxes_folder}/plate_with_boxes.jpg"
        cv2.imwrite(plate_with_boxes_filename, frame)
        print(f"Saved plate with boxes: {plate_with_boxes_filename}")
        
        print(f"Processed {filename}: Found {character_count} characters")
        
        # Display the processed image with detections
        cv2.imshow("Character Detection", frame)
        cv2.waitKey(100)  # Brief pause to show the image
        
cv2.destroyAllWindows()
print("Character detection completed!")
