import os
import csv
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder

# ----------------------------
# Preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ----------------------------
# CNN Model
# ----------------------------
class CNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128*8*8, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ----------------------------
# Load classes
# ----------------------------
dataset = ImageFolder("characters")
classes = dataset.classes
num_classes = len(classes)

# Load trained model
model = CNN(num_classes)
model.load_state_dict(torch.load("best3.pth", map_location="cpu"))
model.eval()

# ----------------------------
# Prediction function
# ----------------------------
def predict_image(img_path):
    img = Image.open(img_path)
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    return classes[predicted.item()]

# ----------------------------
# Process plates
# ----------------------------
root_test = "corepted_char"
csv_output = "plates_predictions.csv"
txt_output = "plates_predictions.txt"

rows = []

for plate_folder in sorted(os.listdir(root_test)):
    full_plate_path = os.path.join(root_test, plate_folder)
    if not os.path.isdir(full_plate_path):
        continue

    # Sort images numerically (1.jpg, 2.jpg, 3.jpg ...)
    files = [f for f in os.listdir(full_plate_path) if f.lower().endswith((".jpg", ".jpeg"))]
    files.sort(key=lambda x: int(os.path.splitext(x)[0]))  # sort by numeric part

    predictions = []
    for file in files:
        img_path = os.path.join(full_plate_path, file)
        pred = predict_image(img_path)
        predictions.append(pred)

    # Reverse for Arabic RTL order
    predictions_rtl = predictions[::]

    row_csv = [plate_folder, " ".join(predictions_rtl)]
    rows.append(row_csv)

# ----------------------------
# Save TXT
# ----------------------------
with open(txt_output, "w", encoding="utf-8") as f:
    for plate, chars in rows:
        f.write(f"{plate} {chars}\n")

print(f"\nPredictions saved to:\nTXT: {txt_output}")
