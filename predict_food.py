import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from nutrition_db import food_nutrition


# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/food_classifier.pth"
SELECTED_CLASSES = ['pizza', 'french_fries', 'ice_cream', 'hamburger', 'apple_pie']

# Transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load the model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(SELECTED_CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Load image
img_path = r"D:\nutrisnap\data\food-101\images\pizza\s2965.jpg"
image = Image.open(img_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(DEVICE)

# Predict
with torch.no_grad():
    outputs = model(image_tensor)
    probs = torch.softmax(outputs, dim=1)
    conf, predicted_idx = torch.max(probs, 1)
    predicted_label = SELECTED_CLASSES[predicted_idx.item()]
    confidence = conf.item() * 100

# Show result
print(f"Predicted: {predicted_label} ({confidence:.2f}%)")

# Nutrition info
nutrition = food_nutrition.get(predicted_label)

if nutrition:
    print(f"\n🔍 Nutrition Info (per 100g of {predicted_label}):")
    print(f"Calories: {nutrition['calories']} kcal")
    print(f"Protein:  {nutrition['protein']} g")
    print(f"Fat:      {nutrition['fat']} g")
    print(f"Carbs:    {nutrition['carbs']} g")
else:
    print("\nNo nutrition info found for this item.")

