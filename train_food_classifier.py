import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import Food101
import os

# -------------------------------
# Configuration
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 5
NUM_CLASSES = 5
SELECTED_CLASSES = ['pizza', 'french_fries', 'ice_cream', 'hamburger', 'apple_pie']

# -------------------------------
# Transforms
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------------------------------
# Load & Filter Dataset
# -------------------------------
dataset = Food101(root='./data', download=True, transform=transform)

filtered_indices = [
    i for i, label in enumerate(dataset._labels)
    if dataset.classes[label] in SELECTED_CLASSES
]

subset = Subset(dataset, filtered_indices)
train_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------------
# Load Pretrained Model
# -------------------------------
model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace last layer & make it trainable
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(DEVICE)

# -------------------------------
# Training Setup
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# -------------------------------
# Training Loop
# -------------------------------
print("✅ Training started...")
model.train()

for epoch in range(EPOCHS):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)

        # Convert original labels to new index based on SELECTED_CLASSES
        labels = torch.tensor([SELECTED_CLASSES.index(dataset.classes[label]) for label in labels])
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")

# -------------------------------
# Save the Model
# -------------------------------
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/food_classifier.pth')
print('✅ Model saved to models/food_classifier.pth')
