import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ---------------------------------------------------------------------
# 1. DATA TRANSFORMS & LOADERS
# ---------------------------------------------------------------------
# Standard ImageNet normalization for pretrained ResNet models
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# Training Data Transforms (With Augmentation)
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# Validation Data Transforms (Clean - No Augmentation)
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# Load Datasets from Folders
train_dataset = datasets.ImageFolder(root='./train', transform=train_transform)
val_dataset = datasets.ImageFolder(root='./validate', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class_names = train_dataset.classes
print(f"Detected Classes: {class_names}")

# ---------------------------------------------------------------------
# 2. MODEL DEFINITION (Transfer Learning via ResNet18)
# ---------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on Device: {device}")

# Load Pretrained ResNet18 with default ImageNet weights
brain = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in brain.parameters():
    param.requires_grad = False

# 2. Unfreeze only the final two convolutional blocks (layer3 and layer4)
# These deeper layers process complex textures (like disease lesions)
for param in brain.layer3.parameters():
    param.requires_grad = True
for param in brain.layer4.parameters():
    param.requires_grad = True

# Replace the final fully connected layer (fc) to match 2 output classes
num_features = brain.fc.in_features
brain.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, len(class_names))
)

brain = brain.to(device)

# ---------------------------------------------------------------------
# 3. TRAINING LOOP WITH LOSS & ACCURACY TRACKING
# ---------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
# Use AdamW optimizer with a lower learning rate for fine-tuning
optimizer = optim.AdamW(brain.parameters(), lr=0.0003, weight_decay=1e-4)

epochs = 30
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

print("\nStarting Training Pipeline...\n" + "=" * 60)

for epoch in range(epochs):
    # --- A. TRAINING PHASE ---
    brain.train()
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = brain(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

    epoch_train_loss = running_train_loss / total_train
    epoch_train_acc = (correct_train / total_train) * 100
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)

    # --- B. VALIDATION PHASE ---
    brain.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = brain(images)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

    epoch_val_loss = running_val_loss / total_val
    epoch_val_acc = (correct_val / total_val) * 100
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)

    # --- C. EPOCH SUMMARY OUTPUT ---
    print(f"Epoch [{epoch + 1:02d}/{epochs:02d}] | "
          f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.2f}% | "
          f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}%")

print("=" * 60 + "\nTraining Complete!\n")

# ---------------------------------------------------------------------
# 4. PLOT LOSS AND ACCURACY CURVES
# ---------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Plot Loss
ax[0].plot(range(1, epochs + 1), train_losses, label='Train Loss', color='blue')
ax[0].plot(range(1, epochs + 1), val_losses, label='Val Loss', color='red')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].set_title('Training & Validation Loss')
ax[0].legend()
ax[0].grid(True)

# Plot Accuracy
ax[1].plot(range(1, epochs + 1), train_accuracies, label='Train Acc', color='blue')
ax[1].plot(range(1, epochs + 1), val_accuracies, label='Val Acc', color='red')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy (%)')
ax[1].set_title('Training & Validation Accuracy')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# 5. EVALUATION & CONFUSION MATRIX
# ---------------------------------------------------------------------
brain.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = brain(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Plot Heatmap
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Leaf Classification Results (ResNet18)')
plt.tight_layout()
plt.show()

# Detailed Classification Report
print(classification_report(all_labels, all_preds, target_names=class_names))