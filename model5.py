import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# =====================================================================
# STEP 1: PREPARE DATASETS (No online augmentation needed)
# =====================================================================
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root='./train', transform=data_transform)
val_dataset = datasets.ImageFolder(root='./val', transform=data_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class_names = train_dataset.classes
print(f"Classes detected: {class_names}")

# =====================================================================
# STEP 2: CALCULATE CLASS WEIGHTS FOR IMBALANCE
# =====================================================================
# Count number of images in each class from the training set
class_counts = [0] * len(class_names)
for _, label in train_dataset.samples:
    class_counts[label] += 1

print(f"Training Class Counts: {dict(zip(class_names, class_counts))}")

# Calculate inverse weights: smaller classes get higher weights
total_samples = sum(class_counts)
class_weights = [total_samples / count for count in class_counts]

# Convert weights tensor and push to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

print(f"Applied Class Penalty Weights: {dict(zip(class_names, class_weights))}")

# =====================================================================
# STEP 3: INITIALIZE PRETRAINED RESNET18
# =====================================================================
brain = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Replace the output layer to match your 2 classes
num_features = brain.fc.in_features
brain.fc = nn.Linear(num_features, len(class_names))
brain = brain.to(device)

# =====================================================================
# STEP 4: LOSS FUNCTION WITH WEIGHTS & OPTIMIZER
# =====================================================================
# Pass the class weights into CrossEntropyLoss to handle imbalance
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

# AdamW with low learning rate for fine-tuning
optimizer = optim.AdamW(brain.parameters(), lr=0.0001, weight_decay=1e-4)

# =====================================================================
# STEP 5: TRAINING LOOP
# =====================================================================
epochs = 15  # Pretrained models converge fast

print("\nStarting Training...\n" + "=" * 50)

for epoch in range(epochs):
    # --- TRAINING PHASE ---
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

    # --- VALIDATION PHASE ---
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

    print(f"Epoch [{epoch + 1:02d}/{epochs:02d}] | "
          f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.2f}% | "
          f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}%")

print("=" * 50 + "\nTraining Complete!\n")

torch.save(brain.state_dict(), 'leaf_resnet18_model.pth')
print("Model saved successfully as 'leaf_resnet18_model.pth'!")

# =====================================================================
# STEP 6: CONFUSION MATRIX EVALUATION
# =====================================================================
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

# Generate Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Leaf Classification Matrix (Balanced Loss ResNet18)')
plt.tight_layout()
plt.show()

print(classification_report(all_labels, all_preds, target_names=class_names))