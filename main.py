import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Make all leaf pictures the same size (128x128 pixels)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load leaf photos from folders
train_dataset = datasets.ImageFolder(root='./train/', transform=transform)
val_dataset = datasets.ImageFolder(root='./validate/', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class_names = train_dataset.classes


class LeafBrain(nn.Module):
    def __init__(self, num_classes):
        super(LeafBrain, self).__init__()
        # Magnifying glass 1: Finds simple lines and colors
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)

        # Magnifying glass 2: Finds spots and textures
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Magnifying glass 3: Connects spots into full diseases
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # Final decision-making layer
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Pass leaf image through magnifying glasses
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        # Flatten image into a single list of details
        x = x.view(x.size(0), -1)

        # Make the final choice
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


brain = LeafBrain(num_classes=len(class_names))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(brain.parameters(), lr=0.0001)

for epoch in range(25):
    brain.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = brain(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print("Learning complete!")

brain.eval()
all_preds = []
all_labels = []

# Test the brain without giving away answers
with torch.no_grad():
    for images, labels in val_loader:
        outputs = brain(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# Draw the Scorecard
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Guessed')
plt.ylabel('Ground Truth')
plt.title('Leaf Brain Test Results')
plt.show()