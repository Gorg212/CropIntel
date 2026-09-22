import os

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Recreate the exact model structure
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = ['Black Rot', 'Healthy']  # Ensure order matches training alphabetically

brain = models.resnet18()
num_features = brain.fc.in_features
brain.fc = nn.Linear(num_features, len(class_names))

# 2. Load the trained weights
brain.load_state_dict(torch.load('Cauli-leaf_resnet18_model.pth', map_location=device))
brain = brain.to(device)

# 3. Set the model to evaluation mode!
brain.eval()

# Define the standard inference transform (MUST match validation normalization)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_single_image(image_path, model, transform, class_names):
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension -> (1, 3, 224, 224)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        # Apply Softmax to get percentage probabilities
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        confidence, predicted_idx = torch.max(probabilities, 0)
        predicted_class = class_names[predicted_idx.item()]

    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence.item() * 100:.2f}%")
    return predicted_class, confidence.item()


# Run a test
predict_single_image('test5.png', brain, test_transform, class_names)

# TESTING_DIR = './testing/'
# files = os.listdir(TESTING_DIR)
#
# healthy = 0
# infected = 0
# conf = 0
# count = 0
# for img in files:
#     if not (count % 10):
#         print(f'{count} images done')
#
#     count += 1
#     imgpath = f'./testing/{img}'
#     predict, confidence = predict_single_image(imgpath, brain, test_transform, class_names)
#     if predict == 'Healthy':
#         healthy += 1
#     else:
#         infected += 1
#
#     conf += confidence
#
# print(f'Healthy: {healthy} \nInfected: {infected} \nOverall confidence: {conf/150}')