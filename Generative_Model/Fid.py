import torch
import torchvision.models as models
import torch.nn as nn
import scipy
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = models.resnet50(weights=False)  # Example: ResNet-18, change as needed
num_features = classifier.fc.in_features     #extract fc layers features
classifier.fc = nn.Linear(num_features, 6)
classifier = classifier.to(device)
# Step 2: Define the path to the pre-trained weights file (if available)
pretrained_weights_path = '/content/drive/MyDrive/SER_Teess_Savee_Ravdess_classifier.pth'  # Replace with your actual path
classifier.load_state_dict(torch.load(pretrained_weights_path))

classifier.fc = torch.nn.Identity()
