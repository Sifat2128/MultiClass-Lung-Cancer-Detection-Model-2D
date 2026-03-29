import torch
import joblib
from torchvision import models

# recreate same architecture used during training
model = models.resnet18(weights=None)

# remove final FC layer
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

# load weights
feature_extractor.load_state_dict(
    torch.load("feature_extractor.pth", map_location="cpu")
)

feature_extractor.eval()

# load scaler
scaler = joblib.load("scaler.pkl")

# load svm
svm_model = joblib.load("svm_model.pkl")

print("Models loaded successfully")