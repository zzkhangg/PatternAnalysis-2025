import torch
from modules import ConvNeXt
import os
from PIL import Image
from dataset import build_transform
import argparse
import torch.nn.functional as F
from dataset import get_loaders
from sklearn.metrics import classification_report


BASE_PATH = "/home/zzkhangg/code/final_report/PatternAnalysis-2025/ADNI"
test_loader = get_loaders(BASE_PATH, False, batch_size=64)
all_labels = []
all_preds = []

model = ConvNeXt(1,2)
model.load_state_dict(torch.load("convnet_adni_final.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
with torch.no_grad():
    for images, labels in test_loader:  # your test_loader
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

class_names = ['AD', 'NC']  # adapt to your dataset

report = classification_report(all_labels, all_preds, target_names=class_names)
print(report)