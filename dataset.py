from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- Transforms ---
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # ensure single-channel
    transforms.Resize((224, 224)),                
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])   # normalize to [-1, 1]
])

# --- Load train and test separately ---
train_dataset = datasets.ImageFolder(root='ADNI/AD_NC/train', transform=transform)
test_dataset  = datasets.ImageFolder(root='ADNI/AD_NC/test',  transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

print("Classes:", train_dataset.classes)
# ['Alzheimer', 'CognitiveNormal']

print("Train samples:", len(train_dataset))
print("Test samples:", len(test_dataset))
