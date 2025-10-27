from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

# --- Transforms ---
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),       # ensure single-channel
    transforms.Resize((224, 224)),

    # --- Data Augmentations ---
    transforms.RandomHorizontalFlip(p=0.5),           # flip left-right, safe for brain scans
    transforms.RandomRotation(degrees=10),            # small rotations
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # slight translations
    transforms.ColorJitter(brightness=0.1, contrast=0.1),        # slight intensity changes

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])      # normalize to [-1, 1]
])



# --- Load train and test separately ---
train_dataset = datasets.ImageFolder(root='ADNI/AD_NC/train', transform=transform)
test_dataset  = datasets.ImageFolder(root='ADNI/AD_NC/test',  transform=transform)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=512, shuffle=False)

test_loader  = DataLoader(test_dataset,  batch_size=512, shuffle=False)

print("Classes:", train_dataset.classes)
print("Test samples:", len(test_dataset))
