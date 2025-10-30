import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# --- Paths ---
BASE_PATH = "/home/zzkhangg/code/final_report/PatternAnalysis-2025/ADNI"

# --- Transformations ---
def build_transform(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.05),
            transforms.RandomAdjustSharpness(sharpness_factor=1.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

# --- Load Base Dataset (train folder) ---
base_dataset = datasets.ImageFolder(root=f"{BASE_PATH}/AD_NC/train", transform=None)

# --- Extract patient IDs ---
all_paths = [sample[0] for sample in base_dataset.samples]
patient_ids = [os.path.basename(p).split("_")[0] for p in all_paths]

# --- Map patient IDs to dataset indices ---
id_to_indices = {}
for idx, pid in enumerate(patient_ids):
    id_to_indices.setdefault(pid, []).append(idx)

unique_patients = list(id_to_indices.keys())
print(f"Found {len(unique_patients)} unique patients in training data.")

# --- Split by patient ID (no overlap) ---
validate_split = 0.1
train_patients, val_patients = train_test_split(
    unique_patients, test_size=validate_split, shuffle=True, random_state=42
)

# --- Gather indices for each split ---
train_indices = [i for pid in train_patients for i in id_to_indices[pid]]
val_indices   = [i for pid in val_patients   for i in id_to_indices[pid]]

# --- Build datasets with proper transforms ---
train_dataset = Subset(
    datasets.ImageFolder(root=base_dataset.root, transform=build_transform(is_train=True)),
    train_indices
)
val_dataset = Subset(
    datasets.ImageFolder(root=base_dataset.root, transform=build_transform(is_train=False)),
    val_indices
)

# --- Load Test Dataset  ---
test_dataset = datasets.ImageFolder(root=f"{BASE_PATH}/AD_NC/test", transform=build_transform(is_train=False))

# --- Dataloaders ---
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

print(f"Split {len(unique_patients)} patients → {len(train_patients)} train, {len(val_patients)} val")
print(f"Train images: {len(train_indices)}, Val images: {len(val_indices)}, Test images: {len(test_dataset)}")
print("Classes:", base_dataset.classes)
