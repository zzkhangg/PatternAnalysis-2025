import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split



# --- Transformations ---
def build_transform(is_train=True):
    """
    Builds a set of image transformations for preprocessing MRI data.

    Args:
        is_train (bool): Whether to apply training augmentations. 
                         If False, only basic preprocessing is applied.

    Returns:
        torchvision.transforms.Compose: The composed set of transformations.
    """
    if is_train:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5), #
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

def get_loaders(base_path, is_train=True, batch_size=256, validate_split=0.1):
    """
    Return DataLoaders for AD/NC dataset.

    Args:
        base_path (str): Root path of AD_NC dataset.
        is_train (bool): If True, return train_loader, val_loader; if False, return test_loader.
        batch_size (int): Batch size for train/val loaders.
        validate_split (float): Fraction of training patients for validation.

    Returns:
        If is_train=True: train_loader, val_loader
        If is_train=False: test_loader
    """
    if is_train:
        # --- Load Base Dataset (train folder) ---
        base_dataset = datasets.ImageFolder(root=f"{base_path}/AD_NC/train", transform=None)

        # --- Extract patient IDs ---
        all_paths = [sample[0] for sample in base_dataset.samples]
        patient_ids = [os.path.basename(p).split("_")[0] for p in all_paths]

        # --- Map patient IDs to dataset indices ---
        id_to_indices = {}
        for idx, pid in enumerate(patient_ids):
            id_to_indices.setdefault(pid, []).append(idx)

        unique_patients = list(id_to_indices.keys())

        # --- Split by patient ID ---
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

        # --- Dataloaders ---
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        return train_loader, val_loader

    else:
        # --- Load Test Dataset ---
        test_dataset = datasets.ImageFolder(
            root=f"{base_path}/AD_NC/test",
            transform=build_transform(is_train=False)
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        return test_loader
