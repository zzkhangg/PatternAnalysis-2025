import os
import copy
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from modules import ConvNeXt
from dataset import get_loaders
from sklearn.metrics import classification_report
# --- Paths ---
BASE_PATH = "ADNI"

# For training/validation
train_loader, val_loader = get_loaders(base_path=BASE_PATH, is_train=True, batch_size=256)

test_loader = get_loaders(base_path=BASE_PATH, is_train=False, batch_size=128)
# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper parameters
num_epochs = 260
in_chans = 1
num_classes = 2
drop_path_rate = 0.3
label_smoothing = 0.15
learning_rate = 5e-4
weight_decay = 1e-4

model = ConvNeXt(in_chans=in_chans, num_classes=num_classes, drop_path_rate=drop_path_rate).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=50, T_mult=1, eta_min=5e-6
)

# Create directories
SAVE_DIR = "save"
IMG_DIR = "images"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# --- Tracking ---
train_losses, val_losses = [], []
train_accs, val_accs = [], []
# Seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Data for early stopping
best_val_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
epochs_no_improve = 0
patience = 50  # stop if val acc doesn’t improve for 50 epochs

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    avg_loss = running_loss / len(train_loader)

    # ---- Validation ----
    model.eval()
    val_correct, val_total, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)

    # ---- Early stopping based on validation accuracy ----
    if val_acc > best_val_acc: 
        best_val_acc = val_acc 
        best_model_wts = copy.deepcopy(model.state_dict()) 
        epochs_no_improve = 0 
    else: 
        epochs_no_improve += 1

    

    # Record metrics
    train_losses.append(avg_loss)
    val_losses.append(avg_val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print("-" * 80)
    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}% | ")
    print("-" * 80)

    # ---- Stop if early stopping triggered ----
    if epochs_no_improve >= patience:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break
    scheduler.step()

# ---- Load best model and save ----
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "convnet_adni_best_val.pth"))

# ---- Save final model ----
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "convnet_adni_final.pth"))

# ---- Save plots ----
plt.figure(figsize=(6, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')
plt.savefig(os.path.join(IMG_DIR, "loss_curve.png"))
plt.close()

plt.figure(figsize=(6, 5))
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy Curve')
plt.savefig(os.path.join(IMG_DIR, "accuracy_curve.png"))
plt.close()

all_labels = []
all_preds = []

model = ConvNeXt(in_chans=in_chans, num_classes=num_classes, drop_path_rate=drop_path_rate)
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "convnet_adni_final.pth"), map_location=device, weights_only=True))
model.to(device)
model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

class_names = ['AD', 'NC']

report = classification_report(all_labels, all_preds, target_names=class_names)
print(report)

print("Training complete. Best model saved.")