import os
import copy
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from modules import ConvNeXt
from dataset import train_loader, val_loader

num_epochs = 450
weight_decay = 1e-4
label_smoothing = 0.1
drop_path_rate = 0.1 # Rate for drop whole res block
num_classes = 2
input_channels = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

model = ConvNeXt(in_chans=input_channels, num_classes=num_classes, drop_path_rate=0.1).to(device) 
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing) 
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

# Create directories 
SAVE_DIR = "./save" 
IMG_DIR = os.path.join(SAVE_DIR, "images") 
os.makedirs(SAVE_DIR, exist_ok=True) 
os.makedirs(IMG_DIR, exist_ok=True) 

# Track metrics
train_losses, val_losses = [], [] 
train_accs, val_accs = [], [] 

# Seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Early stopping parameters
patience = 50  # stop if no improvement in val_acc for 50 epochs
best_val_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
epochs_no_improve = 0

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

print("Training complete. Best model saved.")
