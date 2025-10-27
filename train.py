import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import os
from modules import ConvNeXt
from dataset import train_loader, val_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNeXt(in_chans=1, num_classes=2,depths=[2,2,2,2], dims=[32,64,128,256]).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
# Cosine Annealing scheduler
num_epochs = 450
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
# Directory to save checkpoints
CHECKPOINT_DIR = "/content/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

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
    val_correct = 0
    val_total = 0
    val_loss = 0.0
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
    

    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
       # --- Save checkpoint every 10 epochs ---
    if (epoch + 1) % 50 == 0:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"convnet_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
    if (val_acc > 82):
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"convnet_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

    scheduler.step()  # at end of each epoch


torch.save(model.state_dict(), "convnet_adni.pth")