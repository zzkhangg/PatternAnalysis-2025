model = ConvNeXt(in_chans=1, num_classes=2,depths=[2,2,2,2], dims=[32,64,128,256])      # make sure dims, depths match your training
model.load_state_dict(torch.load("/content/checkpoints/convnet_epoch450.pth"))
model = model.to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval() 

correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():  # no gradients needed for inference
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.append(predicted.cpu())
        all_labels.append(labels.cpu())

# Calculate accuracy
test_acc = 100 * correct / total
print(f"Test Accuracy: {test_acc:.2f}%")