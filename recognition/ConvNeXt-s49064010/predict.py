import torch
from modules import ConvNeXt
import os
from PIL import Image
from dataset import build_transform
import argparse
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---
def load_model(model_path):
    model = ConvNeXt(in_chans=1, num_classes=2, drop_path_rate=0.3)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

# --- Predict a single image ---
def predict_image(model, image_path, transform):
    img = Image.open(image_path).convert("L")
    img = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img)
        probs = F.softmax(output, dim=1)   # probability for each class
        pred_class = torch.argmax(probs, dim=1).item()
        pred_prob = probs[0, pred_class].item()  # probability of the predicted class

    class_names = ['AD', 'NC']
    return class_names[pred_class], pred_prob


# --- Predict a directory of images ---
def predict_directory(model, dir_path, transform):
    results = {}
    for fname in os.listdir(dir_path):
        fpath = os.path.join(dir_path, fname)
        if os.path.isfile(fpath) and fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            cls, prob = predict_image(model, fpath, transform)
            results[fname] = {"class": cls, "probability": prob}
    return results


# --- Main ---
def main():
    DEFAULT_MODEL_PATH = "convnet_adni_final.pth"

    parser = argparse.ArgumentParser(description="Predict AD/NC on image or directory using ConvNeXt model")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the model (.pth)")
    parser.add_argument("--input_path", type=str, required=True, help="Path to image file or directory")
    args = parser.parse_args()

    model = load_model(args.model_path)
    transform = build_transform(is_train=False)

    if os.path.isfile(args.input_path):
        cls, prob = predict_image(model, args.input_path, transform)
        print(f"{args.input_path}: {cls} ({prob:.2f})")

    elif os.path.isdir(args.input_path):
        results = predict_directory(model, args.input_path, transform)
        for fname, res in results.items():
            print(f"{fname}: {res['class']} ({res['probability']:.2f})")
    else:
        print(f"Error: {args.input_path} is neither a file nor a directory.")

if __name__ == "__main__":
    main()
