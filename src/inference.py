import torch
from torchvision import models, transforms
from PIL import Image
import argparse
import os

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint['class_names']
    num_classes = len(class_names)
    model = models.convnext_tiny(pretrained=False)
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, class_names

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    return preprocess(img).unsqueeze(0)

def predict(image_path, model, class_names, device):
    input_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
        predicted_class = class_names[pred.item()]
    return predicted_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cattle/Buffalo Breed Classifier Inference")
    parser.add_argument('image', type=str, help='Path to the image file')
    parser.add_argument('--weights', type=str, default='../models/convnext_tiny_best_breed-classifier.pth', help='Path to the trained weights')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(args.weights, device)

    if not os.path.isfile(args.image):
        print(f"Image file not found: {args.image}")
        exit(1)

    pred_class = predict(args.image, model, class_names, device)
    print(f"Predicted Breed: {pred_class}")
