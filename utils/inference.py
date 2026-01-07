import torch
from torchvision import transforms
from PIL import Image

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_model(model_path):
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

def predict(image_path, model):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    return {
        "BG": float(probs[0]),
        "WSSV": float(probs[1])
    }
