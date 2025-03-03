from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
from u2net import U2NET

# Load U-2-Net model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device for U-2-Net: {device}")
model = U2NET().eval()
model_path = "models/u2net.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

def remove_background(image_file):
    image = Image.open(image_file).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        d1, *_ = model(input_tensor)
        pred = torch.sigmoid(d1[:, 0, :, :])
        pred = (pred > 0.5).float()

    mask = pred.squeeze().cpu().numpy()
    mask = Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size)
    image_np = np.array(image)
    image_np[np.array(mask) == 0] = 255
    return Image.fromarray(image_np)