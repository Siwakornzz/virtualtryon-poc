from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from torchvision.models import resnet101

class SCHP(nn.Module):
    def __init__(self, num_classes=20):
        super(SCHP, self).__init__()
        # Backbone: ResNet101
        self.backbone = resnet101(pretrained=False)
        
        # ปรับจาก fc เป็น convolutional layer (ตาม SCHP)
        self.backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.backbone.fc = nn.Identity()  # ลบ fc layer เดิม
        
        # Context Encoding
        self.context_encoding = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Context Encoding
        x = self.context_encoding(x)
        
        # Decoder
        x = self.decoder(x)
        
        # Upscale ให้ขนาดเท่า input
        x = nn.functional.interpolate(x, scale_factor=32, mode='bilinear', align_corners=True)
        return x

class LIPParser:
    def __init__(self, model_path="models/lip_atr.pth"):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device for LIP-Parsing: {device}")
        
        self.model = SCHP(num_classes=20)
        state_dict = torch.load(model_path, map_location=device)
        
        # ถ้ามี 'state_dict' prefix
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        # ลบ prefix 'module.' ถ้ามี
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict, strict=False)  # strict=False เพื่อข้าม keys ที่ไม่ match
        self.model.eval()
        self.model.to(device)
        self.device = device

    def predict(self, image):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
        return output

parser = LIPParser()

def parse_human(image):
    pred = parser.predict(image)
    mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
    unique_classes = np.unique(mask)
    print(f"Mask unique values: {unique_classes}")  # ดูว่า model เจอ Class อะไรบ้าง
    
    # รวม Class เสื้อชั้นใน + เสื้อคลุม
    upper_clothes_mask = (mask == 5) | (mask == 6)
    
    # Debug mask
    mask_img = Image.fromarray(upper_clothes_mask.astype(np.uint8) * 255)
    mask_img.save("debug/debug_mask_torso.png")
    
    return image, mask_img.resize(image.size)
