from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

# โมเดล TOM (ปรับจาก CP-VTON: https://github.com/minar09/cp-vton-plus)
class CPVTON_TOM(nn.Module):
    def __init__(
        self, in_channels=6, out_channels=3
    ):  # Input: image (3) + cloth (3), Output: RGB (3)
        super(CPVTON_TOM, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),  # Output range [-1, 1]
        )

    def forward(self, image, clothing, mask):
        # Concatenate inputs (image + clothing)
        x = torch.cat((image, clothing), dim=1)

        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder with skip connections
        d1 = self.up1(b)
        d1 = torch.cat((d1, e2), dim=1)  # Skip connection
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat((d2, e1), dim=1)  # Skip connection
        output = self.dec2(d2)

        # Apply mask (optional)
        output = output * mask + image * (1 - mask)
        return output


class CPVTON_TOM_Wrapper:
    def __init__(self, model_path="models/cp_vton_tom.pth"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device for CP-VTON TOM: {device}")

        self.model = CPVTON_TOM(in_channels=6, out_channels=3)
        state_dict = torch.load(model_path, map_location=device)

        # ถ้ามี 'state_dict' prefix
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # ลบ prefix 'module.' ถ้ามี
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        self.model.load_state_dict(
            state_dict, strict=False
        )  # strict=False เพื่อข้าม keys ที่ไม่ match
        self.model.eval()
        self.model.to(device)
        self.device = device

    def predict(self, parsed_image, clothing, mask):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        img_tensor = transform(parsed_image).unsqueeze(0).to(self.device)
        cloth_tensor = transform(clothing).unsqueeze(0).to(self.device)
        mask_tensor = transforms.ToTensor()(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor, cloth_tensor, mask_tensor)
        output = output.squeeze().cpu().numpy().transpose(1, 2, 0)
        output = ((output + 1) * 127.5).astype(
            np.uint8
        )  # Denormalize from [-1, 1] to [0, 255]
        return Image.fromarray(output)


tom_model = CPVTON_TOM_Wrapper()


def overlay_clothing(parsed_image, clothing_file, mask):
    clothing = Image.open(clothing_file).convert("RGB").resize(parsed_image.size)
    result = tom_model.predict(parsed_image, clothing, mask)
    # Debug: บันทึกผลลัพธ์ก่อนส่ง
    result.save("debug/debug_result.png")
    # ถ้า TOM ไม่ overlay ชัด ใช้ manual overlay สำหรับ Phase 1
    parsed_np = np.array(parsed_image)
    clothing_np = np.array(clothing)
    mask_np = np.array(mask) / 255  # Normalize to [0, 1]
    result_np = (
        parsed_np * (1 - mask_np[..., np.newaxis])
        + clothing_np * mask_np[..., np.newaxis]
    )
    final_result = Image.fromarray(result_np.astype(np.uint8))
    final_result.save("debug/debug_manual_result.png")
    return final_result
