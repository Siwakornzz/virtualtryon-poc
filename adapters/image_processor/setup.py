import os
import urllib.request

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def download_model(url, filename):
    if not os.path.exists(os.path.join(MODEL_DIR, filename)):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, os.path.join(MODEL_DIR, filename))
    else:
        print(f"{filename} already exists.")

if __name__ == "__main__":
    # U-2-Net
    download_model("https://huggingface.co/lilpotat/pytorch3d/resolve/346374a95673795896e94398d65700cb19199e31/u2net.pth?download=true", "u2net.pth")
    # LIP-Parsing
    download_model("https://huggingface.co/aravindhv10/Self-Correction-Human-Parsing/resolve/3d207b0b21209372af194a57fb083cd1866233f6/checkpoints/lip.pth?download=true", "lip_atr.pth")
