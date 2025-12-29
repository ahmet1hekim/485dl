import matplotlib

matplotlib.use("Agg")  # GTK hatasını önlemek için

import glob
import os
import random

import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image

# Model mimarisini çek
from model import AgeResNet

# --- AYARLAR ---
MODEL_PATH = "model/age_resnet.pth"
TEST_DATA_DIR = "data/test"  # train.py buraya resim kaydetti
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modeli Yükle
model = AgeResNet()
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
except:
    print("⚠️ Model dosyası bulunamadı, lütfen önce train.py çalıştırın.")

model.to(DEVICE)
model.eval()

# Transform (ResNet Standardı)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def predict(image):
    if image is None:
        return "Resim Yok"

    try:
        img = image.convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # Tahmin
            pred = model(img_tensor).item()

        return f"Tahmini Yaş: {pred:.1f}"
    except Exception as e:
        return f"Hata: {e}"


# --- Test Klasöründen Örnekler Çek ---
examples = []
if os.path.exists(TEST_DATA_DIR):
    # Klasördeki tüm jpg'leri bul
    all_files = glob.glob(os.path.join(TEST_DATA_DIR, "*.jpg"))
    # Rastgele 10 tane seç
    if len(all_files) > 0:
        examples = random.sample(all_files, min(len(all_files), 10))

# Arayüz
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Yüz Yükle"),
    outputs=gr.Textbox(label="Sonuç", type="text"),
    title="🧠 AI Yaş Tahmincisi (ResNet18)",
    description=f"Model: {MODEL_PATH}. Aşağıdaki örneklerden birine tıklayarak test edebilirsiniz.",
    examples=examples,  # İşte burası data klasöründen gelen resimler
)

if __name__ == "__main__":
    print(f"🔗 Örnek resimler {TEST_DATA_DIR} klasöründen çekildi.")
    demo.launch(share=True)
