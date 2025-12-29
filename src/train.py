import os
import re

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Import the model structure we defined in src/model.py
from model import AgeResNet

# --- CONFIGURATION ---
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_SAVE_PATH = "model/age_resnet.pth"

BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Keep 0 for stability, increase to 2 or 4 if on Linux for speed
NUM_WORKERS = 0


# --- DATASET CLASS (Reads from Disk) ---
class LocalFaceDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        # We only look for .jpg files that we saved
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.folder_path, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
            # We saved them as: AGE_UniqueID.jpg (e.g., 82_150.jpg)
            # So splitting by "_" and taking the first part gives the age.
            age = float(img_name.split("_")[0])
        except Exception as e:
            # If a file is corrupt, skip to the next one
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor([age], dtype=torch.float32)


# --- DATA PROCESSING (The Fix is Here) ---
def save_dataset_to_disk():
    # Safety check: if data exists, don't download again
    if os.path.exists(TRAIN_DIR) and len(os.listdir(TRAIN_DIR)) > 100:
        print("✅ Data folder already exists. Skipping download.")
        return

    print("📥 Downloading UTKFace from Hugging Face...")
    ds = load_dataset("py97/UTKFace-Cropped", split="train")

    # Shuffle for randomness
    ds = ds.shuffle(seed=42)

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # 90% Training, 10% Testing
    split_idx = int(0.9 * len(ds))

    def save_subset(subset, folder_name):
        print(f"💾 Processing and saving images to {folder_name}...")
        saved_count = 0
        skipped_count = 0

        for i, item in enumerate(tqdm(subset)):
            try:
                # 1. Get the Image
                img = item.get("image") or item.get("jpg.chip.jpg")
                if not img:
                    skipped_count += 1
                    continue

                # 2. Get the Key (Filename)
                # Format is: UTKFace/82_0_2_20170111210110290
                full_key = item.get("__key__", "")

                # 3. Extract Age (The Fix) 🛠️
                # We strip the folder part using basename -> "82_0_2_..."
                filename = os.path.basename(full_key)

                # We split by underscore -> ["82", "0", "2", ...]
                parts = filename.split("_")

                if len(parts) > 0 and parts[0].isdigit():
                    age = int(parts[0])
                else:
                    # If we can't find a number at the start, SKIP IT.
                    # Do not assume 30.
                    skipped_count += 1
                    continue

                # 4. Save to Disk
                # We rename it to "AGE_INDEX.jpg" for easy reading later
                if isinstance(img, str):
                    continue  # Skip URLs

                save_path = os.path.join(folder_name, f"{age}_{i}.jpg")
                img.save(save_path)
                saved_count += 1

            except Exception as e:
                skipped_count += 1
                continue

        print(f"   -> Successfully saved: {saved_count}")
        print(f"   -> Skipped (bad labels): {skipped_count}")

    # Run the save function for both splits
    save_subset(ds.select(range(split_idx)), TRAIN_DIR)
    save_subset(ds.select(range(split_idx, len(ds))), TEST_DIR)


# --- TRAINING LOOP ---
def main():
    # 1. Prepare Data
    save_dataset_to_disk()

    print(f"⚙️  Device: {DEVICE}")

    # Transforms: ResNet expects 224x224.
    # We add augmentation (Flip, Rotation, Color) to prevent overfitting.
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Load datasets from the folders we just created
    train_ds = LocalFaceDataset(TRAIN_DIR, transform=train_transform)
    test_ds = LocalFaceDataset(TEST_DIR, transform=val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Initialize Model
    model = AgeResNet().to(DEVICE)

    # Differential Learning Rates
    # We train the pre-trained body VERY slowly (1e-5) so we don't break its knowledge.
    # We train the new head FAST (1e-3) so it learns to predict age quickly.
    optimizer = optim.Adam(
        [
            {"params": model.net.conv1.parameters(), "lr": 1e-5},
            {"params": model.net.layer1.parameters(), "lr": 1e-5},
            {"params": model.net.layer2.parameters(), "lr": 1e-5},
            {"params": model.net.layer3.parameters(), "lr": 1e-5},
            {"params": model.net.layer4.parameters(), "lr": 1e-5},
            {"params": model.net.fc.parameters(), "lr": 1e-3},
        ]
    )

    # L1 Loss (MAE) is better for age than MSE because it's linear.
    criterion = nn.L1Loss()

    print("🚀 Training Starting...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for imgs, ages in pbar:
            imgs, ages = imgs.to(DEVICE), ages.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, ages)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        # Validation
        model.eval()
        val_loss = 0.0
        sample_preds = []
        sample_actuals = []

        with torch.no_grad():
            for i, (imgs, ages) in enumerate(val_loader):
                imgs, ages = imgs.to(DEVICE), ages.to(DEVICE)
                outputs = model(imgs)
                val_loss += criterion(outputs, ages).item()

                # Grab a few samples from the first batch to print
                if i == 0:
                    sample_preds = outputs[:5].flatten().cpu().numpy()
                    sample_actuals = ages[:5].flatten().cpu().numpy()

        avg_train = running_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        print(
            f"\n✅ Epoch {epoch + 1} | Train MAE: {avg_train:.2f} | Test MAE: {avg_val:.2f}"
        )
        print(f"👀 Samples -> Real: {sample_actuals} | Pred: {sample_preds}")

        # Save model
        os.makedirs("model", exist_ok=True)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()
