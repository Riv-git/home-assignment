# -*- coding: utf-8 -*-
"""shape_count_cnn.py – CNN to count circles, squares and rectangles.

ASCII‑only version (no Unicode bullets or long dashes).
Expected tree (relative to --root):
    shape_count/
        train_images/   1.png … 42500.png
        test_images/    42501.png … 50000.png
        train_labels.csv
        test_labels.csv

Run:
    python shape_count_cnn.py --root ./shape_count --epochs 25 --batch 128
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from PIL import Image


# --------------------- Dataset ---------------------
class ShapeCountDataset(Dataset):
    """Map images to count vectors [circles, squares, rectangles]."""

    def __init__(self, img_dir: Path, labels_csv: Path, transform=None):
        self.img_dir = Path(img_dir)
        self.labels_df = pd.read_csv(labels_csv, header=None, names=["circle", "square", "rectangle"])
        self.transform = transform or T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_path = self.img_dir / f"{idx + 1}.png"
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.labels_df.iloc[idx].values, dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label


# --------------------- Model ---------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        return self.fc(self.features(x))


# --------------------- Utils ---------------------

def mae(pred, target):
    return (pred - target).abs().mean().item()


def run_epoch(model, loader, loss_fn, opt, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss, total_mae = 0.0, 0.0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, labels)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * imgs.size(0)
            total_mae += mae(preds, labels) * imgs.size(0)
    n = len(loader.dataset)
    return total_loss / n, total_mae / n


# --------------------- Main ---------------------

def main(cfg):
    root = Path(cfg.root)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ShapeCountDataset(root / "train_images", root / "train_labels.csv")
    val_len = int(0.1 * len(dataset))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_ld = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, num_workers=4)
    val_ld = DataLoader(val_ds, batch_size=cfg.batch, shuffle=False, num_workers=4)

    model = SimpleCNN().to(device)
    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    best = float("inf")
    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_mae = run_epoch(model, train_ld, loss_fn, opt, device, train=True)
        va_loss, va_mae = run_epoch(model, val_ld, loss_fn, opt, device, train=False)
        print(f"Epoch {epoch:02d}/{cfg.epochs} | train {tr_loss:.4f}/{tr_mae:.3f} | val {va_loss:.4f}/{va_mae:.3f}")
        if va_loss < best:
            best = va_loss
            torch.save(model.state_dict(), ckpt_dir / "best.pt")

    print("Best val loss:", best)

    test_csv = root / "test_labels.csv"
    if test_csv.exists():
        test_ds = ShapeCountDataset(root / "test_images", test_csv)
        test_ld = DataLoader(test_ds, batch_size=cfg.batch, shuffle=False, num_workers=4)
        model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device))
        te_loss, te_mae = run_epoch(model, test_ld, loss_fn, opt, device, train=False)
        print(f"Test loss {te_loss:.4f} / MAE {te_mae:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Folder with shape_count data")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    main(ap.parse_args())
