# shape_counter.py
from pathlib import Path
from functools import lru_cache

import torch
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # .../home-assignment
sys.path.append(str(PROJECT_ROOT))                   # garantiza encontrar raíz

from shape_count_cnn import SimpleCNN  

# ----------------------- one-time model load -------------------------
@lru_cache(maxsize=1)          # keep the model in memory between calls
def _get_model(device="cpu"):
    ckpt = Path(__file__).resolve().parents[1] / "checkpoints" / "best.pt"
#               ▲ sube solo 1 nivel (de server/ → raíz del proyecto)
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model

# ----------------------- prediction util -----------------------------
_transform = Compose([Resize((64, 64)), ToTensor()])

def predict_counts(image_path: str | Path,
                   device: str | torch.device = "cpu") -> dict[str, int]:
    """
    Returns {"circles": C, "squares": S, "rectangles": R}
    for one RGB PNG/JPG.
    """
    img = Image.open(image_path).convert("RGB")
    batch = _transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = _get_model(device)(batch).round().cpu().int().squeeze()

    return {"circles": int(out[0]),
            "squares": int(out[1]),
            "rectangles": int(out[2])}
