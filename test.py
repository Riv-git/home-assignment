import torch
from shape_count_cnn import SimpleCNN          # misma definición de red
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image

# 1. Carga del modelo
model = SimpleCNN()
model.load_state_dict(torch.load("checkpoints/best.pt", map_location="cpu"))
model.eval()                                   # modo inferencia

# 2. Pre-procesado de la imagen
tfm = Compose([Resize((64, 64)), ToTensor()])
img = tfm(Image.open("shape_count/train_images/9998.png").convert("RGB")).unsqueeze(0)  # shape [1,3,64,64]

# 3. Predicción
with torch.no_grad():
    pred = model(img).round().squeeze().int()   # → tensor([circles, squares, rectangles])

print("Predicted counts:", pred.tolist())
