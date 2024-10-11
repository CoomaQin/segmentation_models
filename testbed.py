import torch
from model import BASModel

model = BASModel("DeepLabV3", "resnext50_32x4d", in_channels=3, obj_classes=2, area_classes=4)
model.load_state_dict(torch.load("lightning_logs/version_14/checkpoints/epoch=9-step=9750.ckpt", weights_only=True)["state_dict"])
