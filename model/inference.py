from model import XLSR
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import os
import cv2 as cv
from utils import current_dir


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

model = XLSR(32, 32)


def inference(image_path, model_path):
    model.load_state_dict(torch.load(model_path, "cpu"))
    image = Image.open(image_path).convert("RGB").resize((512, 512))
    input = Image.open(image_path).convert("RGB").resize((128, 128))
    target = image
    tensor = transform(image).unsqueeze(0)
    output = model(tensor)
    result = output[0].detach().cpu().add(1.0).mul(0.5).mul(255.0).numpy().transpose(1, 2, 0).astype(np.uint8)

    psnr = cv.PSNR(result, np.array(target).astype(np.uint8))
    final = Image.new("RGB", size=(512*3, 512), color=(0, 0, 0))
    result = Image.fromarray(result)
    final.paste(input, (192, 192))
    final.paste(result, (512, 0))
    final.paste(target, (1024, 0))
    final.save(os.path.join(current_dir, f"output/{os.path.basename(image_path)}"))
    print(psnr)


