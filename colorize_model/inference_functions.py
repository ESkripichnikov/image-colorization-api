import os
import numpy as np
from PIL import Image
import torch
import onnxruntime
import wandb
from colorize_model.utils import lab_to_rgb
from constants import wandb_project_path


wandb_api = wandb.Api()
model_artifact = wandb_api.artifact(f"{wandb_project_path}/generator:best")
model_dir = model_artifact.download(root="colorize_model/saved_models")
model_path = os.path.join(model_dir, "generator.onnx")

ort_session = onnxruntime.InferenceSession(model_path)
metadata = ort_session.get_modelmeta()
print(f"Model Description: {metadata.description}")
print(f"Model metadata: {metadata.custom_metadata_map}")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def process_image(image_l, size=(256, 256)) -> torch.tensor:
    image_l = Image.open(image_l).convert("L")
    original_size = image_l.size
    image_l = image_l.resize(size, Image.Resampling.BICUBIC)
    image_l = np.asarray(image_l).astype("float32")
    image_l = (image_l / 255 * 100) / 50. - 1.
    image_l = torch.tensor(image_l).unsqueeze(0).unsqueeze(0)
    return image_l, original_size


def get_colorized_image(image_l) -> np.array:
    image_l, original_size = process_image(image_l, size=(256, 256))

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image_l)}
    ort_outs = ort_session.run(None, ort_inputs)
    image_ab = torch.tensor(ort_outs[0])

    image_rgb = lab_to_rgb(image_l, image_ab)
    image_rgb = (image_rgb.squeeze(0) * 255).astype(np.uint8)
    image_rgb = Image.fromarray(image_rgb)
    image_rgb = image_rgb.resize(original_size, Image.Resampling.BICUBIC)
    return image_rgb
