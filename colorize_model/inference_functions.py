import numpy as np
from PIL import Image
import torch
from constants import generator_onnx_path
import onnxruntime
from colorize_model.utils import lab_to_rgb

ort_session = onnxruntime.InferenceSession(generator_onnx_path)
metadata = ort_session.get_modelmeta()
print(f"Model Description: {metadata.description}, Version {metadata.version}")
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
