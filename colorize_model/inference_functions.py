import numpy as np
from PIL import Image
import torch
from skimage.color import lab2rgb
from constants import generator_onnx_path
import onnxruntime
import time
import matplotlib.pyplot as plt

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


def lab_to_rgb(images_l, images_ab) -> np.array:
    """
    Takes a batch of images
    """
    images_l = (images_l + 1.) * 50.
    images_ab = images_ab * 110.
    images_lab = torch.cat([images_l, images_ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    images_rgb = []
    for img in images_lab:
        img_rgb = lab2rgb(img)
        images_rgb.append(img_rgb)
    return np.stack(images_rgb, axis=0)


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


def visualize(generator, data, save=False, path=None, device="cpu"):
    generator.eval()
    generator.eval()
    with torch.no_grad():
        img_l, img_ab = data[0].to(device), data[1].to(device)
        ab_fake = generator(img_l).detach()

    fake_imgs = lab_to_rgb(img_l, ab_fake)
    real_imgs = lab_to_rgb(img_l, img_ab)
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(img_l[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"{path}colorization_{time.time()}.png")
    return real_imgs, fake_imgs
