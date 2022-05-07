import numpy as np
from PIL import Image
import torch
from skimage.color import lab2rgb
from colorize_model.gan_functions import build_res_unet
from constants import generator_path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = build_res_unet(device=device)
generator.load_state_dict(torch.load(generator_path, map_location=device)['model_state_dict'])
generator.eval()


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
    image_ab = generator(image_l).detach()
    image_rgb = lab_to_rgb(image_l, image_ab)
    image_rgb = (image_rgb.squeeze(0) * 255).astype(np.uint8)
    image_rgb = Image.fromarray(image_rgb)
    image_rgb = image_rgb.resize(original_size, Image.Resampling.BICUBIC)
    return image_rgb
