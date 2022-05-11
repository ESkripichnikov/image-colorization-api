import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb


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


def get_rgb_images(generator, data, show=False, save=False, path=None, device="cpu"):
    generator.eval()
    with torch.no_grad():
        img_l, img_ab = data[0].to(device), data[1].to(device)
        ab_fake = generator(img_l).detach()
        img_l, img_ab = torch.clip(img_l, -1., 1.), torch.clip(img_ab, -1., 1.)
        ab_fake = torch.clip(ab_fake, -1., 1.)

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
    if show:
        plt.show()
    if save:
        fig.savefig(f"{path}colorization_{time.time()}.png")
    return real_imgs, fake_imgs
