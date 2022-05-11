import torch
import wandb
from torch import nn
from tqdm import tqdm, trange
from constants import models_path
from colorize_model.utils import get_rgb_images
from colorize_model.get_onnx_model import save_model_onnx


def get_gan_loss(preds, target_is_real, device='cpu'):
    if target_is_real:
        targets = torch.ones(preds.shape).type(torch.float).to(device)
    else:
        targets = torch.zeros(preds.shape).type(torch.float).to(device)

    return nn.functional.binary_cross_entropy_with_logits(preds, targets)


def train_model(generator, discriminator, opt_g, opt_d, criterion, train_dataloader, val_batch,
                n_epochs, device, models_metadata, lambda_l1=100.):
    for epoch in trange(1, n_epochs + 1):
        generator.train()
        discriminator.train()
        for img_l, img_ab in tqdm(train_dataloader, desc=f"Training, epoch {epoch}", leave=False):
            img_l, img_ab = img_l.to(device), img_ab.to(device)

            ab_fake = generator(img_l)
            for param in discriminator.parameters():
                param.requires_grad = True
            fake_image = torch.cat([img_l, ab_fake], dim=1)
            real_image = torch.cat([img_l, img_ab], dim=1)
            fake_preds = discriminator(fake_image.detach())
            real_preds = discriminator(real_image)
            loss_d_fake = criterion(fake_preds, False, device)
            loss_d_real = criterion(real_preds, True, device)
            loss_d = 0.5 * (loss_d_fake + loss_d_real)
            loss_d.backward()
            opt_d.step()
            opt_d.zero_grad()

            for param in discriminator.parameters():
                param.requires_grad = False
            fake_image = torch.cat([img_l, ab_fake], dim=1)
            fake_preds = discriminator(fake_image)
            loss_g_gan = criterion(fake_preds, True, device)
            loss_g_l1 = nn.functional.l1_loss(ab_fake, img_ab) * lambda_l1
            loss_g = loss_g_gan + loss_g_l1
            loss_g.backward()
            opt_g.step()
            opt_g.zero_grad()

            wandb.log({"Discriminator Loss": loss_d.detach().mean().item(),
                       "Generator Loss": loss_g.detach().mean().item()})

        real_images, fake_images = get_rgb_images(generator, val_batch, save=False)
        wandb.log({"Real Images": [wandb.Image(img, caption="real") for img in real_images],
                   "Fake Images": [wandb.Image(img, caption="fake") for img in fake_images]})

        path_to_save = f"{models_path}/generator.onnx"
        save_model_onnx(generator, img_l, "image_l", "image_ab", path_to_save, **models_metadata["generator"])
        wandb.save(path_to_save, policy="now")

        path_to_save = f"{models_path}/discriminator.onnx"
        save_model_onnx(discriminator, fake_image, "image", "patch", path_to_save, **models_metadata["discriminator"])
        wandb.save(path_to_save, policy="now")
