import wandb
from torch import optim
from constants import wandb_project_name
from colorize_model.training_pipeline.extract_data import get_dataloaders
from colorize_model.gan_functions import build_res_unet, PatchDiscriminator
from colorize_model.training_pipeline.training_functions import get_gan_loss, train_model


def pipeline(data_path, hyperparameters, models_metadata, device="cpu"):
    # tell wandb to get started
    with wandb.init(project=wandb_project_name, config=hyperparameters,
                    name=models_metadata['generator']["experiment_name"]):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        generator, discriminator, criterion, optimizer_g, optimizer_d,\
            train_dataloader, val_dataloader = get_all_parts(data_path, config, device)
        wandb.watch((discriminator, generator))

        # and use them to train the model
        val_batch = next(iter(val_dataloader))
        train_model(generator, discriminator, optimizer_g, optimizer_d, criterion, train_dataloader, val_batch,
                    n_epochs=config.n_epochs, device=device, models_metadata=models_metadata,
                    lambda_l1=config.lambda_l1)

    return generator, discriminator


def get_all_parts(data_path, config, device="cpu"):
    # Make the data
    train_dataloader, val_dataloader = get_dataloaders(data_path, batch_size=config.batch_size)

    # Make the model
    generator = build_res_unet(device=device)
    discriminator = PatchDiscriminator().to(device)

    # Make the loss and optimizer
    criterion = get_gan_loss
    optimizer_g = optim.Adam(generator.parameters(),
                             lr=config.generator_learning_rate, betas=config.generator_betas)
    optimizer_d = optim.Adam(discriminator.parameters(),
                             lr=config.discr_learning_rate, betas=config.discr_betas)

    return generator, discriminator, criterion, optimizer_g, optimizer_d, train_dataloader, val_dataloader
