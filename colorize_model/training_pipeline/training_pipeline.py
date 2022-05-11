import os
import torch
import wandb
from torch import optim
from constants import wandb_project_name, models_path
from colorize_model.training_pipeline.extract_data import get_dataloaders
from colorize_model.gan_functions import build_res_unet, PatchDiscriminator
from colorize_model.training_pipeline.training_functions import get_gan_loss, train_model


def pipeline(hyperparameters, models_metadata):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # tell wandb to get started
    with wandb.init(project=wandb_project_name, config=hyperparameters, job_type="training",
                    name=models_metadata['generator']["experiment_name"]) as run:
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config
        model_artifact = wandb.Artifact(
            "generator", type="model",
            description=models_metadata['generator']["model_description"],
            metadata=models_metadata['generator'])

        # make the model, data, and optimization problem
        generator, discriminator, criterion, optimizer_g, optimizer_d,\
            train_dataloader, val_dataloader = get_all_parts(config, run, device)
        wandb.watch((discriminator, generator))

        # and use them to train the model
        val_batch = next(iter(val_dataloader))
        train_model(generator, discriminator, optimizer_g, optimizer_d, criterion, train_dataloader, val_batch,
                    n_epochs=config.n_epochs, device=device, models_metadata=models_metadata,
                    lambda_l1=config.lambda_l1)

        path_to_save = f"{models_path}/generator.onnx"
        model_artifact.add_file(path_to_save, name="generator.onnx")
        run.log_artifact(model_artifact, aliases=[models_metadata["generator"]["experiment_name"]])
    return run.id


def get_all_parts(config, run, device="cpu"):
    # Update to the latest dataset version
    coco = run.use_artifact('coco-10k:original')
    data_path = coco.download(root="colorize_model/dataset")

    # Make the data
    train_dataloader, val_dataloader = get_dataloaders(data_path, batch_size=config.batch_size)

    # Make the model
    generator = build_res_unet(device=device)
    discriminator = PatchDiscriminator().to(device)
    if config.use_pretrained_generator:
        pretrained_artifact = run.use_artifact("pretrained_generator:baseline")
        pretrained_dir = pretrained_artifact.download("colorize_model/saved_models")
        pretrained_path = os.path.join(pretrained_dir, "pretrained_generator.pt")

        checkpoint = torch.load(pretrained_path, map_location=device)
        generator.load_state_dict(checkpoint['model_state_dict'])

    # Make the loss and optimizer
    criterion = get_gan_loss
    optimizer_g = optim.Adam(generator.parameters(),
                             lr=config.generator_learning_rate, betas=config.generator_betas)
    optimizer_d = optim.Adam(discriminator.parameters(),
                             lr=config.discr_learning_rate, betas=config.discr_betas)

    return generator, discriminator, criterion, optimizer_g, optimizer_d, train_dataloader, val_dataloader
