import torch
from constants import dataset_path
from colorize_model.training_pipeline.training_pipeline import pipeline

config = dict(
    n_epochs=2,
    batch_size=16,
    generator_learning_rate=2e-4,
    discr_learning_rate=2e-4,
    generator_betas=(0.5, 0.999),
    discr_betas=(0.5, 0.999),
    lambda_l1=100.,
)

models_metadata = {
    "generator": {
        "path_to_save": "colorize_model/saved_models/experiments/generator_2.onnx",
        "experiment_name": "new_lr",
        "model_description": "Generator from GAN converted from PyTorch",
        "model_version": 2,
    },
    "discriminator": {
        "path_to_save": "colorize_model/saved_models/experiments/discriminator_2.onnx",
        "experiment_name": "new_lr",
        "model_description": "Discriminator from GAN converted from PyTorch",
        "model_version": 2,
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator, discriminator = pipeline(dataset_path, config, models_metadata, device)
