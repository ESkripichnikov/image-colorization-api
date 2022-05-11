from colorize_model.training_pipeline.training_pipeline import pipeline

config = dict(
    n_epochs=2,
    batch_size=16,
    use_pretrained_generator=True,
    generator_learning_rate=2e-4,
    discr_learning_rate=2e-4,
    generator_betas=(0.5, 0.999),
    discr_betas=(0.5, 0.999),
    lambda_l1=100.,
)

models_metadata = {
    "generator": {
        "experiment_name": "new_lr",
        "model_description": "Generator from GAN converted from PyTorch",
    },
    "discriminator": {
        "experiment_name": "new_lr",
        "model_description": "Discriminator from GAN converted from PyTorch",
    }
}

run_path = pipeline(config, models_metadata)
