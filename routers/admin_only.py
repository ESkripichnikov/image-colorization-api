import aiofiles
import wandb
from typing import List
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from dependencies import verify_password
from colorize_model.inference import metadata
from colorize_model.training_pipeline.extract_data import log_dataset
from colorize_model.training_pipeline.training_pipeline import pipeline
from constants import wandb_project_name, wandb_project_path, dataset_path
from models import (
    MetadataResponse,
    AddDataResponse,
    ExperimentConfig,
    ModelsMetadata,
    DeployResponse,
    MetricsResponse,
    RetrainResponse,
)

router = APIRouter(
    prefix="/admin",
    tags=["Admin only"],
    dependencies=[Depends(verify_password)],
    responses={403: {"description": "Operation forbidden"}},
)


@router.get('/metadata', response_model=MetadataResponse)
async def get_metadata():
    """
    Get model metadata
    """
    return metadata.custom_metadata_map


@router.post('/add_data', response_model=AddDataResponse,
             responses={400: {"description": "Bad request"}})
async def add_new_data(images: List[UploadFile] = File(...,
                                                       description="Multiple color images in jpeg format")):
    """
    Add new images to the dataset
    """
    for image in images:
        if image.content_type != "image/jpeg":
            raise HTTPException(status_code=400, detail="Please, make sure that all images in jpeg format")
    for image in images:
        async with aiofiles.open(f"{dataset_path}/{image.filename}", 'wb') as out_file:
            content = await image.read()  # async read
            await out_file.write(content)  # async write

    log_dataset(name="adding_new_data", aliases=["latest", "custom"])

    return {"result": "Images added to dataset successfully"}


@router.put('/retrain', response_model=RetrainResponse)
async def run_experiment(config: ExperimentConfig, models_metadata: ModelsMetadata):
    """
    Run an experiment by training a model
    """
    run_id = pipeline(config.dict(), models_metadata.dict())
    return {"experiment_id": run_id}


@router.get('/metrics/{experiment_id}', response_model=MetricsResponse,
            responses={400: {"description": "Bad request"}})
async def get_metrics(experiment_id: str):
    """
    Get experiment metrics by experiment id
    """
    wandb_api = wandb.Api()
    run_path = f"{wandb_project_path}/{experiment_id}"
    try:
        run = wandb_api.run(run_path)
    except wandb.errors.CommError:
        raise HTTPException(status_code=400, detail="Experiment id is not valid")

    return [{"metric_name": "Generator Loss", "metric_value": run.summary['Generator Loss']},
            {"metric_name": "Discriminator Loss'", "metric_value": run.summary['Discriminator Loss']}]


@router.post('/deploy/{experiment_id}', response_model=DeployResponse,
             responses={400: {"description": "Bad request"}})
async def deploy_model(experiment_id: str):
    """
    Replace current inference model on a model from experiment id
    """
    wandb_api = wandb.Api()
    run_path = f"{wandb_project_path}/{experiment_id}"
    try:
        run = wandb_api.run(run_path)
    except wandb.errors.CommError:
        raise HTTPException(status_code=400, detail="Experiment id is not valid")
    new_artifact = run.logged_artifacts()[0]
    new_artifact.aliases.append('best')
    new_artifact.save()

    prev_artifact = wandb_api.artifact(f"{wandb_project_name}/generator:best")
    prev_artifact.aliases.remove('best')
    prev_artifact.save()

    return {'result': "Model has been replaced successfully."}
