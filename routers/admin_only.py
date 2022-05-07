from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from io import BytesIO
from models import (
    ForwardResponse,
    MetadataResponse,
    MetricsErrorResponse,
    MetricsResponse,
    RetrainResponse,
)
from dependencies import verify_password

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

    return {'hash': 1, 'model_save_date': 2, 'experiment_name': 3}
    pass


@router.post('/add_data', response_model=None)
async def post_add_data() -> None:
    """
    Add new images to the dataset
    """
    pass


@router.post('/deploy/{experiment_id}', response_model=None)
async def deploy_model(experiment_id: int) -> None:
    """
    Replace current model
    """
    pass


@router.get('/metrics/{experiment_id}', response_model=MetricsResponse)
async def get_metrics(experiment_id: int) -> MetricsResponse:
    """
    Get experiment metrics
    """
    pass


@router.put('/retrain', response_model=RetrainResponse)
async def retrain_model() -> RetrainResponse:
    """
    Retrain current model
    """
    pass
