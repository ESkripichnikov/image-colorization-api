from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.responses import Response
from io import BytesIO
from typing import Union
from colorize_model.inference_functions import get_colorized_image
from models import (
    ForwardResponse,
    MetricsErrorResponse,
    MetricsResponse,
)

router = APIRouter(tags=["For users"])


@router.post('/forward',
             response_model=ForwardResponse,
             responses={400: {"description": "Bad request"},
                        403: {"description": "The model was unable to process the data"}},)
async def colorize_image(image: UploadFile):
    """
    Colorize input image in jpeg format
    """
    if image.content_type != "image/jpeg":
        raise HTTPException(status_code=400, detail="Please, send image in jpeg format")

    image = BytesIO(image.file.read())

    try:
        image_rgb = get_colorized_image(image)
    except Exception:
        raise HTTPException(status_code=403, detail="Please, try later")

    with BytesIO() as output:
        image_rgb.save(output, format="jpeg")
        contents = output.getvalue()

    return Response(contents, media_type="image/jpeg")


@router.post('/forward_batch', response_model=None)
async def forward_batch() -> None:
    """
    Upload a couple of images
    """
    pass


@router.post(
    '/evaluate',
    response_model=MetricsResponse,
    responses={'403': {'model': MetricsErrorResponse}},
)
async def evaluate() -> Union[MetricsResponse, MetricsErrorResponse]:
    """
    Apply model to the whole dataset
    """
    pass
