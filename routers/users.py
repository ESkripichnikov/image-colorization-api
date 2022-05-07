from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import Response, StreamingResponse
from io import BytesIO
from typing import List
from zipfile import ZipFile
from PIL import Image
import numpy as np
from colorize_model.inference_functions import get_colorized_image
from models import ForwardResponse

router = APIRouter(tags=["For users"],
                   responses={400: {"description": "Bad request"},
                              403: {"description": "The model was unable to process the data"}},
                   )


@router.post('/forward', response_model=ForwardResponse)
async def colorize_image(image: UploadFile = File(..., description='Image in jpeg format')):
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


@router.post('/forward_batch')
async def colorize_images(images: List[UploadFile] = File(..., description="Multiple images in jpeg format")):
    """
    Colorize input images in jpeg format and response .zip
    """
    for image in images:
        if image.content_type != "image/jpeg":
            raise HTTPException(status_code=400, detail="Please, make sure that all images in jpeg format")

    images_rgb = dict()
    for image in images:
        image_ = BytesIO(image.file.read())

        try:
            image_rgb = get_colorized_image(image_)
        except Exception:
            raise HTTPException(status_code=403, detail=f"{image.filename} unable to process."
                                                        f"Please, try later")
        images_rgb[image.filename] = image_rgb

    # Create an in-memory zip file from the in-memory image file data.
    zip_file_bytes_io = BytesIO()

    with ZipFile(zip_file_bytes_io, 'w') as zip_file:
        for image_name, image_rgb in images_rgb.items():
            with BytesIO() as bytes_stream:
                image_rgb.save(bytes_stream, format="jpeg")
                zip_file.writestr(image_name, bytes_stream.getvalue())

    return StreamingResponse(iter([zip_file_bytes_io.getvalue()]),
                             media_type="application/x-zip-compressed",
                             headers={"Content-Disposition": "attachment; filename=colorized_zip"})


@router.post('/evaluate')
async def evaluate_metrics(images: List[UploadFile] = File(...,
                                                           description="Multiple color images in jpeg format")):
    """
    Colorize input images in jpeg format and response .zip with average metrics for the batch
    """
    for image in images:
        if image.content_type != "image/jpeg":
            raise HTTPException(status_code=400, detail="Please, make sure that all images in jpeg format")

    images_rgb = dict()
    metrics = {"l1_loss": 0, "mse_loss": 0}
    for image in images:
        image_ = BytesIO(image.file.read())

        try:
            image_rgb = get_colorized_image(image_)
        except Exception:
            raise HTTPException(status_code=403, detail=f"{image.filename} unable to process."
                                                        f"Please, try later")
        images_rgb[image.filename] = image_rgb

        size = image_rgb.size[0] * image_rgb.size[1]
        metrics["l1_loss"] += np.sum(abs(np.asarray(Image.open(image_)) - np.asarray(image_rgb))) / size
        metrics["mse_loss"] += np.sum(np.power(np.asarray(Image.open(image_)) - np.asarray(image_rgb), 2)) / size
    metrics["l1_loss"] = str(round(metrics["l1_loss"] / len(images), 3))
    metrics["mse_loss"] = str(round(metrics["mse_loss"] / len(images), 3))

    zip_file_bytes_io = BytesIO()

    with ZipFile(zip_file_bytes_io, 'w') as zip_file:
        for image_name, image_rgb in images_rgb.items():
            with BytesIO() as bytes_stream:
                image_rgb.save(bytes_stream, format="jpeg")
                zip_file.writestr(image_name, bytes_stream.getvalue())

    headers = {"Content-Disposition": "attachment; filename=colorized_zip"}
    headers.update(metrics)
    return StreamingResponse(iter([zip_file_bytes_io.getvalue()]),
                             media_type="application/x-zip-compressed",
                             headers=headers)
