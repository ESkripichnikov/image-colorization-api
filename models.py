from __future__ import annotations
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class ForwardResponse(BaseModel):
    colorized_image: str


class MetadataResponse(BaseModel):
    hash: str
    model_save_date: datetime
    experiment_name: str


class MetricsResponseItem(BaseModel):
    metric_name: str
    metric_value: float


class MetricsResponse(BaseModel):
    __root__: List[MetricsResponseItem]


class MetricsErrorResponseItem(BaseModel):
    file_name: str
    error_text: str


class MetricsErrorResponse(BaseModel):
    __root__: List[MetricsErrorResponseItem]


class RetrainResponse(BaseModel):
    experiment_id: float


class ForwardPostRequest(BaseModel):
    file: Optional[bytes] = Field(None, description='Image to upload')


class ForwardBatchPostRequest(BaseModel):
    file: Optional[bytes] = Field(None, description='Zip to upload')


class EvaluatePostRequest(BaseModel):
    file: Optional[bytes] = Field(
        None, description='Zip to upload with correct answers'
    )


class AddDataPostRequest(BaseModel):
    file: Optional[bytes] = Field(None, description='Zip to upload')
