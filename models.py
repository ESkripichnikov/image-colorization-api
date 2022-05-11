from __future__ import annotations
from pydantic import BaseModel
from typing import List, Optional, Tuple


class ForwardResponse(BaseModel):
    colorized_image: str


class MetadataResponse(BaseModel):
    commit_hash: str
    creation_date: str
    experiment_name: str


class AddDataResponse(BaseModel):
    result: str


class ExperimentConfig(BaseModel):
    n_epochs: Optional[int] = 5
    batch_size: Optional[int] = 16
    use_pretrained_generator: Optional[bool] = True
    generator_learning_rate: Optional[float] = 2e-4
    discr_learning_rate: Optional[float] = 2e-4
    generator_betas: Optional[Tuple[float, ...]] = (0.5, 0.999)
    discr_betas: Optional[Tuple[float, ...]] = (0.5, 0.999)
    lambda_l1: Optional[float] = 100.


class ModelMetadata(BaseModel):
    experiment_name: str
    model_description: str


class ModelsMetadata(BaseModel):
    generator: ModelMetadata
    discriminator: ModelMetadata


class RetrainResponse(BaseModel):
    experiment_id: str


class MetricsResponseItem(BaseModel):
    metric_name: str
    metric_value: float


class MetricsResponse(BaseModel):
    __root__: List[MetricsResponseItem]


class DeployResponse(BaseModel):
    result: str
