import traceback
from abc import ABC, abstractmethod
from typing import Any, Tuple

from loguru import logger

from .objects import Device


class BaseModel(ABC):
    def __init__(
        self,
        model_path: str,
        model_name: str = "",
        model_description: str = "",
        **kwargs,
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.model_description = model_description
        self.model = None

    @logger.catch(onerror=lambda _: traceback.print_exc())
    @abstractmethod
    def preprocess(self, *args, **kwargs) -> Tuple[Any]:
        raise NotImplementedError

    @logger.catch(onerror=lambda _: traceback.print_exc())
    @abstractmethod
    def postprocess(self, *args, **kwargs) -> Tuple[Any]:
        raise NotImplementedError

    @logger.catch(onerror=lambda _: traceback.print_exc())
    @abstractmethod
    def model_load(self, device: Device) -> None:
        raise NotImplementedError

    def forward(self, data: Any) -> Any:
        result = self.model(data)
        return result

    @logger.catch(onerror=lambda _: traceback.print_exc())
    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args)

    @logger.catch(onerror=lambda _: traceback.print_exc())
    @abstractmethod
    def process_sample(self, image: Any) -> Any:
        raise NotImplementedError
