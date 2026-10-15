"""Expose the public ExtrinsicManifoldData generation API."""

from .generation import GenerationOptions, generate_dataset
from .geometry import residual_transform

__all__ = ["GenerationOptions", "generate_dataset", "residual_transform"]
__version__ = "0.1.0"
