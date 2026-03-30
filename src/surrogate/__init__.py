"""surrogate  -- DataFrame-in/DataFrame-out surrogate modelling with Deep GPs."""

from .model import SurrogateModel
from .preprocessing import DataFrameEncoder, OutputScaler

__all__ = ["SurrogateModel", "DataFrameEncoder", "OutputScaler"]

# Optional plotting  -- available only when matplotlib is installed.
try:
    from . import plotting  # noqa: F401

    __all__ += ["plotting"]
except ImportError:
    pass
