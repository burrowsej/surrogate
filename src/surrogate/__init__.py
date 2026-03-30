"""surrogate - pandas-friendly GP and Deep GP surrogates."""

from .model import SurrogateModel
from .preprocessing import DataFrameEncoder, OutputScaler

__all__ = ["SurrogateModel", "DataFrameEncoder", "OutputScaler"]

# Optional plotting - available only when matplotlib is installed.
try:
    from . import plotting  # noqa: F401

    __all__ += ["plotting"]
except ImportError:
    pass
