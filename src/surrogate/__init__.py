"""surrogate — DataFrame-in/DataFrame-out surrogate modelling with Deep GPs."""

from .model import SurrogateModel
from .preprocessing import DataFrameEncoder, OutputScaler

__all__ = ["SurrogateModel", "DataFrameEncoder", "OutputScaler"]
