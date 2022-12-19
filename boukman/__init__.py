__all__ = (
    "__version__",
    "to_normalized_adjacency_matrix",
    "get_path_from_matrix",
    "path_as_brightway_objects",
)

import importlib.metadata
from typing import Union

from .graph_traversal import to_normalized_adjacency_matrix, get_path_from_matrix, path_as_brightway_objects

def get_version_tuple() -> tuple:
    def as_integer(x: str) -> Union[int, str]:
        try:
            return int(x)
        except ValueError:
            return x

    return tuple(
        as_integer(v)
        for v in importlib.metadata.version("boukman")
        .strip()
        .split(".")
    )

__version__ = get_version_tuple()