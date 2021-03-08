# type: ignore[attr-defined]
"""3D line art engine (Python port of fogleman/ln)"""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version


try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

from . import paths, utility
from .paths import Box, ClipFilter, Filter, Hit, Path, Paths
from .shape import BooleanShape, Op, Shape, TransformedShape
from .shapes import (
    Cone,
    Cube,
    Cylinder,
    Direction,
    Function,
    Mesh,
    OutlineCone,
    OutlineCylinder,
    OutlineSphere,
    Plane,
    Scene,
    Sphere,
    StripedCube,
    TransformedOutlineCone,
    Triangle,
)
from .tree import Node, Tree

__all__ = [
    "paths",
    "shape",
    "utility",
    "Path",
    "Paths",
    "Box",
    "Hit",
    "Filter",
    "ClipFilter",
    "Shape",
    "BooleanShape",
    "TransformedShape",
    "Tree",
    "Node",
    "Cone",
    "Cube",
    "StripedCube",
    "Cylinder",
    "Mesh",
    "Op",
    "OutlineCone",
    "OutlineCylinder",
    "OutlineSphere",
    "Plane",
    "Scene",
    "Sphere",
    "TransformedOutlineCone",
    "Triangle",
    "Function",
    "Direction",
]
