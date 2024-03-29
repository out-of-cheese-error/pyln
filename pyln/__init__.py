"""3D line art engine (Python port of fogleman/ln)"""

from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

from . import paths, utility
from .paths import Box, ClipFilter, Filter, Path, Paths
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
    Stl,
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
    "Stl",
    "Scene",
    "Sphere",
    "TransformedOutlineCone",
    "Triangle",
    "Function",
    "Direction",
]
