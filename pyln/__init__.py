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

from . import logic, utility
from .logic import BooleanShape, Box, Hit, Op, Paths, Shape, TransformedShape
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

__all__ = [
    "logic",
    "utility",
    "Shape",
    "Paths",
    "Box",
    "Hit",
    "BooleanShape",
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
    "TransformedShape",
    "Triangle",
    "Function",
    "Direction",
]
