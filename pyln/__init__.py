# type: ignore[attr-defined]
"""3D line art engine (based on ln)"""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version


try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

from . import logic, utility
from .logic import Box, Hit, Paths, Shape
from .shapes import (
    BooleanShape,
    Cone,
    Cube,
    Cylinder,
    Direction,
    Function,
    Mesh,
    Op,
    OutlineCone,
    OutlineCylinder,
    OutlineSphere,
    Plane,
    Scene,
    Sphere,
    TransformedOutlineCone,
    TransformedShape,
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
