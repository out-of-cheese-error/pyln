import typing as ty

import warnings

import numba as nb
import numpy as np
from numba.core.errors import NumbaWarning

from .. import utility
from ..paths import Box, Paths
from ..shape import Shape

warnings.simplefilter("ignore", category=NumbaWarning)


class Triangle(Shape):
    def __init__(
        self,
        v1: ty.Optional[ty.Union[ty.List[float], np.ndarray]] = None,
        v2: ty.Optional[ty.Union[ty.List[float], np.ndarray]] = None,
        v3: ty.Optional[ty.Union[ty.List[float], np.ndarray]] = None,
    ):
        super().__init__()
        self.v1: np.ndarray = (
            np.zeros(3) if v1 is None else np.asarray(v1, dtype=np.float64)
        )
        self.v2: np.ndarray = (
            np.zeros(3) if v2 is None else np.asarray(v2, dtype=np.float64)
        )
        self.v3: np.ndarray = (
            np.zeros(3) if v3 is None else np.asarray(v3, dtype=np.float64)
        )
        self.box = None
        self.update_bounding_box()

    def compile(self):
        pass

    def bounding_box(self) -> Box:
        return self.box

    def update_bounding_box(self):
        min_v = np.minimum(np.minimum(self.v1, self.v2), self.v3)
        max_v = np.maximum(np.maximum(self.v1, self.v2), self.v3)
        self.box = Box(min_v, max_v)

    def contains(self, v: np.ndarray, f: float) -> bool:
        return False

    def intersect(self, ray_origin: np.ndarray, ray_direction: np.ndarray) -> float:
        return Triangle._intersect(self.v1, self.v2, self.v3, ray_origin, ray_direction)

    @staticmethod
    @nb.njit(
        "float64(float64[:], float64[:], float64[:], float64[:], float64[:])",
        cache=True,
    )
    def _intersect(
        v1: np.ndarray,
        v2: np.ndarray,
        v3: np.ndarray,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
    ) -> float:
        e1 = v2 - v1
        e2 = v3 - v1
        px = ray_direction[1] * e2[2] - ray_direction[2] * e2[1]
        py = ray_direction[2] * e2[0] - ray_direction[0] * e2[2]
        pz = ray_direction[0] * e2[1] - ray_direction[1] * e2[0]
        det = e1[0] * px + e1[1] * py + e1[2] * pz

        if -utility.EPS < det < utility.EPS:
            return utility.INF

        inv = 1 / det
        t = ray_origin - v1
        u = (t[0] * px + t[1] * py + t[2] * pz) * inv

        if u < 0.0 or u > 1.0:
            return utility.INF

        qx = t[1] * e1[2] - t[2] * e1[1]
        qy = t[2] * e1[0] - t[0] * e1[2]
        qz = t[0] * e1[1] - t[1] * e1[0]
        v = (
            ray_direction[0] * qx + ray_direction[1] * qy + ray_direction[2] * qz
        ) * inv

        if v < 0.0 or (u + v) > 1.0:
            return utility.INF

        d = (e2[0] * qx + e2[1] * qy + e2[2] * qz) * inv
        if d < utility.EPS:
            return utility.INF
        return d

    def paths(self) -> Paths:
        return Paths([[self.v1, self.v2], [self.v2, self.v3], [self.v3, self.v1]])

    def transform(self, matrix: np.ndarray):
        self.v1 = utility.matrix_mul_position_vector(matrix, self.v1)
        self.v2 = utility.matrix_mul_position_vector(matrix, self.v2)
        self.v3 = utility.matrix_mul_position_vector(matrix, self.v3)
        self.update_bounding_box()

    def show_tree(self, level):
        result = level * " "
        for vector in [self.v1, self.v2, self.v3]:
            result += ",".join([f"{v:5.2f}" for v in vector]) + " "
        return f"{result}\n"

    def __str__(self):
        return f"V1: {self.v1}, V2: {self.v2}, V3: {self.v3}"
