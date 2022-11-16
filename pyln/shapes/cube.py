import typing as ty

import warnings

import numba as nb
import numpy as np
from numba.core.errors import NumbaWarning

from .. import utility
from ..paths import Box, Paths
from ..shape import Shape

warnings.simplefilter("ignore", category=NumbaWarning)


class Cube(Shape):
    def __init__(self, min_box, max_box):
        super().__init__()
        self.min: np.ndarray = np.asarray(min_box, dtype=np.float64)
        self.max: np.ndarray = np.asarray(max_box, dtype=np.float64)
        self.box: Box = Box(self.min, self.max)

    def compile(self):
        pass

    def bounding_box(self) -> Box:
        return self.box

    def contains(self, v: np.ndarray, f: float) -> bool:
        for i in range(3):
            if (v[i] < (self.min[i] - f)) or (v[i] > (self.max[i] + f)):
                return False
        return True

    def intersect(self, ray_origin: np.ndarray, ray_direction: np.ndarray) -> float:
        return Cube._intersect(self.min, self.max, ray_origin, ray_direction)

    @staticmethod
    @nb.njit(
        "float64(float64[:], float64[:], float64[:], float64[:])",
        cache=True,
    )
    def _intersect(
        min_box: np.ndarray,
        max_box: np.ndarray,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
    ) -> float:
        n = (min_box - ray_origin) / ray_direction
        f = (max_box - ray_origin) / ray_direction
        n, f = np.minimum(n, f), np.maximum(n, f)
        t0, t1 = np.amax(n), np.amin(f)
        if t0 < 1e-3 < t1:
            return t1
        if 1e-3 <= t0 < t1:
            return t0
        return utility.INF

    def paths(self) -> Paths:
        x1, y1, z1 = self.min[0], self.min[1], self.min[2]
        x2, y2, z2 = self.max[0], self.max[1], self.max[2]
        paths = [
            [[x1, y1, z1], [x1, y1, z2]],
            [[x1, y1, z1], [x1, y2, z1]],
            [[x1, y1, z1], [x2, y1, z1]],
            [[x1, y1, z2], [x1, y2, z2]],
            [[x1, y1, z2], [x2, y1, z2]],
            [[x1, y2, z1], [x1, y2, z2]],
            [[x1, y2, z1], [x2, y2, z1]],
            [[x1, y2, z2], [x2, y2, z2]],
            [[x2, y1, z1], [x2, y1, z2]],
            [[x2, y1, z1], [x2, y2, z1]],
            [[x2, y1, z2], [x2, y2, z2]],
            [[x2, y2, z1], [x2, y2, z2]],
        ]
        return Paths(paths)


class StripedCube(Cube):
    def __init__(self, min_box, max_box, stripes: int):
        super().__init__(min_box, max_box)
        self.stripes = stripes

    def paths(self) -> Paths:
        paths = []
        x1, y1, z1 = self.min[0], self.min[1], self.min[2]
        x2, y2, z2 = self.max[0], self.max[1], self.max[2]
        for i in range(self.stripes):
            p = i / 10
            x = x1 + (x2 - x1) * p
            y = y1 + (y2 - y1) * p
            paths.append([[x, y1, z1], [x, y1, z2]])
            paths.append([[x, y2, z1], [x, y2, z2]])
            paths.append([[x1, y, z1], [x1, y, z2]])
            paths.append([[x2, y, z1], [x2, y, z2]])
        return Paths(paths)
