import typing as ty

from enum import Enum

import numpy as np

from .. import utility
from ..paths import Box, Paths
from ..shape import Shape


class Direction(Enum):
    Above = 0
    Below = 1


class Function(Shape):
    def __init__(
        self,
        func: ty.Callable[[float, float], float],
        box: Box,
        direction: Direction,
    ):
        super().__init__()
        self.func = func
        self.box = box
        self.direction = direction

    def bounding_box(self) -> Box:
        return self.box

    def contains(self, v: np.ndarray, f: float) -> bool:
        if self.direction == Direction.Below:
            return v[2] < self.func(v[0], v[1])
        else:
            return v[2] > self.func(v[0], v[1])

    def intersect(
        self, ray_origin: np.ndarray, ray_direction: np.ndarray
    ) -> float:
        step = 1 / 64
        sign = self.contains(ray_origin + (step * ray_direction), 0)
        for t in np.arange(step, 10, step):
            v = ray_origin + (t * ray_direction)
            if self.contains(v, 0) != sign and self.box.contains(v):
                return t
        return utility.INF

    def paths(self) -> Paths:
        fine = 1 / 256
        n_paths = 72
        n_per_path = np.arange(0, 8, fine).shape[0]
        paths = np.zeros((n_paths, n_per_path, 3))
        for paths_index, a in enumerate(range(0, n_paths * 5, 5)):
            a = np.deg2rad(a)
            for path_index, r in enumerate(np.arange(0, 8, fine)):
                x, y = np.cos(a) * r, np.sin(a) * r
                z = self.func(x, y)
                o = -np.power(-z, 1.4)
                x, y = np.cos(a - o) * r, np.sin(a - o) * r
                z = np.minimum(z, self.box.max[2])
                z = np.maximum(z, self.box.min[2])
                paths[paths_index, path_index] = [x, y, z]
        return Paths(paths)
