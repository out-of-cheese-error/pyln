from typing import Callable

from enum import Enum

import numba as nb
import numpy as np

from .. import Shape, logic, utility


class TransformedShape(Shape):
    def __init__(self, shape: Shape, matrix: np.ndarray):
        super().__init__()
        self.shape = shape
        self.matrix = matrix
        self.inverse = utility.matrix_inverse(matrix)

    def compile(self):
        self.shape.compile()

    def bounding_box(self) -> logic.Box:
        box = self.shape.bounding_box()
        min_box, max_box = utility.matrix_mul_box(self.matrix, box.min, box.max)
        return logic.Box(min_box, max_box)

    def contains(self, v: np.ndarray, f: float) -> bool:
        return self.shape.contains(
            utility.matrix_mul_position_vector(self.inverse, v), f
        )

    def intersect(
        self, ray_origin: np.ndarray, ray_direction: np.ndarray
    ) -> logic.Hit:
        return self.shape.intersect(
            utility.matrix_mul_ray(self.inverse, ray_origin, ray_direction)
        )

    def paths(self) -> logic.Paths:
        return self.shape.paths().transform(self.matrix)


class Op(Enum):
    Intersection = 0
    Difference = 1


class BooleanShape(Shape, logic.Filter):
    @staticmethod
    def from_shapes(op: Op, shapes: [Shape]) -> Shape:
        if len(shapes) == 0:
            return Shape()
        shape = shapes[0]
        for i in range(1, len(shapes)):
            shape = BooleanShape(op, shape, shapes[i])
        return shape

    def __init__(self, op: Op, shape_a: Shape, shape_b: Shape):
        super().__init__()
        self.shape_a = shape_a
        self.shape_b = shape_b
        self.op = op

    def compile(self):
        pass

    def bounding_box(self) -> logic.Box:
        return self.shape_a.bounding_box().extend(self.shape_b.bounding_box())

    def contains(self, v: np.ndarray, f: float) -> bool:
        f = 1e-3
        if self.op == Op.Intersection:
            return self.shape_a.contains(v, f) and self.shape_b.contains(v, f)
        elif self.op == Op.Difference:
            return self.shape_a.contains(v, f) and not self.shape_b.contains(
                v, -f
            )
        return False

    def intersect(
        self, ray_origin: np.ndarray, ray_direction: np.ndarray
    ) -> logic.Hit:
        hit_a = self.shape_a.intersect(ray_origin, ray_direction)
        hit_b = self.shape_b.intersect(ray_origin, ray_direction)
        hit = hit_a.min(hit_b)
        v = ray_origin + (hit.t * ray_direction)
        if (not hit.ok()) or self.contains(v, 0.0):
            return hit
        return self.intersect(
            ray_origin + ((hit.t + 0.01) * ray_direction), ray_direction
        )

    def paths(self) -> logic.Paths:
        paths = self.shape_a.paths()
        paths.paths += self.shape_b.paths().paths
        return paths.chop(0.01).filter(self)

    def filter(self, v: np.ndarray) -> (np.ndarray, bool):
        return v, self.contains(v, 0)


class Direction(Enum):
    Above = 0
    Below = 1


class Function(Shape):
    def __init__(
        self,
        func: Callable[[float, float], float],
        box: logic.Box,
        direction: Direction,
    ):
        super().__init__()
        self.func = func
        self.box = box
        self.direction = direction

    def bounding_box(self) -> logic.Box:
        return self.box

    def contains(self, v: np.ndarray, f: float) -> bool:
        if self.direction == Direction.Below:
            return v[2] < self.func(v[0], v[1])
        else:
            return v[2] > self.func(v[0], v[1])

    def intersect(
        self, ray_origin: np.ndarray, ray_direction: np.ndarray
    ) -> logic.Hit:
        step = 1 / 64
        sign = self.contains(ray_origin + (step * ray_direction), 0)
        for t in np.arange(step, 10, step):
            v = ray_origin + (t * ray_direction)
            if self.contains(v, 0) != sign and self.box.contains(v):
                return logic.Hit(self, t)
        return logic.NoHit

    def paths(self) -> logic.Paths:
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
        return logic.Paths(paths)
