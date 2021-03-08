import typing as ty

from enum import Enum

import numpy as np

from . import utility
from .paths import Box, Filter, Hit, Paths


class Shape:
    def __init__(self):
        pass

    def compile(self):
        pass

    def bounding_box(self) -> Box:
        pass

    def contains(self, v: np.ndarray, f: float) -> bool:
        pass

    def intersect(
        self, ray_origin: np.ndarray, ray_direction: np.ndarray
    ) -> Hit:
        pass

    def paths(self) -> Paths:
        pass

    def __sub__(self, other):
        return BooleanShape(Op.Difference, self, other)

    def __mul__(self, other):
        return BooleanShape(Op.Intersection, self, other)

    def rotate(
        self,
        axis: ty.Union[ty.List[float], np.ndarray],
        theta: float,
        radians=False,
    ):
        if not radians:
            theta = np.deg2rad(theta)
        if type(axis) == list:
            axis = np.array(axis, dtype=np.float64)
        return TransformedShape(self, utility.vector_rotate(axis, theta))

    def rotate_x(self, theta: float, radians=False):
        if not radians:
            theta = np.deg2rad(theta)
        return TransformedShape(
            self,
            utility.vector_rotate(np.array([1, 0, 0], dtype=np.float64), theta),
        )

    def rotate_y(self, theta: float, radians=False):
        if not radians:
            theta = np.deg2rad(theta)
        return TransformedShape(
            self,
            utility.vector_rotate(np.array([0, 1, 0], dtype=np.float64), theta),
        )

    def rotate_z(self, theta: float, radians=False):
        if not radians:
            theta = np.deg2rad(theta)
        return TransformedShape(
            self,
            utility.vector_rotate(np.array([0, 0, 1], dtype=np.float64), theta),
        )

    def scale(self, scale: np.ndarray):
        if type(scale) == list:
            scale = np.array(scale, dtype=np.float64)
        return TransformedShape(self, utility.vector_scale(scale))

    def translate(self, translate: np.ndarray):
        if type(translate) == list:
            translate = np.array(translate, dtype=np.float64)
        return TransformedShape(self, utility.vector_translate(translate))


class TransformedShape(Shape):
    def __init__(self, shape: Shape, matrix: np.ndarray):
        super().__init__()
        self.shape = shape
        self.matrix = matrix
        self.inverse = utility.matrix_inverse(matrix)

    def compile(self):
        self.shape.compile()

    def bounding_box(self) -> Box:
        box = self.shape.bounding_box()
        min_box, max_box = utility.matrix_mul_box(self.matrix, box.min, box.max)
        return Box(min_box, max_box)

    def contains(self, v: np.ndarray, f: float) -> bool:
        return self.shape.contains(
            utility.matrix_mul_position_vector(self.inverse, v), f
        )

    def intersect(
        self, ray_origin: np.ndarray, ray_direction: np.ndarray
    ) -> Hit:
        origin, direction = utility.matrix_mul_ray(
            self.inverse, ray_origin, ray_direction
        )
        return self.shape.intersect(origin, direction)

    def paths(self) -> Paths:
        return self.shape.paths().transform(self.matrix)


class Op(Enum):
    Intersection = 0
    Difference = 1


class BooleanShape(Shape, Filter):
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

    def bounding_box(self) -> Box:
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
    ) -> Hit:
        hit_a = self.shape_a.intersect(ray_origin, ray_direction)
        hit_b = self.shape_b.intersect(ray_origin, ray_direction)
        hit = hit_a.min(hit_b)
        v = ray_origin + (hit.t * ray_direction)
        if (not hit.ok()) or self.contains(v, 0.0):
            return hit
        return self.intersect(
            ray_origin + ((hit.t + 0.01) * ray_direction), ray_direction
        )

    def paths(self) -> Paths:
        paths = Paths(self.shape_a.paths().paths + self.shape_b.paths().paths)
        return paths.chop(0.01).filter(self)

    def filter(self, v: np.ndarray) -> ty.Tuple[np.ndarray, bool]:
        return v, self.contains(v, 0.0)
