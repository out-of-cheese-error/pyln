from enum import Enum

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

    def intersect(self, ray: utility.Ray) -> logic.Hit:
        return self.shape.intersect(ray.mul_matrix(self.inverse))

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

    def intersect(self, ray: utility.Ray) -> logic.Hit:
        hit_a = self.shape_a.intersect(ray)
        hit_b = self.shape_b.intersect(ray)
        hit = hit_a.min(hit_b)
        v = ray.position(hit.t)
        if (not hit.ok()) or self.contains(v, 0.0):
            return hit
        return self.intersect(
            utility.Ray(ray.position(hit.t + 0.01), ray.direction)
        )

    def paths(self) -> logic.Paths:
        paths = self.shape_a.paths()
        paths.paths += self.shape_b.paths().paths
        return paths.chop(0.01).filter(self)

    def filter(self, v: np.ndarray) -> (np.ndarray, bool):
        return v, self.contains(v, 0)
