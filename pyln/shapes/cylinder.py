import typing as ty

import warnings

import numba as nb
import numpy as np
from numba.core.errors import NumbaWarning

from .. import utility
from ..paths import Box, Paths
from ..shape import Shape, TransformedShape

warnings.simplefilter("ignore", category=NumbaWarning)


class Cylinder(Shape):
    def __init__(self, radius: float, z0: float, z1: float):
        super().__init__()
        self.radius = radius
        self.z0 = z0
        self.z1 = z1

    def bounding_box(self) -> Box:
        return Box(
            np.array([-self.radius, -self.radius, self.z0]),
            np.array([self.radius, self.radius, self.z1]),
        )

    def contains(self, v: np.ndarray, f: float) -> bool:
        if utility.vector_length(np.array([v[0], v[1], 0])) > self.radius + f:
            return False
        return self.z0 - f <= v[2] <= self.z1 + f

    def intersect(
        self, ray_origin: np.ndarray, ray_direction: np.ndarray
    ) -> float:
        return Cylinder._intersect(
            self.radius, self.z0, self.z1, ray_origin, ray_direction
        )

    @staticmethod
    @nb.njit(
        "float64(float64, float64, float64, float64[:], float64[:])",
        cache=True,
    )
    def _intersect(
        radius: float,
        z0: float,
        z1: float,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
    ) -> float:
        a = ray_direction[0] ** 2 + ray_direction[1] ** 2
        b = (
            2 * ray_origin[0] * ray_direction[0]
            + 2 * ray_origin[1] * ray_direction[1]
        )
        c = ray_origin[0] ** 2 + ray_origin[1] ** 2 - radius ** 2
        slope = b * b - 4 * a * c
        if slope < 0:
            return utility.INF
        slope = np.sqrt(slope)
        t0 = (-b + slope) / 2 * a
        t1 = (-b - slope) / 2 * a
        if t0 > t1:
            t0, t1 = t1, t0
        for root in [t0, t1]:
            z = ray_origin[2] + root * ray_direction[2]
            if root > 1e-6 and z0 < z < z1:
                return root
        return utility.INF

    def paths(self) -> Paths:
        result = []
        for a in range(0, 360, 10):
            a = np.deg2rad(a)
            x = self.radius * np.cos(a)
            y = self.radius * np.sin(a)
            result.append([[x, y, self.z0], [x, y, self.z1]])
        return Paths(result)


class OutlineCylinder(Cylinder):
    def __init__(
        self,
        eye: np.ndarray,
        up: np.ndarray,
        radius: float,
        z0: float,
        z1: float,
    ):
        super().__init__(radius, z0, z1)
        self.eye = eye
        self.up = up

    def paths(self) -> Paths:
        ab = []
        for z in [self.z0, self.z1]:
            center = np.array([0, 0, z])
            hypotenuse = utility.vector_length(center - self.eye)
            theta = np.arcsin(self.radius / hypotenuse)
            adj = self.radius / np.tan(theta)
            d = np.cos(theta) * adj
            w = utility.vector_normalize(center - self.eye)
            u = utility.vector_normalize(np.cross(w, self.up))
            c = self.eye + (w * d)
            ab.append(c + (self.radius * 1.01 * u))
            ab.append(c + (-self.radius * 1.01 * u))

        paths_0 = []
        paths_1 = []
        for a in range(360):
            a = np.deg2rad(a)
            x = self.radius * np.cos(a)
            y = self.radius * np.sin(a)
            paths_0.append([x, y, self.z0])
            paths_1.append([x, y, self.z1])
        paths = [paths_0, paths_1]
        a0, b0, a1, b1 = ab
        paths.append([[a0[0], a0[1], self.z0], [a1[0], a1[1], self.z1]])
        paths.append([[b0[0], b0[1], self.z0], [b1[0], b1[1], self.z1]])
        return Paths(paths)

    @staticmethod
    def TransformedCylinder(
        eye: np.ndarray,
        up: np.ndarray,
        v0: np.ndarray,
        v1: np.ndarray,
        radius: float,
    ) -> TransformedShape:
        d = v1 - v0
        a = np.arccos(np.dot(utility.vector_normalize(d), up))
        matrix = utility.vector_translate(v0)
        if a != 0:
            u = utility.vector_normalize(np.cross(d, up))
            matrix = utility.matrix_transform(
                [
                    (utility.Transform.Translate, v0),
                    (utility.Transform.Rotate, (u, a)),
                ],
            )
        outline_cylinder = OutlineCylinder(
            utility.matrix_mul_position_vector(
                utility.matrix_inverse(matrix), eye
            ),
            up,
            radius,
            0,
            utility.vector_length(d),
        )
        return TransformedShape(outline_cylinder, matrix)
