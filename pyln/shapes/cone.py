import typing as ty

import warnings

import numba as nb
import numpy as np
from numba.core.errors import NumbaWarning

from .. import utility
from ..paths import Box, Paths
from ..shape import Shape, TransformedShape

warnings.simplefilter("ignore", category=NumbaWarning)


class Cone(Shape):
    def __init__(self, radius: float, height: float):
        super().__init__()
        self.radius = radius
        self.height = height

    def compile(self):
        pass

    def bounding_box(self):
        return Box(
            np.array([-self.radius, -self.radius, 0]),
            np.array([self.radius, self.radius, self.height]),
        )

    def contains(self, v: np.ndarray, f: float) -> bool:
        return False

    def intersect(
        self, ray_origin: np.ndarray, ray_direction: np.ndarray
    ) -> float:
        return Cone._intersect(
            self.radius, self.height, ray_origin, ray_direction
        )

    @staticmethod
    @nb.njit(
        "float64(float64, float64, float64[:], float64[:])",
        cache=True,
    )
    def _intersect(
        radius: float,
        height: float,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
    ) -> float:
        k = radius / height
        k *= k
        a = (
            ray_direction[0] ** 2
            - ray_direction[1] ** 2
            - k * ray_direction[2] ** 2
        )
        b = 2 * (
            ray_direction[0] * ray_origin[0]
            + ray_direction[1] * ray_origin[1]
            - k * ray_direction[2] * (ray_origin[2] - height)
        )
        c = (
            ray_origin[0] ** 2
            + ray_origin[1] ** 2
            - k * (ray_origin[2] - height) ** 2
        )
        slope = b * b - 4 * a * c
        if slope <= 0:
            return utility.INF
        slope = np.sqrt(slope)
        t0 = (-b + slope) / 2 * a
        t1 = (-b - slope) / 2 * a
        if t0 > t1:
            t0, t1 = t1, t0
        for root in [t0, t1]:
            if root > 1e-6:
                p = ray_origin + (root * ray_direction)
                if 0 < p[2] < height:
                    return root
        return utility.INF

    def paths(self) -> Paths:
        result = []
        for a in range(0, 360, 30):
            a = np.deg2rad(a)
            x = self.radius * np.cos(a)
            y = self.radius * np.sin(a)
            result.append([[x, y, 0], [0, 0, self.height]])
        return Paths(result)


class OutlineCone(Cone):
    def __init__(
        self, eye: np.ndarray, up: np.ndarray, radius: float, height: float
    ):
        super().__init__(radius, height)
        self.eye = eye
        self.up = up

    def paths(self) -> Paths:
        center = np.zeros(3)
        hypotenuse = utility.vector_length(center - self.eye)
        theta = np.arcsin(self.radius / hypotenuse)
        adj = self.radius / np.tan(theta)
        d = np.cos(theta) * adj
        w = utility.vector_normalize(center - self.eye)
        u = utility.vector_normalize(np.cross(w, self.up))
        c0 = self.eye + (w * d)
        a0 = c0 + (self.radius * 1.01 * u)
        b0 = c0 + (-self.radius * 1.01 * u)
        paths = [
            [
                self.radius * np.cos(np.deg2rad(a)),
                self.radius * np.sin(np.deg2rad(a)),
                0,
            ]
            for a in range(360)
        ]
        paths.append([[a0[0], a0[1], 0], [0, 0, self.height]])
        paths.append([[b0[0], b0[1], 0], [0, 0, self.height]])
        return Paths(paths)


class TransformedOutlineCone(TransformedShape):
    def __init__(
        self,
        eye: np.ndarray,
        up: np.ndarray,
        v0: np.ndarray,
        v1: np.ndarray,
        radius: float,
    ):
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
        outline_cone = OutlineCone(
            utility.matrix_mul_position_vector(
                utility.matrix_inverse(matrix), eye
            ),
            up,
            radius,
            utility.vector_length(d),
        )
        super().__init__(outline_cone, matrix)
