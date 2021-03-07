import typing as ty

import numba as nb
import numpy as np

from .. import logic, utility


class Sphere(logic.Shape):
    def __init__(self, center=None, radius=1.0, texture=1):
        super().__init__()
        self.center = np.zeros(3) if center is None else center
        self.radius = radius
        radius_vec = np.array([radius, radius, radius])
        self.box = logic.Box(self.center - radius_vec, self.center + radius_vec)
        self.texture = texture

    def compile(self):
        pass

    def bounding_box(self) -> logic.Box:
        return self.box

    def contains(self, v: np.ndarray, f) -> bool:
        return utility.vector_length(v - self.center) <= self.radius + f

    def intersect(
        self, ray_origin: np.ndarray, ray_direction: np.ndarray
    ) -> logic.Hit:
        ok, root = Sphere._intersect(
            self.radius, self.center, ray_origin, ray_direction
        )
        if ok:
            return logic.Hit(self, root)
        else:
            return logic.NoHit

    @staticmethod
    @nb.njit(
        "Tuple((boolean, float64))(float64, float64[:], float64[:], float64[:])",
        cache=True,
    )
    def _intersect(
        radius: float,
        center: np.ndarray,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
    ) -> ty.Tuple[bool, float]:
        to = ray_origin - center
        b = np.dot(to, ray_direction)
        c = np.dot(to, to) - radius * radius
        d = b * b - c

        if d > 0:
            d = np.sqrt(d)
            t1 = -b - d
            if t1 > 1e-2:
                return True, t1

            t2 = -b + d
            if t2 > 1e-2:
                return True, t2

        return False, 0

    def paths(self) -> logic.Paths:
        if self.texture == 1:
            return logic.Paths(Sphere.paths_1(self.radius, self.center))
        elif self.texture == 2:
            return logic.Paths(Sphere.paths_2(self.radius, self.center))
        elif self.texture == 3:
            return logic.Paths(Sphere.paths_3(self.radius, self.center))
        elif self.texture == 4:
            return logic.Paths(Sphere.paths_4(self.radius, self.center))

    @staticmethod
    def paths_1(
        radius: float, center: np.ndarray
    ) -> ty.List[ty.List[np.ndarray]]:
        # Grid pattern
        paths = []
        n = 5
        o = 10
        for lat in range(-90 + o, 91 - o, n):
            paths.append(
                [
                    Sphere.latlng_to_xyz(lat, lng, radius) + center
                    for lng in range(0, 361)
                ]
            )
        for lng in range(0, 361, n):
            paths.append(
                [
                    Sphere.latlng_to_xyz(lat, lng, radius) + center
                    for lat in np.arange(-90 + o, 91 - o)
                ]
            )
        return paths

    @staticmethod
    def paths_2(radius: float, center: np.ndarray) -> ty.List[logic.Path]:
        # Criss-cross pattern
        paths = []
        equator = logic.Path(
            [Sphere.latlng_to_xyz(0, lng, radius) for lng in range(360)]
        )
        for i in range(100):
            matrix = np.identity(4)
            for j in range(3):
                v = utility.random_unit_vector()
                matrix = utility.matrix_mul_matrix(
                    utility.vector_rotate(v, np.random.random() * 2 * np.pi),
                    matrix,
                )
                matrix = utility.matrix_mul_matrix(
                    utility.vector_translate(center), matrix
                )
                paths.append(equator.transform(matrix))
        return paths

    @staticmethod
    @nb.njit("float64[:,:,:](float64, float64[:])")
    def paths_3(radius: float, center: np.ndarray) -> np.ndarray:
        paths = np.zeros((20000, 2, 3))
        for i in range(20000):
            v = utility.random_unit_vector() * radius + center
            paths[i, 0] = v
            paths[i, 1] = v.copy()
        return paths

    @staticmethod
    def paths_4(
        radius: float, center: np.ndarray
    ) -> ty.List[ty.List[np.ndarray]]:
        # Criss-cross with circles
        paths = []
        seen = []
        radii = []
        for i in range(140):
            while True:
                v = utility.random_unit_vector()
                m = np.random.random() * 0.25 + 0.05
                ok = True
                for s in range(len(seen)):
                    threshold = m + radii[s] + 0.02
                    if utility.vector_length(seen[s] - v) < threshold:
                        ok = False
                        break
                if ok:
                    seen.append(v)
                    radii.append(m)
                    break
            p = utility.vector_normalize(
                np.cross(v, utility.random_unit_vector())
            )
            q = utility.vector_normalize(np.cross(p, v))
            n = np.random.randint(0, 4) + 1
            for k in range(n):
                path = []
                for j in range(0, 360, 5):
                    a = np.deg2rad(j)
                    path.append(
                        utility.vector_normalize(
                            v + p * np.cos(a) * m + q * np.sin(a) * m
                        )
                        * radius
                        + center
                    )
                paths.append(path)
                m += 0.75
        return paths

    @staticmethod
    @nb.njit("float64[:](float64, float64, float64)", cache=True)
    def latlng_to_xyz(lat, lng, radius) -> np.ndarray:
        lat, lng = np.deg2rad(lat), np.deg2rad(lng)
        x = radius * np.cos(lat) * np.cos(lng)
        y = radius * np.cos(lat) * np.sin(lng)
        z = radius * np.sin(lat)
        return np.array([x, y, z])


class OutlineSphere(Sphere):
    def __init__(
        self, eye: np.ndarray, up: np.ndarray, center: np.ndarray, radius: float
    ):
        super().__init__(center, radius)
        self.eye = eye
        self.up = up

    def paths(self) -> logic.Paths:
        hyp = utility.vector_length(self.center - self.eye)
        theta = np.arcsin(self.radius / hyp)
        adj = self.radius / np.tan(theta)
        d = np.cos(theta) * adj
        r = np.sin(theta) * adj

        w = utility.vector_normalize(self.center - self.eye)
        u = utility.vector_normalize(np.cross(w, self.up))
        v = utility.vector_normalize(np.cross(w, u))
        c = self.eye + (w * d)
        path = []
        for a in range(360):
            a = np.deg2rad(a)
            path.append(c + u * np.cos(a) * r + v * np.sin(a) * r)

        return logic.Paths([path])
