import numpy as np

from .. import Shape, logic, utility


class Sphere(Shape):
    def __init__(self, center=None, radius=1.0):
        super().__init__()
        self.center = np.zeros(3) if center is None else center
        self.radius = radius
        radius_vec = np.array([radius, radius, radius])
        self.box = logic.Box(self.center - radius_vec, self.center + radius_vec)

    def compile(self):
        pass

    def bounding_box(self) -> logic.Box:
        return self.box

    def contains(self, v: np.ndarray, f) -> bool:
        return utility.vector_length(v - self.center) <= self.radius + f

    def intersect(self, r: utility.Ray) -> logic.Hit:
        radius = self.radius
        to = r.origin - self.center
        b = np.dot(to, r.direction)
        c = np.dot(to, to) - radius * radius
        d = b * b - c

        if d > 0:
            d = np.sqrt(d)
            t1 = -b - d
            if t1 > 1e-2:
                return logic.Hit(self, t1)

            t2 = -b + d
            if t2 > 1e-2:
                return logic.Hit(self, t2)

        return logic.NoHit

    def paths(self) -> logic.Paths:
        paths = []
        n = 10
        o = 10
        for lat in range(-90 + o, 91 - o, n):
            paths.append(
                [
                    Sphere.latlng_to_xyz(lat, lng, self.radius) + self.center
                    for lng in range(0, 361)
                ]
            )
        for lng in range(0, 361, n):
            paths.append(
                [
                    Sphere.latlng_to_xyz(lat, lng, self.radius) + self.center
                    for lat in np.arange(-90 + o, 91 - o)
                ]
            )
        return logic.Paths(paths)

    @staticmethod
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
        hypotenuse = utility.vector_length(self.center - self.eye)
        theta = np.arcsin(self.radius / hypotenuse)
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
