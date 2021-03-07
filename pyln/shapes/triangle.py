import numpy as np

from .. import Shape, logic, utility


class Triangle(Shape):
    def __init__(self, v1=None, v2=None, v3=None):
        super().__init__()
        self.v1 = np.zeros(3) if v1 is None else v1
        self.v2 = np.zeros(3) if v2 is None else v2
        self.v3 = np.zeros(3) if v3 is None else v3
        self.box = None
        self.update_bounding_box()

    def compile(self):
        pass

    def bounding_box(self) -> logic.Box:
        return self.box

    def update_bounding_box(self):
        min_v = np.minimum(np.minimum(self.v1, self.v2), self.v3)
        max_v = np.maximum(np.maximum(self.v1, self.v2), self.v3)
        self.box = logic.Box(min_v, max_v)

    def contains(self, v: np.ndarray, f: float) -> bool:
        return False

    def intersect(self, r: utility.Ray) -> logic.Hit:
        e1 = self.v2 - self.v1
        e2 = self.v3 - self.v1
        px = r.direction[1] * e2[2] - r.direction[2] * e2[1]
        py = r.direction[2] * e2[0] - r.direction[0] * e2[2]
        pz = r.direction[0] * e2[1] - r.direction[1] * e2[0]
        det = e1[0] * px + e1[1] * py + e1[2] * pz

        if -utility.EPS < det < utility.EPS:
            return logic.NoHit

        inv = 1 / det
        t = r.origin - self.v1
        u = (t[0] * px + t[1] * py + t[2] * pz) * inv

        if u < 0.0 or u > 1.0:
            return logic.NoHit

        qx = t[1] * e1[2] - t[2] * e1[1]
        qy = t[2] * e1[0] - t[0] * e1[2]
        qz = t[0] * e1[1] - t[1] * e1[0]
        v = (
            r.direction[0] * qx + r.direction[1] * qy + r.direction[2] * qz
        ) * inv

        if v < 0.0 or (u + v) > 1.0:
            return logic.NoHit

        d = (e2[0] * qx + e2[1] * qy + e2[2] * qz) * inv
        if d < utility.EPS:
            return logic.NoHit
        return logic.Hit(self, d)

    def paths(self) -> logic.Paths:
        return logic.Paths(
            [[self.v1, self.v2], [self.v2, self.v3], [self.v3, self.v1]]
        )

    def show_tree(self, level):
        result = level * " "
        for vector in [self.v1, self.v2, self.v3]:
            result += ",".join([f"{v:5.2f}" for v in vector]) + " "
        return f"{result}\n"

    def __str__(self):
        return f"V1: {self.v1}, V2: {self.v2}, V3: {self.v3}"
