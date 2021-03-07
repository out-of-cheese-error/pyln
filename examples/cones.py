import numpy as np
import pyln


class PoissonGrid:
    def __init__(self, radius: float):
        self.radius = radius
        self.size = radius / np.sqrt(2)
        self.cells = dict()

    def normalize(self, vector: np.ndarray):
        return np.array(
            [
                np.floor(vector[0] / self.size),
                np.floor(vector[1] / self.size),
                0,
            ]
        ).astype(np.int32)

    def insert(self, vector: np.ndarray) -> bool:
        n_vector = self.normalize(vector)
        for i in np.arange(n_vector[0] - 2, n_vector[0] + 3):
            for j in np.arange(n_vector[1] - 2, n_vector[1] + 3):
                if (i, j, 0) in self.cells:
                    m = self.cells[(i, j, 0)]
                    if (
                        np.sqrt((m[0] - vector[0]) ** 2 + m[1] - vector[1] ** 2)
                        < self.radius
                    ):
                        return False
        self.cells[tuple(n_vector)] = vector
        return True


def poisson_disc(x1, y1, x2, y2, radius, num):
    result = []
    x = x1 + (x2 - x1) / 2
    y = y1 + (y2 - y1) / 2
    v = np.array([x, y, 0])
    active = [v]
    grid = PoissonGrid(radius)
    grid.insert(v)
    while len(active):
        index = np.random.randint(0, len(active))
        point = active[index]
        ok = False
        for i in range(num):
            a = np.random.random() * 2 * np.pi
            d = np.random.random() * radius + radius
            x = point[0] + np.cos(a) * d
            y = point[0] + np.sin(a) * d
            if x < x1 or y < y1 or x > x2 or y > y2:
                continue
            v = np.array([x, y, 0])
            if not grid.insert(v):
                continue
            result.append(v)
            active.append(v)
            ok = True
            break
        if not ok:
            active = active[:index] + active[index + 1 :]
    return result


class ConeTree(pyln.TransformedOutlineCone):
    def __init__(
        self,
        eye: np.ndarray,
        up: np.ndarray,
        v0: np.ndarray,
        v1: np.ndarray,
        radius: float,
    ):
        super().__init__(eye, up, v0, v1, radius)
        self.v0 = v0
        self.v1 = v1

    def paths(self) -> pyln.Paths:
        paths = self.paths()
        for i in range(128):
            p = np.power(np.random.random(), 1.5) * 0.5 + 0.5
            c = self.v0 + ((self.v1 - self.v0) * p)
            a = np.random.random() * 2 * np.pi
            e = (
                c
                + pyln.utility.vector_normalize(
                    np.array([np.cos(a), np.sin(a), -2.75])
                )
                * (1 - p)
                * 8
            )
            paths.paths.append(pyln.logic.Path([c, e]))
        return paths


def main():
    # define camera parameters
    eye = np.array([0, 0, 0])  # camera position
    center = np.array([0.5, 0, 8])  # camera looks at
    up = np.array([0, 0, 1])  # up direction

    # create a scene and add a single cube
    scene = pyln.Scene()
    n = 9.0
    points = poisson_disc(-n, -n, n, n, 2, 32)
    for p in points:
        z = np.random.random() * 5 + 20
        v0 = np.array([p[0], p[1], 0])
        v1 = np.array([p[0], p[1], z])
        if pyln.utility.vector_length(v0 - eye) < 1:
            continue
        scene.add(ConeTree(eye, up, v0, v1, z / 64))
    print("Made scene")
    # define rendering parameters
    width = 500  # rendered width
    height = 500  # rendered height
    fovy = 90.0  # vertical field of view, degrees
    znear = 0.1  # near z plane
    zfar = 100.0  # far z plane
    step = 0.01  # how finely to chop the paths for visibility testing

    # compute 2D paths that depict the 3D scene
    paths = scene.render(
        eye, center, up, width, height, fovy, znear, zfar, step
    )

    # save results
    paths.write_to_svg("out.svg", width, height)


if __name__ == "__main__":
    main()
