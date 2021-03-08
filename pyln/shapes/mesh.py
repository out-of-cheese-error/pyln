import typing as ty

import datetime
import struct
from pathlib import Path

import numpy as np

from .. import __version__, utility
from ..paths import Box, Paths
from ..shape import Shape
from ..tree import Tree
from .cube import Cube
from .triangle import Triangle


class Mesh(Shape):
    def __init__(self, triangles: ty.List[Triangle]):
        super().__init__()
        self.triangles: ty.List[Triangle] = triangles
        self.tree: ty.Union[Tree, None] = None
        self.box: Box = Box.BoxForShapes(self.triangles)

    def compile(self):
        if self.tree is None:
            shapes = []
            for triangle in self.triangles:
                shapes.append(triangle)
            self.tree = Tree.from_shapes(shapes)

    def bounding_box(self) -> Box:
        return self.box

    def contains(self, v: np.ndarray, f: float) -> bool:
        return False

    def intersect(
        self, ray_origin: np.ndarray, ray_direction: np.ndarray
    ) -> float:
        return self.tree.intersect(ray_origin, ray_direction)

    def paths(self) -> Paths:
        result = []
        for t in self.triangles:
            result.extend(t.paths().paths)
        return Paths(result)

    def update_bounding_box(self):
        self.box = Box.BoxForShapes(self.triangles)

    def unit_cube(self):
        self.fit_inside(Box(np.zeros(3), np.ones(3)), np.zeros(3))
        self.move_to(np.zeros(3), np.array([0.5, 0.5, 0.5]))

    def move_to(self, position: np.ndarray, anchor: np.ndarray):
        matrix = utility.vector_translate(position - self.box.anchor(anchor))
        self.transform(matrix)

    def fit_inside(
        self, box: Box, anchor: ty.Union[ty.List[float], np.ndarray]
    ):
        anchor = np.asarray(anchor, dtype=np.float64)
        scale = np.amin(box.size() / self.bounding_box().size())
        extra = box.size() - (self.bounding_box().size() * scale)

        matrix = utility.matrix_transform(
            [
                (utility.Transform.Translate, box.min + (extra * anchor)),
                (utility.Transform.Scale, [scale, scale, scale]),
                (utility.Transform.Translate, -self.bounding_box().min),
            ],
        )
        self.transform(matrix)

    def transform(self, matrix: np.ndarray):
        for t in self.triangles:
            t.v1 = utility.matrix_mul_position_vector(matrix, t.v1)
            t.v2 = utility.matrix_mul_position_vector(matrix, t.v2)
            t.v3 = utility.matrix_mul_position_vector(matrix, t.v3)
            t.update_bounding_box()
        self.update_bounding_box()
        self.tree = None

    def show_tree(self, level=0):
        return " " * level + "Mesh\n" + self.tree.show_tree(level + 1)

    def voxelize(self, size: float) -> ty.List[Cube]:
        vector_set = set()
        for z in np.arange(self.box.min[2], self.box.max[2] + 1, size):
            plane = Plane(
                np.array([0, 0, z], dtype=np.float64),
                np.array([0, 0, 1], dtype=np.float64),
            )
            paths = plane.intersect_mesh(self)
            for path in paths.paths:
                for vector in path.path:
                    vector_set.add(tuple(np.floor(vector / size + 0.5) * size))
        vector_set = [np.array(v) for v in vector_set]
        return [Cube(v - (size / 2), v + (size / 2)) for v in vector_set]

    @classmethod
    def from_obj(cls, path: str) -> "Mesh":
        def parse_index(value: str, length: int) -> int:
            vi = value.split("/")
            n = int(vi[0])
            if n < 0:
                n += length
            return n

        with open(path) as file:
            vs = []
            triangles = []
            for index, line in enumerate(file):
                fields = line.split()
                if len(fields) == 0:
                    continue
                keyword = fields[0]
                args = fields[1:]
                if keyword == "v":
                    vs.append(np.array(args, dtype=np.float64))
                elif keyword == "f":
                    fvs = [parse_index(i, len(vs)) for i in args]
                    for i in range(1, len(fvs) - 1):
                        i1, i2, i3 = fvs[0], fvs[i], fvs[i + 1]
                        t = Triangle(vs[i1 - 1], vs[i2 - 1], vs[i3 - 1])
                        triangles.append(t)
        return Mesh(triangles)

    @classmethod
    def from_stl(cls, path: str) -> "Mesh":
        vertexes = []
        with open(path) as file:
            for line in file:
                fields = line.split()
                if len(fields) == 4 and fields[0] == "vertex":
                    vertexes.append(
                        np.array([float(f) for f in fields[1 : 1 + 3]])
                    )
        triangles = []
        for i in range(0, len(vertexes), 3):
            triangles.append(
                Triangle(vertexes[i], vertexes[i + 1], vertexes[i + 2])
            )
        return Mesh(triangles)

    @classmethod
    def from_binary_stl(cls, path: str) -> "Mesh":
        stl = Stl.from_file(path)
        triangles = []
        for v0, v1, v2 in zip(stl.data["v0"], stl.data["v1"], stl.data["v2"]):
            triangles.append(
                Triangle(
                    v0.astype(np.float64),
                    v1.astype(np.float64),
                    v2.astype(np.float64),
                )
            )
        return Mesh(triangles)

    def write_binary_stl(self, path: str):
        # Format the header
        header = (
            f"pyln ({__version__}) {datetime.datetime.now()} {Path(path).stem}"
        )
        # Make it exactly 80 characters
        header = header[:80].ljust(80, " ")
        data = np.zeros(len(self.triangles), dtype=Stl.dtype)
        data["v0"] = [t.v1.astype(np.float32) for t in self.triangles]
        data["v1"] = [t.v2.astype(np.float32) for t in self.triangles]
        data["v2"] = [t.v3.astype(np.float32) for t in self.triangles]
        Stl(header, data).to_file(path)


class Stl:
    # from https://w.wol.ph/2014/10/11/reading-writing-binary-stl-files-numpy/
    dtype = np.dtype(
        [
            ("normals", np.float32, (3,)),
            ("v0", np.float32, (3,)),
            ("v1", np.float32, (3,)),
            ("v2", np.float32, (3,)),
            ("attr", "u2", (1,)),
        ]
    )

    def __init__(self, header, data):
        self.header = header
        self.data = data

    @classmethod
    def from_file(cls, filename, mode="rb"):
        with open(filename, mode) as fh:
            header = fh.read(80)
            (size,) = struct.unpack("@i", fh.read(4))
            data = np.fromfile(fh, dtype=cls.dtype, count=size)
            return Stl(header, data)

    def to_file(self, filename, mode="wb"):
        with open(filename, mode) as fh:
            header = self.header
            if type(header) == str:
                header = bytes(header, "ascii", "replace")
            fh.write(header)
            fh.write(struct.pack("@i", self.data.size))
            self.data.tofile(fh)


class Plane:
    def __init__(
        self,
        point: ty.Union[ty.List[float], np.ndarray],
        normal: ty.Union[ty.List[float], np.ndarray],
    ):
        self.point: np.ndarray = np.asarray(point, dtype=np.float64)
        self.normal: np.ndarray = np.asarray(normal, dtype=np.float64)

    def intersect_segment(
        self, v0: np.ndarray, v1: np.ndarray
    ) -> ty.Tuple[ty.Union[None, np.ndarray], bool]:
        u = v1 - v0
        w = v0 - self.point
        d = self.normal.dot(u)
        if -utility.EPS < d < utility.EPS:
            return None, False
        t = -self.normal.dot(w) / d
        if t < 0 or t > 1:
            return None, False
        return v0 + u * t, True

    def intersect_triangle(
        self, triangle: Triangle
    ) -> ty.Tuple[ty.Union[None, np.ndarray], ty.Union[None, np.ndarray], bool]:
        v1, ok1 = self.intersect_segment(triangle.v1, triangle.v2)
        v2, ok2 = self.intersect_segment(triangle.v2, triangle.v3)
        v3, ok3 = self.intersect_segment(triangle.v3, triangle.v1)
        if ok1 and ok2:
            return v1, v2, True
        if ok1 and ok3:
            return v1, v3, True
        if ok2 and ok3:
            return v2, v3, True
        return None, None, False

    def intersect_mesh(self, mesh: Mesh) -> Paths:
        result = []
        for triangle in mesh.triangles:
            v1, v2, ok = self.intersect_triangle(triangle)
            if ok:
                result.append([v1, v2])
        return Paths(result)
