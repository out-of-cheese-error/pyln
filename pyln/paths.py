import typing as ty

import warnings

import numba as nb
import numpy as np
from numba.core.errors import NumbaWarning
from PIL import Image, ImageDraw

from . import utility

warnings.simplefilter("ignore", category=NumbaWarning)


class Box:
    def __init__(
        self,
        min_box: ty.Union[ty.List[float], np.ndarray],
        max_box: ty.Union[ty.List[float], np.ndarray],
    ):
        self.box = np.array([min_box, max_box], dtype=np.float64)

    @classmethod
    def BoxForShapes(cls, shapes: ty.List["Shape"]) -> "Box":
        assert len(shapes)
        box = shapes[0].bounding_box()
        for shape in shapes:
            box = box.extend(shape.bounding_box())
        return box

    @classmethod
    def BoxForVectors(cls, vectors: ty.List[np.ndarray]) -> "Box":
        assert len(vectors)
        return Box(np.min(vectors, axis=0), np.max(vectors, axis=0))

    @property
    def min(self) -> np.ndarray:
        return self.box[0]

    @property
    def max(self) -> np.ndarray:
        return self.box[1]

    def anchor(self, anchor: np.ndarray) -> np.ndarray:
        return self.min + (self.size() * anchor)

    def center(self) -> np.ndarray:
        return self.anchor(np.array([0.5, 0.5, 0.5]))

    def size(self) -> np.ndarray:
        return self.max - self.min

    def contains(self, b: np.ndarray) -> bool:
        return all(self.min[i] <= b[i] <= self.max[i] for i in range(3))

    def extend(self, other):
        return Box(
            np.minimum(self.min, other.min), np.maximum(self.max, other.max)
        )

    def intersect(
        self, ray_origin: np.ndarray, ray_direction: np.ndarray
    ) -> ty.Tuple[float, float]:
        return Box._intersect(self.min, self.max, ray_origin, ray_direction)

    @staticmethod
    @nb.njit(
        "Tuple((float64, float64))(float64[:], float64[:], float64[:], float64[:])",
        cache=True,
    )
    def _intersect(
        min_box: np.ndarray,
        max_box: np.ndarray,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
    ) -> ty.Tuple[float, float]:
        difference_min = (min_box - ray_origin) / ray_direction
        difference_max = (max_box - ray_origin) / ray_direction
        t1 = np.amax(np.minimum(difference_min, difference_max))
        t2 = np.amin(np.maximum(difference_min, difference_max))
        return t1, t2

    def partition(self, axis: int, point) -> ty.Tuple[bool, bool]:
        return self.min[axis] <= point, self.max[axis] >= point

    def __str__(self):
        return f"Box: {self.min} {self.max}"

    def __repr__(self):
        return f"Box: {self.min} {self.max}"


class Hit:
    def __init__(self, shape, t: float):
        self.shape = shape
        self.t = t

    def ok(self) -> bool:
        return self.t < utility.INF

    def min(self, other):
        if self.t <= other.t:
            return self
        return other

    def max(self, other):
        if self.t > other.t:
            return self
        return other


NoHit = Hit(None, utility.INF)


class Filter:
    def __init__(self):
        pass

    def filter(self, v: np.ndarray) -> ty.Tuple[ty.Optional[np.ndarray], bool]:
        return None, False


class ClipFilter(Filter):
    def __init__(self, matrix: np.ndarray, eye: np.ndarray, scene):
        super().__init__()
        self.matrix = matrix
        self.eye = eye
        self.scene = scene
        self.clip_box = Box([-1, -1, -1], [1, 1, 1])

    def filter(self, v: np.ndarray) -> ty.Tuple[np.ndarray, bool]:
        w = utility.matrix_mul_position_vector(self.matrix, v)
        if not self.scene.visible(self.eye, v):
            return w, False
        if not self.clip_box.contains(w):
            return w, False
        return w, True


class Path:
    def __init__(self, path: ty.Union[np.ndarray, ty.List[np.ndarray]] = None):
        self.path: ty.List[np.ndarray] = (
            [np.array(p) for p in path] if path is not None else []
        )

    def bounding_box(self) -> Box:
        box = Box(self.path[0], self.path[0])
        for point in self.path:
            box = box.extend(Box(point, point))
        return box

    def transform(self, matrix: np.ndarray):
        return [
            utility.matrix_mul_position_vector(matrix, v) for v in self.path
        ]

    def chop(self, step):
        result = []
        for i in range(len(self.path) - 1):
            a = self.path[i]
            b = self.path[i + 1]
            v = b - a
            length = utility.vector_length(v)
            if i == 0:
                result.append(a)
            d = step
            while d < length:
                result.append(a + (v * (d / length)))
                d += step
            result.append(b)
        return result

    def filter(self, clip_filter: Filter):
        result = []
        path = []
        for v in self.path:
            v, ok = clip_filter.filter(v)
            if ok:
                path.append(v)
            else:
                if len(path) > 1:
                    result.append(path)
                path = []
        if len(path) > 1:
            result.append(path)
        return result

    def simplify(self, threshold):
        path_length = len(self.path)
        if path_length < 3:
            return self.path
        a = self.path[0]
        b = self.path[path_length - 1]
        index = -1
        distance = 0.0
        for i in range(1, path_length - 1):
            d = utility.segment_distance(self.path[i], a, b)
            if d > distance:
                index = i
                distance = d

        if distance > threshold:
            r1 = Path(self.path[: index + 1]).simplify(threshold)
            r2 = Path(self.path[index:]).simplify(threshold)
            return r1[: len(r1) - 1] + r2
        else:
            return [a, b]

    def to_string(self) -> str:
        result = ""
        for v in self.path:
            result += f"{v[0]:.4f},{v[1]:.4f},{v[2]:.4f};"
        return result

    def to_svg(self, color="black") -> str:
        coords = []
        for v in self.path:
            coords.append(f"{v[0]},{v[1]}")
        points = " ".join(coords)
        return f'<polyline stroke="{color}" fill="none" points="{points}" />'


class Paths:
    def __init__(
        self,
        paths: ty.Union[
            np.ndarray,
            ty.List[
                ty.Union[
                    np.ndarray,
                    Path,
                    ty.List[np.ndarray],
                    ty.List[ty.List[float]],
                ]
            ],
        ] = None,
    ):
        if paths is None or not len(paths):
            self.paths = []
        else:
            self.paths = [Path(p) if type(p) != Path else p for p in paths]

    def bounding_box(self) -> Box:
        box = self.paths[0].bounding_box()
        for path in self.paths:
            box = box.extend(path.bounding_box())
        return box

    def transform(self, matrix: np.ndarray) -> "Paths":
        return Paths([path.transform(matrix) for path in self.paths])

    def chop(self, step) -> "Paths":
        return Paths([path.chop(step) for path in self.paths])

    def filter(self, f: Filter):
        result = []
        for path in self.paths:
            result.extend(path.filter(f))
        return Paths(result)

    def simplify(self, threshold):
        return Paths([path.simplify(threshold) for path in self.paths])

    def __str__(self):
        return "\n".join(path.to_string() for path in self.paths)

    def to_image(self, width, height, fill="black", linewidth=3):
        canvas = (int(width), int(height))
        im = Image.new("RGBA", canvas, (255, 255, 255, 255))
        draw = ImageDraw.Draw(im)
        for ps in self.paths:
            for i, v1 in enumerate(ps.path):
                if i >= len(ps.path) - 1:
                    break
                v2 = ps.path[i + 1]
                draw.line(
                    (v1[0], height - v1[1], v2[0], height - v2[1]),
                    fill=fill,
                    width=linewidth,
                )
        return im

    def write_to_png(
        self, file_path: str, width, height, fill="black", linewidth=3
    ):
        self.to_image(width, height, fill, linewidth).save(file_path)

    def to_svg(
        self, width, height, line_color="black", background_color=None
    ) -> str:
        if background_color is None:
            bg = ""
        else:
            bg = f' style="background-color:{background_color}"'

        lines = [
            f'<svg width="{width}" height="{height}"{bg} '
            f'version="1.1" baseProfile="full" '
            f'xmlns="http://www.w3.org/2000/svg">',
            f'<g transform="translate(0,{height}) scale(1,-1)">',
        ]
        lines += [path.to_svg(line_color) for path in self.paths]
        lines.append("</g></svg>")
        return "\n".join(lines)

    def write_to_svg(
        self,
        path: str,
        width,
        height,
        line_color="black",
        background_color=None,
    ):
        with open(path, "w") as file:
            file.write(self.to_svg(width, height, line_color, background_color))

    def write_to_txt(self, path: str):
        with open(path, "w") as file:
            file.write(str(self))
