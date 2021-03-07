import typing as ty

from enum import Enum

import numba as nb
import numpy as np
from PIL import Image, ImageDraw

from . import utility


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

    def filter(self, v: np.ndarray) -> ty.Tuple[np.ndarray, bool]:
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
        result = []
        for v in self.path:
            result.append(utility.matrix_mul_position_vector(matrix, v))
        return result

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

    def to_svg(self) -> str:
        coords = []
        for v in self.path:
            coords.append(f"{v[0]},{v[1]}")
        points = " ".join(coords)
        return f'<polyline stroke="black" fill="none" points="{points}" />'


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

    def to_image(self, width, height):
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
                    fill="black",
                    width=3,
                )
        return im

    def write_to_png(self, file_path: str, width, height):
        self.to_image(width, height).save(file_path)

    def to_svg(self, width, height, background_color=None) -> str:
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
        lines += [path.to_svg() for path in self.paths]
        lines.append("</g></svg>")
        return "\n".join(lines)

    def write_to_svg(self, path: str, width, height, background_color=None):
        with open(path, "w") as file:
            file.write(self.to_svg(width, height, background_color))

    def write_to_txt(self, path: str):
        with open(path, "w") as file:
            file.write(str(self))


class Tree:
    def __init__(self, box, node):
        self.box = box
        self.root = node

    @staticmethod
    def from_shapes(shapes):
        box = Box.BoxForShapes(shapes)
        node = Node(shapes)
        node.split(0)
        return Tree(box, node)

    def intersect(self, ray_origin: np.ndarray, ray_direction: np.ndarray):
        tmin, tmax = self.box.intersect(ray_origin, ray_direction)
        if tmax < tmin or tmax <= 0:
            return NoHit
        return self.root.intersect(ray_origin, ray_direction, tmin, tmax)

    def show_tree(self, level=0):
        return " " * level + "Tree\n" + self.root.show_tree(level + 1)


class Node:
    def __init__(self, shapes):
        self.axis = None
        self.point = 0
        self.shapes = shapes
        self.left = None
        self.right = None

    def intersect(
        self, ray_origin: np.ndarray, ray_direction: np.ndarray, tmin, tmax
    ) -> Hit:
        if self.axis is None:
            return self.intersect_shapes(ray_origin, ray_direction)
        tsplit = (self.point - ray_origin[self.axis]) / ray_direction[self.axis]
        leftFirst = (ray_origin[self.axis] < self.point) or (
            ray_origin[self.axis] == self.point
            and ray_direction[self.axis] <= 0
        )
        if leftFirst:
            first = self.left
            second = self.right
        else:
            first = self.right
            second = self.left
        if tsplit > tmax or tsplit <= 0:
            return first.intersect(ray_origin, ray_direction, tmin, tmax)
        elif tsplit < tmin:
            return second.intersect(ray_origin, ray_direction, tmin, tmax)
        else:
            h1 = first.intersect(ray_origin, ray_direction, tmin, tsplit)
            if h1.t <= tsplit:
                return h1
            h2 = second.intersect(
                ray_origin, ray_direction, tsplit, min(tmax, h1.t)
            )
            if h1.t <= h2.t:
                return h1
            else:
                return h2

    def intersect_shapes(
        self, ray_origin: np.ndarray, ray_direction: np.ndarray
    ) -> Hit:
        hit = NoHit
        for shape in self.shapes:
            h = shape.intersect(ray_origin, ray_direction)
            if h.t < hit.t:
                hit = h
        return hit

    def partition_score(self, axis: int, point) -> int:
        left, right = 0, 0
        for shape in self.shapes:
            box = shape.bounding_box()
            l, r = box.partition(axis, point)
            if l:
                left += 1
            if r:
                right += 1
        return max(left, right)

    def partition(self, axis: int, point: np.ndarray):
        left = []
        right = []
        for shape in self.shapes:
            box = shape.bounding_box()
            l, r = box.partition(axis, point)
            if l:
                left.append(shape)
            if r:
                right.append(shape)
        return left, right

    def split(self, depth: int):
        if len(self.shapes) < 8:
            return
        values = []
        for shape in self.shapes:
            box = shape.bounding_box()
            values += [box.min, box.max]
        medians = np.median(np.array(values), axis=0)

        best = int(len(self.shapes) * 0.85)
        best_axis = None
        best_point = 0.0

        for i in range(3):
            s = self.partition_score(i, medians[i])
            if s < best:
                best = s
                best_axis = i
                best_point = medians[i]

        if best_axis is None:
            return

        l, r = self.partition(best_axis, best_point)
        self.axis = best_axis
        self.point = best_point
        self.left = Node(l)
        self.right = Node(r)
        self.left.split(depth + 1)
        self.right.split(depth + 1)
        self.shapes = None  # only needed at leaf nodes

    def show_tree(self, level=0):
        s = " " * level + "Node:"

        if self.axis is None:
            s += f"Shapes ({len(self.shapes)})\n"
            for shape in self.shapes:
                if getattr(shape, "show_tree", None):
                    s += shape.show_tree(level + 1)
            return s

        s += f"{self.axis} {self.point:.2f}\n"
        s += self.left.show_tree(level + 1)
        s += self.right.show_tree(level + 1)
        return s


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

    def rotate(self, axis: ty.Union[ty.List[float], np.ndarray], theta: float):
        if type(axis) == list:
            axis = np.array(axis, dtype=np.float64)
        return TransformedShape(
            self, utility.vector_rotate(axis, np.deg2rad(theta))
        )

    def rotate_x(self, theta: float):
        # TODO: document theta = degrees
        return TransformedShape(
            self,
            utility.vector_rotate(
                np.array([1, 0, 0], dtype=np.float64), np.deg2rad(theta)
            ),
        )

    def rotate_y(self, theta: float):
        return TransformedShape(
            self,
            utility.vector_rotate(
                np.array([0, 1, 0], dtype=np.float64), np.deg2rad(theta)
            ),
        )

    def rotate_z(self, theta: float):
        return TransformedShape(
            self,
            utility.vector_rotate(
                np.array([0, 0, 1], dtype=np.float64), np.deg2rad(theta)
            ),
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
