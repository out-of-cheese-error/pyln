from typing import List

import numba as nb
import numpy as np
from PIL import Image, ImageDraw

from . import utility


class Box:
    def __init__(self, min_box, max_box):
        self.box = np.array([min_box, max_box], dtype=np.float64)

    @staticmethod
    def BoxForShapes(shapes: ["Shape"]):
        assert len(shapes)
        box = shapes[0].bounding_box()
        for shape in shapes:
            box = box.extend(shape.bounding_box())
        return box

    @staticmethod
    def BoxForVectors(vectors: [np.ndarray]):
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

    def intersect(self, ray_origin: np.ndarray, ray_direction: np.ndarray):
        return Box._intersect(self.min, self.max, ray_origin, ray_direction)

    @staticmethod
    @nb.njit(cache=True)
    def _intersect(
        min_box: np.ndarray,
        max_box: np.ndarray,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
    ) -> (float, float):
        difference_min = (min_box - ray_origin) / ray_direction
        difference_max = (max_box - ray_origin) / ray_direction
        t1 = np.amax(np.minimum(difference_min, difference_max))
        t2 = np.amin(np.maximum(difference_min, difference_max))
        return t1, t2

    def partition(self, axis: int, point) -> (bool, bool):
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

    def filter(self, v: np.ndarray) -> (np.ndarray, bool):
        return None, None


class ClipFilter(Filter):
    def __init__(self, matrix: np.ndarray, eye: np.ndarray, scene):
        super().__init__()
        self.matrix = matrix
        self.eye = eye
        self.scene = scene
        self.clip_box = Box([-1, -1, -1], [1, 1, 1])

    def filter(self, v: np.ndarray) -> (np.ndarray, bool):
        w = utility.matrix_mul_position_vector(self.matrix, v, True)
        if not self.scene.visible(self.eye, v):
            return w, False
        if not self.clip_box.contains(w):
            return w, False
        return w, True


class Path:
    def __init__(self, path=None):
        self.path: List[np.ndarray] = (
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
    def __init__(self, paths=None):
        if paths is None or not len(paths):
            self.paths = []
        elif type(paths[0]) == Path:
            self.paths = paths
        else:
            self.paths = [Path(p) for p in paths]

    def bounding_box(self) -> Box:
        box = self.paths[0].bounding_box()
        for path in self.paths:
            box = box.extend(path.bounding_box())
        return box

    def transform(self, matrix: np.ndarray):
        return Paths([path.transform(matrix) for path in self.paths])

    def chop(self, step):
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

    def write_to_png(self, file_path: str, width, height):
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
                    fill=0,
                    width=3,
                )
        im.save(file_path)

    def to_svg(self, width, height) -> str:
        lines = [
            f'<svg width="{width}" height="{height}" '
            f'version="1.1" baseProfile="full" '
            f'xmlns="http://www.w3.org/2000/svg">',
            f'<g transform="translate(0,{height}) scale(1,-1)">',
        ]
        lines += [path.to_svg() for path in self.paths]
        lines.append("</g></svg>")
        return "\n".join(lines)

    def write_to_svg(self, path: str, width, height):
        with open(path, "w") as file:
            file.write(self.to_svg(width, height))

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
