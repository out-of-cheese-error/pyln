import typing as ty

import numpy as np

from .. import utility
from ..paths import ClipFilter, Paths
from ..shape import Shape
from ..tree import Tree


class Scene(Shape):
    def __init__(self):
        super().__init__()
        self.shapes = []
        self.tree = None

    def compile(self):
        for shape in self.shapes:
            shape.compile()

        if self.tree is None:
            self.tree = Tree.from_shapes(self.shapes)

    def add(self, shape):
        self.shapes.append(shape)

    def intersect(
        self, ray_origin: np.ndarray, ray_direction: np.ndarray
    ) -> float:
        return self.tree.intersect(ray_origin, ray_direction)

    def visible(self, eye: np.ndarray, point: np.ndarray) -> bool:
        v = eye - point
        hit = self.intersect(point, utility.vector_normalize(v))
        return hit >= utility.vector_length(v)

    def paths(self) -> Paths:
        result = []
        for shape in self.shapes:
            result.extend(shape.paths().paths)
        return Paths(result)

    def render(
        self,
        eye: ty.Union[ty.List[float], np.ndarray],
        center: ty.Union[ty.List[float], np.ndarray],
        up: ty.Union[ty.List[float], np.ndarray],
        width,
        height,
        fovy,
        near,
        far,
        step,
        debug=False,
    ) -> Paths:
        eye = np.asarray(eye, dtype=np.float64)
        center = np.asarray(center, dtype=np.float64)
        up = np.asarray(up, dtype=np.float64)
        aspect = width / height
        matrix = utility.matrix_look_at(eye, center, up)
        matrix = utility.matrix_mul_matrix(
            utility.matrix_perspective_projection(fovy, aspect, near, far),
            matrix,
        )
        return self.render_with_matrix(matrix, eye, width, height, step, debug)

    def render_with_matrix(
        self,
        matrix: np.ndarray,
        eye: np.ndarray,
        width,
        height,
        step,
        debug=False,
    ) -> Paths:
        self.compile()
        paths = self.paths()
        if debug:
            print(f"Starting with {len(paths.paths)} paths")
        if step > 0:
            paths = paths.chop(step)
            if debug:
                print(f"After chopping: {len(paths.paths)} paths")
        paths = paths.filter(ClipFilter(matrix, eye, self))
        if debug:
            print(f"After filtering: {len(paths.paths)} paths")
        if step > 0:
            paths = paths.simplify(1e-6)
            if debug:
                print(f"After simplifying {len(paths.paths)} paths")
        matrix = utility.matrix_transform(
            [
                (utility.Transform.Scale, [width / 2, height / 2, 0]),
                (utility.Transform.Translate, [1, 1, 0]),
            ],
        )
        return paths.transform(matrix)
