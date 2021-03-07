import numpy as np

from .. import Shape, logic, utility


class Scene(Shape):
    def __init__(self):
        super().__init__()
        self.shapes = []
        self.tree = None

    def compile(self):
        for shape in self.shapes:
            shape.compile()

        if self.tree is None:
            self.tree = logic.Tree.from_shapes(self.shapes)

    def add(self, shape):
        self.shapes.append(shape)

    def intersect(
        self, ray_origin: np.ndarray, ray_direction: np.ndarray
    ) -> logic.Hit:
        return self.tree.intersect(ray_origin, ray_direction)

    def visible(self, eye: np.ndarray, point: np.ndarray) -> bool:
        v = eye - point
        hit = self.intersect(point, utility.vector_normalize(v))
        return hit.t >= utility.vector_length(v)

    def paths(self) -> logic.Paths:
        result = []
        for shape in self.shapes:
            result.extend(shape.paths().paths)
        return logic.Paths(result)

    def render(
        self,
        eye: np.ndarray,
        center: np.ndarray,
        up: np.ndarray,
        width,
        height,
        fovy,
        near,
        far,
        step,
    ) -> logic.Paths:
        aspect = width / height
        matrix = utility.matrix_look_at(eye, center, up)
        matrix = utility.matrix_mul_matrix(
            utility.matrix_perspective_projection(fovy, aspect, near, far),
            matrix,
        )
        return self.render_with_matrix(matrix, eye, width, height, step)

    def render_with_matrix(
        self, matrix: np.ndarray, eye: np.ndarray, width, height, step
    ) -> logic.Paths:
        self.compile()
        paths = self.paths()
        if step > 0:
            paths = paths.chop(step)
        paths = paths.filter(logic.ClipFilter(matrix, eye, self))
        if step > 0:
            paths = paths.simplify(1e-6)
        translation = utility.vector_translate(np.array([1.0, 1.0, 0.0]))
        scale = utility.vector_scale([width / 2, height / 2, 0])
        matrix = utility.matrix_mul_matrix(scale, translation)
        return paths.transform(matrix)
