import numba as nb
import numpy as np

EPS = 1e-9
INF = 1e9


class Ray:
    def __init__(self, origin: np.ndarray, direction: np.ndarray):
        self.origin = origin
        self.direction = direction

    def position(self, t) -> np.ndarray:
        return self.origin + (t * self.direction)

    def __str__(self):
        return f"O: {self.origin}, D: {self.direction}"

    def mul_matrix(self, matrix):
        return Ray(
            matrix_mul_position_vector(matrix, self.origin),
            matrix_mul_direction_vector(matrix, self.direction),
        )


def compile_numba():
    v = np.random.random(3)
    vector_normalize(v)
    vector_length(v)
    vector_squared_length(v)
    m = vector_translate(v)
    m1 = vector_rotate(v, 0.5)
    matrix_mul_position_vector(m, v)
    matrix_inverse(m)
    matrix_mul_matrix(m, m1)
    matrix_mul_box(m, v - 0.5, v + 0.5)
    matrix_look_at(v, v - 1.0, v + 5.0)
    matrix_perspective_projection(50.0, 1.0, 0.1, 100.0)


@nb.njit
def vector_translate(vector: np.ndarray) -> np.ndarray:
    mat = np.identity(4)
    # mat[:3, -1] = vector[:3]
    mat[3, 0:3] = vector[:3]
    return mat


@nb.njit
def vector_rotate(vector: np.ndarray, a: float) -> np.ndarray:
    vector = vector_normalize(vector)
    sin, cos = np.sin(a), np.cos(a)
    m = 1 - cos
    return np.array(
        [
            [
                m * vector[0] * vector[0] + cos,
                m * vector[0] * vector[1] + vector[2] * sin,
                m * vector[2] * vector[0] - vector[1] * sin,
                0.0,
            ],
            [
                m * vector[0] * vector[1] - vector[2] * sin,
                m * vector[1] * vector[1] + cos,
                m * vector[1] * vector[2] + vector[0] * sin,
                0.0,
            ],
            [
                m * vector[2] * vector[0] + vector[1] * sin,
                m * vector[1] * vector[2] - vector[0] * sin,
                m * vector[2] * vector[2] + cos,
                0.0,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def vector_scale(vector):
    return np.diagflat([vector[0], vector[1], vector[2], 1.0])


@nb.njit
def vector_length(v: np.ndarray):
    return np.sqrt(np.sum(v ** 2.0, axis=-1))


@nb.njit
def vector_squared_length(v: np.ndarray):
    return np.sum(v ** 2.0, axis=-1)


@nb.njit
def vector_normalize(v):
    length = vector_length(v)
    return v / length


@nb.njit
def matrix_mul_position_vector(matrix, vector, normalize=False) -> np.ndarray:
    output = np.array(
        [vector[0], vector[1], vector[2], 1.0], dtype=vector.dtype
    )
    output = np.dot(output, matrix)
    if normalize:
        output /= output[3]
    return output[:3]


@nb.njit
def matrix_mul_direction_vector(matrix, vector) -> np.ndarray:
    x = (
        matrix[0, 0] * vector[0]
        + matrix[0, 1] * vector[1]
        + matrix[0, 2] * vector[2]
    )
    y = (
        matrix[1, 0] * vector[0]
        + matrix[1, 1] * vector[1]
        + matrix[1, 2] * vector[2]
    )
    z = (
        matrix[2, 0] * vector[0]
        + matrix[2, 1] * vector[1]
        + matrix[2, 2] * vector[2]
    )
    return vector_normalize(np.array([x, y, z]))


@nb.njit
def matrix_look_at(eye, target, up):
    # up = vector_normalize(up)
    # forward = vector_normalize(center - eye)
    # side = vector_normalize(np.cross(forward, up))
    # up = vector_normalize(np.cross(side, forward))
    #
    # return matrix_inverse(np.array(
    #     (
    #         (side[0], up[0], -forward[0], eye[0]),
    #         (side[1], up[1], -forward[1], eye[1]),
    #         (side[2], up[2], -forward[2], eye[2]),
    #         (0., 0., 0., 1.0),
    #     )
    # ))
    eye = np.asarray(eye)
    target = np.asarray(target)
    up = np.asarray(up)

    forward = vector_normalize(target - eye)
    side = vector_normalize(np.cross(forward, up))
    up = vector_normalize(np.cross(side, forward))

    return np.array(
        (
            (side[0], up[0], -forward[0], 0.0),
            (side[1], up[1], -forward[1], 0.0),
            (side[2], up[2], -forward[2], 0.0),
            (-np.dot(side, eye), -np.dot(up, eye), np.dot(forward, eye), 1.0),
        ),
        dtype=eye.dtype,
    )


@nb.njit
def matrix_perspective_projection(fovy, aspect, near, far):
    ymax = near * np.tan(fovy * np.pi / 360.0)
    xmax = ymax * aspect
    return frustum(-xmax, xmax, -ymax, ymax, near, far)


@nb.njit
def frustum(left, right, bottom, top, near, far):
    # t1 = 2. * n
    # t2 = r - l
    # t3 = t - b
    # t4 = f - n
    # return np.array(
    #     (
    #         (t1 / t2, 0., (r + l) / t2, 0.),
    #         (0., t1 / t3, (t + b) / t3, 0.),
    #         (0., 0., (-f - n) / t4, (-t1 * f) / t4),
    #         (0., 0., -1., 0.),
    #     )
    # )
    A = (right + left) / (right - left)
    B = (top + bottom) / (top - bottom)
    C = -(far + near) / (far - near)
    D = -2.0 * far * near / (far - near)
    E = 2.0 * near / (right - left)
    F = 2.0 * near / (top - bottom)

    return np.array(
        (
            (E, 0.0, 0.0, 0.0),
            (0.0, F, 0.0, 0.0),
            (A, B, C, -1.0),
            (0.0, 0.0, D, 0.0),
        )
    )


@nb.njit
def matrix_mul_box(
    matrix: np.ndarray, min_box: np.ndarray, max_box: np.ndarray
) -> (np.ndarray, np.ndarray):
    # http://dev.theomader.com/transform-bounding-boxes/
    r = matrix[:3, 0]
    u = matrix[:3, 1]
    b = matrix[:3, 2]
    t = matrix[:3, 3]
    xa, xb = (
        np.minimum(r * min_box[0], r * max_box[0]),
        np.maximum(r * min_box[0], r * max_box[0]),
    )
    ya, yb = (
        np.minimum(u * min_box[1], u * max_box[1]),
        np.maximum(u * min_box[1], u * max_box[1]),
    )
    za, zb = (
        np.minimum(b * min_box[2], b * max_box[2]),
        np.maximum(b * min_box[2], b * max_box[2]),
    )
    return xa + ya + za + t, xb + yb + zb + t


@nb.njit
def matrix_mul_matrix(m1, m2):
    return np.dot(m2, m1)


@nb.njit
def matrix_inverse(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.inv(matrix)
