import typing as ty

import numba as nb
import numpy as np

EPS = 1e-9
INF = 1e9


@nb.njit("float64(float64[:])", cache=True)
def vector_length(v: np.ndarray) -> float:
    return np.sqrt(np.sum(v ** 2.0, axis=-1))


@nb.njit("float64(float64[:])", cache=True)
def vector_squared_length(v: np.ndarray) -> float:
    return np.sum(v ** 2.0, axis=-1)


@nb.njit("float64[:](float64[:])", cache=True)
def vector_normalize(v: np.ndarray) -> np.ndarray:
    length = vector_length(v)
    return v / length


@nb.njit("float64[:,:](float64[:])", cache=True)
def vector_translate(vector: np.ndarray) -> np.ndarray:
    mat = np.identity(4)
    mat[3, 0:3] = vector[:3]
    return mat


@nb.njit("float64[:,:](float64[:], float64)", cache=True)
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


@nb.njit("float64[:](float64[:,:], float64[:])", cache=True)
def matrix_mul_position_vector(
    matrix: np.ndarray, vector: np.ndarray
) -> np.ndarray:
    output = np.array([vector[0], vector[1], vector[2], 1.0])
    output = np.dot(output, matrix)
    output /= output[3]
    return output[:3]


@nb.njit("float64[:](float64[:,:], float64[:])", cache=True)
def matrix_mul_direction_vector(
    matrix: np.ndarray, vector: np.ndarray
) -> np.ndarray:
    return vector_normalize(np.dot(vector, matrix[:3, :3]))


@nb.njit(
    "Tuple((float64[:], float64[:]))(float64[:,:], float64[:], float64[:])",
    cache=True,
)
def matrix_mul_ray(
    matrix: np.ndarray, ray_origin: np.ndarray, ray_direction: np.ndarray
) -> ty.Tuple[np.ndarray, np.ndarray]:
    return (
        matrix_mul_position_vector(matrix, ray_origin),
        matrix_mul_direction_vector(matrix, ray_direction),
    )


@nb.njit("float64[:,:](float64[:], float64[:], float64[:])", cache=True)
def matrix_look_at(eye, target, up):
    # from Pyrr
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


@nb.njit(
    "float64[:,:](float64, float64, float64, float64, float64, float64)",
    cache=True,
)
def frustum(left, right, bottom, top, near, far):
    # from Pyrr
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


@nb.njit("float64[:,:](float64, float64, float64, float64)", cache=True)
def matrix_perspective_projection(fovy, aspect, near, far):
    # from Pyrr
    ymax = near * np.tan(fovy * np.pi / 360.0)
    xmax = ymax * aspect
    return frustum(-xmax, xmax, -ymax, ymax, near, far)


@nb.njit(
    "Tuple((float64[:], float64[:]))(float64[:,:], float64[:], float64[:])",
    cache=True,
)
def matrix_mul_box(
    matrix: np.ndarray, min_box: np.ndarray, max_box: np.ndarray
) -> ty.Tuple[np.ndarray, np.ndarray]:
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


@nb.njit("float64[:,:](float64[:,:], float64[:,:])", cache=True)
def matrix_mul_matrix(m1, m2):
    return np.dot(m2, m1)


@nb.njit("float64[:,:](float64[:,:])", cache=True)
def matrix_inverse(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.inv(matrix)


@nb.njit("float64(float64[:], float64[:], float64[:])", cache=True)
def segment_distance(p: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    l2 = vector_squared_length(v - w)
    if l2 == 0:
        return vector_length(p - v)
    t = np.dot((p - v), (w - v)) / l2
    if t < 0:
        return vector_length(p - v)
    if t > 1:
        return vector_length(p - w)
    return vector_length((v + ((w - v) * t)) - p)


@nb.njit("float64[:]()", cache=True)
def random_unit_vector():
    while True:
        v = np.random.random(3) * 2 - 1
        if vector_squared_length(v) > 1:
            continue
        return vector_normalize(v)


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
    matrix_mul_ray(m, v - 0.5, v + 0.5)
    segment_distance(v, v + 5, v - 10)
    random_unit_vector()
