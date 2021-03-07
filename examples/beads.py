import numba as nb
import numpy as np
import pyln


@nb.njit(cache=True)
def low_pass(values: np.ndarray, alpha: float):
    result = np.zeros_like(values)
    y = 0.0
    for i in range(values.shape[0]):
        y -= alpha * (y - values[i])
        result[i] = y
    return result


@nb.njit(cache=True)
def normalize(values: np.ndarray, a: float, b: float):
    lo = np.amin(values)
    hi = np.amax(values)
    return ((values - lo) / (hi - lo)) * (b - a) + a


@nb.njit(cache=True)
def low_pass_noise(num: int, alpha: float, iterations: int):
    result = np.random.random(num)
    for i in range(iterations):
        result = low_pass(result, alpha)
    return normalize(result, -1, 1)


def main():
    pyln.utility.compile_numba()
    # define camera parameters
    eye = np.array([8.0, 8.0, 8.0])  # camera position
    center = np.array([0.0, 0.0, 0.0])  # camera looks at
    up = np.array([0.0, 0.0, 1.0])  # up direction

    scene = pyln.Scene()
    position = np.zeros(3)
    for a in range(10):
        num = 100
        values = np.array([low_pass_noise(num, 0.3, 4) for _ in range(4)])
        for i in range(num):
            scene.add(pyln.OutlineSphere(eye, up, position, 0.1))
            s = (values[-1, i] + 1) / 2 * 0.1 + 0.01
            position += pyln.utility.vector_normalize(values[:3, i]) * s

    # define rendering parameters
    width = 380.0 * 5  # rendered width
    height = 315.0 * 5  # rendered height
    fovy = 50.0  # vertical field of view, degrees
    znear = 0.1  # near z plane
    zfar = 100.0  # far z plane
    step = 0.01  # how finely to chop the paths for visibility testing
    # compute 2D paths that depict the 3D scene
    paths = scene.render(
        eye, center, up, width, height, fovy, znear, zfar, step
    )

    # save results
    paths.write_to_svg(
        "examples/images/beads.svg", width, height, background_color="white"
    )


if __name__ == "__main__":
    main()
