import numpy as np
import pyln


def main():
    pyln.utility.compile_numba()
    # define camera parameters
    eye = np.array([8, 8, 8], dtype=np.float64)  # camera position
    center = np.array([0, 0, 0], dtype=np.float64)  # camera looks at
    up = np.array([0, 0, 1], dtype=np.float64)  # up direction
    scene = pyln.Scene()
    n = 10
    for x in np.arange(-n, n):
        for y in np.arange(-n, n):
            scene.add(
                pyln.OutlineSphere(
                    eye, up, np.array([x, y, np.random.random()]), 0.45
                )
            )

    # define rendering parameters
    width = 1920  # rendered width
    height = 1200  # rendered height
    fovy = 50.0  # vertical field of view, degrees
    znear = 0.1  # near z plane
    zfar = 100.0  # far z plane
    step = 0.01  # how finely to chop the paths for visibility testing

    # compute 2D paths that depict the 3D scene
    paths = scene.render(
        eye, center, up, width, height, fovy, znear, zfar, step
    )

    # save results
    paths.write_to_svg("examples/images/outline.svg", width, height)


if __name__ == "__main__":
    main()
